"""
Voice input: microphone, speech detection, wake word and command capture.

One background thread (VoiceInput.run) owns the microphone and reads it in 80 ms chunks:
  - "hey jarvis" is spotted by a small wake word model (pyopen-wakeword) that costs ~1 ms per chunk,
    instead of transcribing every noise. Falls back to scanning phrases with Whisper if it's missing.
  - The command after the wake word (or the TALK button) is recorded until the speaker pauses,
    then transcribed once by stt.py.
"""
import re
import time
import difflib
import threading
from collections import deque

import numpy as np

from config import (
    WAKE_WORDS, WAKE_THRESHOLD, WHISPER_MODEL,
    COMMAND_PAUSE_S, COMMAND_MAX_S, COMMAND_WAIT_S,
)
import stt

RATE    = 16000
CHUNK   = 1280               # 80 ms — the wake word model's step size
CHUNK_S = CHUNK / RATE

WAKE_HISTORY_S = 1.5         # audio kept from before the wake word fires ("hey jarvis, what's…")
SCAN_PAUSE_S   = 0.7         # phrase end for the Whisper wake word fallback
SCAN_MAX_S     = 6

# "hey jarvis, what time is it" -> match ends after "jarvis, "
WAKE_RE = re.compile(
    r"\b(?:(?:hey|hi|ok|okay)[\s,]+)?(?:" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b[\s,.!?]*",
    re.I,
)


def after_wake_word(heard: str):
    """None if no wake word; otherwise the command around it ('' if only the wake word was said)."""
    m = WAKE_RE.search(heard)
    if not m:
        return None
    after = heard[m.end():].strip(" ,.!?")
    before = heard[:m.start()].strip(" ,.!?")
    if not after and len(before.split()) >= 3:   # "what time is it, jarvis"
        return before
    return after


def rms(chunk: bytes) -> float:
    x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


# ── MICROPHONE ────────────────────────────────────────────────────────────────
class MicStream:
    """16 kHz mono int16 microphone stream, read in CHUNK-sized pieces.
    Mics that can't record at 16 kHz are opened at their own rate and resampled."""

    def __init__(self):
        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._fmt = pyaudio.paInt16
        try:
            self.rate = RATE
            self._stream = self._open(RATE, CHUNK)
        except Exception:
            self.rate = int(self._pa.get_default_input_device_info()["defaultSampleRate"])
            self._stream = self._open(self.rate, self._native_frames())
        self._frames = CHUNK if self.rate == RATE else self._native_frames()

    def _native_frames(self) -> int:
        return int(round(CHUNK * self.rate / RATE))

    def _open(self, rate, frames):
        return self._pa.open(format=self._fmt, channels=1, rate=rate, input=True, frames_per_buffer=frames)

    def read(self) -> bytes:
        data = self._stream.read(self._frames, exception_on_overflow=False)
        return data if self.rate == RATE else resample(data, self.rate)

    def flush(self):
        """Drop audio that queued up while nobody was reading (beep, LLM thinking, our own voice)."""
        try:
            available = self._stream.get_read_available()
            if available > 0:
                self._stream.read(available, exception_on_overflow=False)
        except Exception:
            pass

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


def resample(chunk: bytes, rate: int) -> bytes:
    """Resample one chunk to CHUNK samples at 16 kHz (crude low-pass + interpolation; fine for speech)."""
    x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return bytes(CHUNK * 2)
    width = max(1, int(round(rate / RATE)))
    if width > 1:
        x = np.convolve(x, np.ones(width, dtype=np.float32) / width, mode="same")
    y = np.interp(np.linspace(0, x.size - 1, CHUNK), np.arange(x.size), x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


# ── SPEECH DETECTION ──────────────────────────────────────────────────────────
class SpeechGate:
    """Energy-based speech detector with an adaptive noise floor, so a fan or TV doesn't count as talking."""

    def __init__(self, ratio=2.5, min_rms=120.0):
        self.ratio   = ratio
        self.min_rms = min_rms
        self.floor   = None

    def calibrate(self, chunks):
        self.floor = float(np.median([rms(c) for c in chunks])) if chunks else None

    def threshold(self) -> float:
        return max((self.floor or 0.0) * self.ratio, self.min_rms)

    def is_speech(self, chunk: bytes) -> bool:
        level = rms(chunk)
        if self.floor is None:
            self.floor = level
        speech = level > self.threshold()
        # Follow the room's background level: quickly when quiet, very slowly during "speech"
        # so a noise that starts and never stops is eventually treated as background.
        rate = 0.002 if speech else 0.05
        self.floor += rate * (level - self.floor)
        return speech


def record_utterance(read, gate, wait_s, pause_s, max_s, preroll=(), started=False, timing=None) -> bytes:
    """Record one utterance. Returns PCM bytes, or b'' if nobody started speaking within wait_s.
    preroll/started: audio already captured mid-sentence (the words right after the wake word).
    timing: optional dict; gets "speech_end", the time.time() when the speaker stopped talking."""
    chunks  = list(preroll)
    before  = deque(maxlen=4)      # the 2 chunks that start speech + ~0.16 s before them (more makes Moonshine add "Yeah")
    voiced  = 0
    silent  = 0
    waited  = 0
    while True:
        chunk  = read()
        speech = gate.is_speech(chunk)
        if not started:
            before.append(chunk)
            waited += 1
            voiced = voiced + 1 if speech else 0
            if voiced >= 2:                    # 160 ms of speech, not just a click
                started = True
                chunks.extend(before)
            elif waited * CHUNK_S >= wait_s:
                return b""
            continue
        chunks.append(chunk)
        silent = 0 if speech else silent + 1
        if silent * CHUNK_S >= pause_s or len(chunks) * CHUNK_S >= max_s:
            break
    if timing is not None:                 # mic reads block in real time, so now = end of the audio
        timing["speech_end"] = time.time() - silent * CHUNK_S
    trailing = max(0, silent - 3)          # keep ~0.25 s of the closing silence
    if trailing:
        chunks = chunks[:-trailing]
    return b"".join(chunks)


class PhraseCollector:
    """Non-blocking phrase splitter for the Whisper wake word fallback: feed chunks, get whole phrases."""

    def __init__(self, gate, pause_s=SCAN_PAUSE_S, max_s=SCAN_MAX_S):
        self.gate, self.pause_s, self.max_s = gate, pause_s, max_s
        self.speech_end = 0.0      # time.time() when the last returned phrase's speaker stopped talking
        self.reset()

    def reset(self):
        self.before = deque(maxlen=4)
        self.chunks = []
        self.voiced = 0
        self.silent = 0

    def feed(self, chunk: bytes):
        speech = self.gate.is_speech(chunk)
        if not self.chunks:
            self.before.append(chunk)
            self.voiced = self.voiced + 1 if speech else 0
            if self.voiced >= 2:
                self.chunks = list(self.before)
                self.silent = 0
            return None
        self.chunks.append(chunk)
        self.silent = 0 if speech else self.silent + 1
        if self.silent * CHUNK_S >= self.pause_s or len(self.chunks) * CHUNK_S >= self.max_s:
            phrase = b"".join(self.chunks)
            self.speech_end = time.time() - self.silent * CHUNK_S
            self.reset()
            return phrase
        return None


# ── WAKE WORD ─────────────────────────────────────────────────────────────────
class WakeWord:
    """openWakeWord's "hey jarvis" model via pyopen-wakeword. Use from one thread only —
    the TFLite runtime crashes the process if two threads share an instance."""

    def __init__(self, threshold=WAKE_THRESHOLD, frames=2):
        from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures
        import pyopen_wakeword.openwakeword as oww_module
        # A missing model file segfaults instead of raising, so check the bundled files first
        models_dir = getattr(oww_module, "_MODELS_DIR", None)
        if models_dir is not None:
            for name in (Model.HEY_JARVIS.value, "melspectrogram", "embedding_model"):
                if not (models_dir / f"{name}.tflite").is_file():
                    raise FileNotFoundError(f"wake word model {name}.tflite missing")
        self._model    = OpenWakeWord.from_builtin(Model.HEY_JARVIS)   # ValueError/OSError if TFLite can't load
        self._features = OpenWakeWordFeatures.from_builtin()
        self.threshold = threshold
        self.frames    = frames    # consecutive 80 ms chunks above threshold
        self._run      = 0
        self.reset()

    def detect(self, chunk: bytes) -> bool:
        best = 0.0
        for emb in self._features.process_streaming(chunk):      # lazy generators: must be drained
            for prob in self._model.process_streaming(emb):
                best = max(best, prob)
        self._run = self._run + 1 if best >= self.threshold else 0
        if self._run >= self.frames:
            self._run = 0
            return True
        return False

    def reset(self):
        """Forget recent audio (e.g. our own voice). The features refill with ~8 s of virtual silence;
        passing that through the model now means it can detect again from the very next chunk."""
        self._model.reset()
        self._features.reset()
        self._run = 0
        for emb in self._features.process_streaming(bytes(CHUNK * 2)):
            for _ in self._model.process_streaming(emb):
                pass


# ── LISTENER ──────────────────────────────────────────────────────────────────
class VoiceInput:
    """Owns the microphone. Waits for the wake word (or TALK), records the command and passes
    the transcript to on_command, which runs in this thread (one command at a time)."""

    def __init__(self, speaker, on_command, on_status, notice, is_busy):
        self.speaker    = speaker
        self.on_command = on_command
        self.on_status  = on_status
        self.notice     = notice          # shown in the chat window
        self.is_busy    = is_busy         # True while a query is being answered
        self.ptt        = threading.Event()
        self.ready      = False           # listening
        self.failed     = False           # gave up (reason shown in the chat window)
        self.running    = True
        self.stt        = None            # command transcriber
        self.scan_stt   = None            # wake word fallback transcriber
        self.wake       = None
        self.gate       = SpeechGate()
        self.spoke_until = 0.0            # time.time() the last command's speaker stopped talking

    # ── setup ────────────────────────────────────────────────────────────────
    def _load(self) -> bool:
        self.stt = stt.load_command_stt(self.notice)
        if self.stt is None:
            self.notice("Voice input disabled: no speech recogniser (Moonshine or whisper.cpp). "
                        "Run ./setup.sh. You can still type.")
            return False
        try:
            self.wake = WakeWord()
            print(f"[Voice] Wake word: 'hey jarvis' model (threshold {self.wake.threshold})")
        except Exception as e:
            print(f"[Voice] Wake word model unavailable: {type(e).__name__}: {e}")
            reason = "not installed — run ./setup.sh" if isinstance(e, ImportError) else f"{type(e).__name__}: {e}"
            self.scan_stt = stt.WhisperSTT.if_available(WHISPER_MODEL) or self.stt
            self.notice(f"Wake word model unavailable ({reason}) — "
                        f"listening for 'Jarvis' with {self.scan_stt.name} instead.")
        return True

    def run(self):
        if not self._load():
            self.failed = True
            return
        lost = False          # the mic worked, then stopped (e.g. unplugged): keep trying to reopen it
        while self.running:
            mic = None
            try:
                mic = MicStream()
                self.gate.calibrate([mic.read() for _ in range(int(1 / CHUNK_S))])   # 1 s of room noise
            except Exception as e:
                if mic:
                    try:
                        mic.close()
                    except Exception:
                        pass
                if lost:
                    time.sleep(3)
                    continue
                self.failed = True
                self.notice(f"Voice input disabled: could not open the microphone ({e}). "
                            f"Check 'arecord -l'. You can still type.")
                return
            self.ready = True
            if lost:
                self.notice("Microphone reconnected.")
                lost = False
            print(f"[Voice] Listening (mic {mic.rate} Hz, speech threshold {self.gate.threshold():.0f}).")
            try:
                self._loop(mic)       # returns if the mic keeps failing
            finally:
                self.ready = False
                try:
                    mic.close()
                except Exception:
                    pass
            if self.running:
                lost = True
                self.notice("Microphone stopped working — trying to reconnect. You can still type.")
                time.sleep(3)

    # ── main loop ────────────────────────────────────────────────────────────
    def _loop(self, mic):
        history = deque(maxlen=int(WAKE_HISTORY_S / CHUNK_S))
        phrases = PhraseCollector(self.gate)
        resume  = True        # (re)start listening cleanly: drop stale audio, clear the wake model
        errors  = 0
        while self.running:
            try:
                if self.ptt.is_set():
                    if self.speaker.is_speaking():    # TALK while Jarvis talks: silence it first
                        self.speaker.stop()
                        while self.speaker.is_speaking():
                            mic.read()
                    self.ptt.clear()
                    self._command(mic, beep=True)
                    resume = True
                    continue

                # Half-duplex: keep draining the mic while Jarvis talks, but don't listen to it
                if self.speaker.is_speaking():
                    mic.read()
                    self.on_status("● SPEAKING", "#00d4ff")
                    resume = True
                    continue

                if resume:
                    mic.flush()
                    if self.wake:
                        self.wake.reset()
                    history.clear()
                    phrases.reset()
                    resume = False
                if not self.is_busy():
                    self.on_status("● STANDBY", "#00d4ff")

                chunk = mic.read()
                if self.wake:
                    history.append(chunk)
                    self.gate.is_speech(chunk)        # keeps the noise floor in step with the room
                    if self.wake.detect(chunk):
                        self._on_wake(mic, list(history))
                        resume = True
                else:
                    phrase = phrases.feed(chunk)
                    if phrase:
                        self.spoke_until = phrases.speech_end
                        self._on_phrase(mic, phrase)
                        resume = True
                errors = 0
            except Exception as e:
                print(f"[Voice] Loop error: {type(e).__name__}: {e}")
                errors += 1
                if errors >= 5:                       # the mic is probably gone: let run() reopen it
                    return
                time.sleep(1)
                resume = True

    def _on_wake(self, mic, history):
        print("[Voice] Wake word detected")
        self.on_status("● LISTENING", "#f85149")
        # Did they carry straight on ("hey jarvis, what time is it"), or wait for the beep?
        follow, quiet, spoke = [], 0, 0
        for i in range(12):                       # look at up to ~1 s
            chunk = mic.read()
            follow.append(chunk)
            if self.gate.is_speech(chunk):
                quiet = 0
                if i >= 2:                        # the first chunks can be the end of "jarvis"
                    spoke += 1
            else:
                quiet += 1
            if spoke >= 2 or quiet >= 8:          # talking again / ~0.6 s of silence
                break
        if spoke < 2:
            self._command(mic, beep=True)
            return

        timing = {}
        audio = record_utterance(mic.read, self.gate, COMMAND_WAIT_S, COMMAND_PAUSE_S, COMMAND_MAX_S,
                                 preroll=history + follow, started=True, timing=timing)
        self.spoke_until = timing["speech_end"]
        self.ptt.clear()                          # TALK pressed while we were already listening
        text = self._transcribe(audio)
        command = self._command_text(text)
        if command is None:
            print(f"[Voice] No wake word in '{text}' — ignored")
        elif command:
            self.on_command(command)
        else:
            self._command(mic, beep=True)         # only the wake word was said

    def _on_phrase(self, mic, phrase):
        """Whisper fallback: transcribe each phrase and look for the wake word in the text."""
        heard = self.scan_stt.transcribe(phrase)
        command = after_wake_word(heard)
        if command is None:
            return
        print(f"[Voice] Wake word heard: '{heard}'")
        if command and self.scan_stt is not self.stt:
            # Said in one breath — re-transcribe with the accurate model
            self.on_status("● TRANSCRIBING", "#d29922")
            precise  = self.stt.transcribe(phrase)
            stripped = after_wake_word(precise)
            if stripped:
                command = stripped
            elif stripped is None and precise:
                command = precise
        if command:
            self.on_command(command)
        else:
            self._command(mic, beep=True)

    def _command(self, mic, beep):
        self.on_status("● LISTENING", "#f85149")
        if beep:
            self.speaker.beep()
            mic.flush()                           # don't record the beep
        timing = {}
        audio = record_utterance(mic.read, self.gate, COMMAND_WAIT_S, COMMAND_PAUSE_S, COMMAND_MAX_S,
                                 timing=timing)
        self.spoke_until = timing.get("speech_end", time.time())
        self.ptt.clear()                          # TALK pressed while we were already listening
        if not audio:
            self.notice("I didn't hear a command.")
            return
        text = self._transcribe(audio)
        stripped = after_wake_word(text)          # drop a repeated "hey jarvis"
        if stripped is not None:
            text = stripped
        if text:
            self.on_command(text)
        else:
            self.notice("Sorry, I didn't catch that.")

    def _transcribe(self, audio: bytes) -> str:
        self.on_status("● TRANSCRIBING", "#d29922")
        return self.stt.transcribe(audio)

    @staticmethod
    def _command_text(text: str):
        """The command in 'hey jarvis, <command>'. '' if only the wake word was said,
        None if the transcript has no wake word at all (the detector misfired on other speech)."""
        command = after_wake_word(text)
        if command is not None or not text:
            return command or ""
        # The transcript may spell "Jarvis" differently ("Hey Java's, what's..."): look near the start
        words = text.split()
        for i, word in enumerate(words[:3]):
            if difflib.SequenceMatcher(None, word.lower().strip(",.!?"), "jarvis").ratio() >= 0.6:
                return " ".join(words[i + 1:]).strip(" ,.!?")
        return None
