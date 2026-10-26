import os
import re
import json
import math
import wave
import queue
import struct
import threading
import subprocess
import time

import numpy as np

from config import (
    TTS_ENGINE, TTS_THREADS, SPEECH_SPEED, KOKORO_MODEL, KOKORO_VOICES, KOKORO_VOICE,
    PIPER_PATH, PIPER_MODEL, PIPER_NOISE_SCALE, PIPER_NOISE_W, TEMP_DIR,
)

# Mic stays muted this long after the last audio finishes (room echo / aplay tail)
SPEAKING_TAIL_S = 0.5


# ── VOICES ────────────────────────────────────────────────────────────────────
# Each voice is loaded once and turns one sentence into (16-bit mono PCM bytes, sample rate).

def _onnx_session(model_path, threads):
    """ONNX Runtime session limited to `threads` cores — the LLM needs the rest."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level   = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # idle threads sleep, not spin
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


class KokoroVoice:
    """Kokoro-82M: the most natural-sounding voice, but slower than Piper on the Pi."""

    def __init__(self):
        from kokoro_onnx import Kokoro
        for path in (KOKORO_MODEL, KOKORO_VOICES):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        try:
            self._kokoro = Kokoro.from_session(_onnx_session(KOKORO_MODEL, TTS_THREADS), KOKORO_VOICES)
        except AttributeError:   # from_session relies on a private onnxruntime attribute
            print("[Speech] Kokoro: can't limit its CPU threads with this onnxruntime version")
            self._kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        if KOKORO_VOICE not in self._kokoro.get_voices():
            raise ValueError(f"unknown Kokoro voice '{KOKORO_VOICE}' (British: bm_george, bm_lewis, "
                             f"bm_daniel, bm_fable, bf_emma, bf_isabella, bf_alice, bf_lily)")
        self._style = self._kokoro.get_voice_style(KOKORO_VOICE)
        self._lang  = "en-gb" if KOKORO_VOICE[:1] == "b" else "en-us"   # bm_/bf_ are British
        self._speed = min(2.0, max(0.5, SPEECH_SPEED))
        self.name   = f"Kokoro {KOKORO_VOICE}"
        self.synth("Ready.")     # the first sentence is ~2.5x slower; pay it now

    def _create(self, text, speed):
        try:
            audio, rate = self._kokoro.create(text, voice=self._style, speed=speed, lang=self._lang)
        except ValueError:   # kokoro-onnx 0.6.1: silence trimming fails when the audio is NaN (below)
            audio, rate = self._kokoro.create(text, voice=self._style, speed=speed, lang=self._lang, trim=False)
        return audio, rate

    def synth(self, text):
        # The int8 model returns all-NaN audio for some one-word sentences ("Sixth."), more often at
        # speed != 1. Retry at normal speed; if it's still broken, raise so espeak says this sentence.
        audio, rate = self._create(text, self._speed)
        if not np.isfinite(audio).all() and self._speed != 1.0:
            audio, rate = self._create(text, 1.0)
        if not np.isfinite(audio).all():
            raise RuntimeError("Kokoro produced invalid audio for this sentence")
        # Kokoro is quiet (peaks ~0.35): bring each sentence up to a consistent level
        peak  = float(np.max(np.abs(audio))) if audio.size else 0.0
        gain  = min(3.0, 0.9 / peak) if peak > 0 else 1.0
        return (np.clip(audio * gain, -1.0, 1.0) * 32767).astype("<i2").tobytes(), int(rate)


class PiperVoice:
    """Piper (piper-tts) running in-process, so the voice loads once instead of per sentence."""

    def __init__(self):
        from piper import PiperConfig, PiperVoice as Voice, SynthesisConfig
        with open(f"{PIPER_MODEL}.json", encoding="utf-8") as f:
            config = PiperConfig.from_dict(json.load(f))
        self._voice = Voice(session=_onnx_session(PIPER_MODEL, TTS_THREADS), config=config)
        # Slightly slower with more variation than the defaults (1.0 / 0.667 / 0.8) — less robotic
        self._config = SynthesisConfig(length_scale=1.05 / SPEECH_SPEED,
                                       noise_scale=PIPER_NOISE_SCALE, noise_w_scale=PIPER_NOISE_W)
        self.name = f"Piper {os.path.basename(PIPER_MODEL)[:-5]}"

    def synth(self, text):
        pcm = b"".join(chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._config))
        return pcm, self._voice.config.sample_rate


class PiperBinaryVoice:
    """The standalone Piper binary from earlier setups (one process per sentence)."""

    def __init__(self):
        if not (os.path.exists(PIPER_PATH) and os.path.exists(PIPER_MODEL)):
            raise FileNotFoundError(PIPER_PATH)
        with open(f"{PIPER_MODEL}.json", encoding="utf-8") as f:
            self._rate = json.load(f)["audio"]["sample_rate"]
        self.name = "Piper (binary)"

    def synth(self, text):
        r = subprocess.run(
            [PIPER_PATH, "--model", PIPER_MODEL, "--output_raw", "--length_scale", str(1.05 / SPEECH_SPEED),
             "--noise_scale", str(PIPER_NOISE_SCALE), "--noise_w", str(PIPER_NOISE_W)],
            input=text.encode(), capture_output=True, timeout=60
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="ignore")[-300:])
        return r.stdout, self._rate


def load_voice(notice):
    """Configured voice first, then the faster/simpler ones. None means espeak-ng."""
    order = [KokoroVoice, PiperVoice, PiperBinaryVoice]
    if TTS_ENGINE == "piper":
        order.remove(KokoroVoice)
    reason = None    # why the preferred voice failed
    for cls in order:
        try:
            t0 = time.time()
            voice = cls()
            print(f"[Speech] Voice: {voice.name} (loaded in {time.time() - t0:.1f}s)")
            if reason:
                notice(f"Preferred voice not available ({reason}) — speaking with {voice.name}.")
            return voice
        except Exception as e:
            print(f"[Speech] {cls.__name__} unavailable: {type(e).__name__}: {e}")
            if reason is None:
                missing = isinstance(e, (ImportError, FileNotFoundError))
                reason = "not installed — run ./setup.sh" if missing else f"{type(e).__name__}: {e}"
    notice(f"No neural voice available ({reason}) — using the espeak-ng fallback voice.")
    return None


# ── SPEAKER ───────────────────────────────────────────────────────────────────
class Speaker:
    """
    Two-stage TTS pipeline so synthesis time is hidden:
      synth thread  — renders sentence N+1
      player thread — plays sentence N with aplay
    The voice loads in the synth thread at startup; espeak-ng is the last-resort fallback.

    is_speaking() lets the voice listener mute the mic while Jarvis talks,
    and stop() cuts speech off immediately (TALK button / Escape).
    """

    def __init__(self, notice=print):
        self.notice  = notice
        self.voice   = None
        self.text_q  = queue.Queue()
        self.pcm_q   = queue.Queue(maxsize=2)   # synthesise at most two sentences ahead
        self.lock    = threading.Lock()
        self.epoch   = 0          # bumped by stop(); stale items are dropped
        self.pending = 0          # sentences queued but not finished playing
        self.last_end    = 0.0
        self.player_proc = None
        self.beep_path   = os.path.join(TEMP_DIR, "jarvis_beep.wav")
        self._write_beep()

        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop,  daemon=True).start()

    # ── PUBLIC ────────────────────────────────────────────────────────────────
    def say(self, sentence: str):
        """Queue one sentence. Returns immediately."""
        with self.lock:
            self.pending += 1
            epoch = self.epoch
        self.text_q.put((epoch, sentence))

    def say_text(self, text: str):
        """Queue a whole reply, split into sentences."""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence.strip():
                self.say(sentence.strip())

    def is_speaking(self) -> bool:
        return self.pending > 0 or time.time() - self.last_end < SPEAKING_TAIL_S

    def stop(self):
        with self.lock:
            self.epoch += 1
            self.pending  = 0             # older sentences no longer count (see _done)
            self.last_end = time.time()
        for q in (self.text_q, self.pcm_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        proc = self.player_proc
        if proc:
            proc.terminate()

    def beep(self):
        """Short tone so voice users know Jarvis is listening."""
        try:
            subprocess.run(["aplay", "-q", self.beep_path], stderr=subprocess.DEVNULL, timeout=2)
        except Exception:
            pass

    # ── WORKERS ───────────────────────────────────────────────────────────────
    def _synth_loop(self):
        self.voice = load_voice(self.notice)
        while True:
            epoch, text = self.text_q.get()
            if epoch != self.epoch:
                continue
            pcm, rate = None, 0          # None = speak with espeak-ng
            if not re.search(r"[A-Za-z0-9]", text):
                pcm = b""                # nothing speakable (e.g. "...")
            elif self.voice:
                try:
                    pcm, rate = self.voice.synth(text)
                except Exception as e:
                    print(f"[Speech] {self.voice.name} failed: {type(e).__name__}: {e}")
            if epoch == self.epoch:      # stopped while synthesising -> drop it
                self.pcm_q.put((epoch, pcm, rate, text))

    def _play_loop(self):
        while True:
            epoch, pcm, rate, text = self.pcm_q.get()
            if epoch == self.epoch:
                if pcm:
                    cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"]
                elif pcm is None:
                    cmd = ["espeak-ng", "-v", "en-gb", "-s", "145", "-p", "42", text]
                else:
                    cmd = None    # nothing speakable (e.g. "...")
                try:
                    if cmd:
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if pcm else None,
                                                stderr=subprocess.DEVNULL)
                        self.player_proc = proc
                        if epoch != self.epoch:   # stop() ran while it was starting
                            proc.terminate()
                        proc.communicate(pcm or None)
                except Exception as e:
                    print(f"[Speech] Playback failed ({cmd[0]}): {e}")
                finally:
                    self.player_proc = None
            self._done(epoch)

    def _done(self, epoch):
        with self.lock:
            if epoch != self.epoch:       # stop() already reset the count
                return
            self.pending = max(0, self.pending - 1)
            if self.pending == 0:
                self.last_end = time.time()

    def _write_beep(self):
        rate, secs, freq = 16000, 0.12, 880
        frames = b"".join(
            struct.pack("<h", int(9000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(rate * secs))
        )
        try:
            with wave.open(self.beep_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(frames)
        except Exception:
            pass
