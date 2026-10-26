import os
import re
import math
import wave
import queue
import struct
import threading
import subprocess
import time

from config import PIPER_PATH, PIPER_MODEL, TEMP_DIR

# Mic stays muted this long after the last audio finishes (room echo / aplay tail)
SPEAKING_TAIL_S = 0.5


class Speaker:
    """
    Two-stage TTS pipeline so Piper's model-load time is hidden:
      synth thread  — renders sentence N+1 to a WAV with Piper
      player thread — plays sentence N with aplay
    Falls back to espeak-ng if Piper is missing or fails.

    is_speaking() lets the voice listener mute the mic while Jarvis talks,
    and stop() cuts speech off immediately (TALK button / Escape).
    """

    def __init__(self):
        self.text_q  = queue.Queue()
        self.wav_q   = queue.Queue(maxsize=2)   # with 5 rotating files, a file is never overwritten while queued/playing
        self.lock    = threading.Lock()
        self.epoch   = 0          # bumped by stop(); stale items are dropped
        self.pending = 0          # sentences queued but not finished playing
        self.last_end    = 0.0
        self.player_proc = None
        self.use_piper   = os.path.exists(PIPER_PATH) and os.path.exists(PIPER_MODEL)
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

    def active_since(self, t: float) -> bool:
        """True if any speech played at or after time t (used to discard self-heard audio)."""
        return self.pending > 0 or self.last_end + SPEAKING_TAIL_S > t

    def stop(self):
        with self.lock:
            self.epoch += 1
        for q in (self.text_q, self.wav_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                self._done()
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
        n = 0
        while True:
            epoch, text = self.text_q.get()
            if epoch != self.epoch:
                self._done()
                continue
            path = None
            if self.use_piper:
                path = os.path.join(TEMP_DIR, f"jarvis_tts_{n % 5}.wav")
                n += 1
                try:
                    r = subprocess.run(
                        [PIPER_PATH, "--model", PIPER_MODEL, "--output_file", path],
                        input=text.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60
                    )
                    if r.returncode != 0 or not os.path.getsize(path):
                        print(f"[Speech] Piper failed: {r.stderr.decode(errors='ignore')[-300:]}")
                        path = None
                except Exception as e:
                    print(f"[Speech] Piper error: {e}")
                    path = None
            self.wav_q.put((epoch, path, text))

    def _play_loop(self):
        while True:
            epoch, path, text = self.wav_q.get()
            if epoch == self.epoch:
                cmd = (["aplay", "-q", path] if path else
                       ["espeak-ng", "-v", "en-gb", "-s", "145", "-p", "42", text])
                try:
                    self.player_proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
                    self.player_proc.wait()
                except Exception as e:
                    print(f"[Speech] Playback failed ({cmd[0]}): {e}")
                finally:
                    self.player_proc = None
            self._done()

    def _done(self):
        with self.lock:
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
