"""
Speech-to-text.

MoonshineSTT (default): the model stays loaded, ~0.5-1 s per command on a Pi 5, and is biased
towards "Jarvis" with keyterms. WhisperSTT: the previous whisper.cpp path, used if Moonshine
isn't installed and as the wake word fallback scanner.
"""
import os
import re
import subprocess
import time
import wave

import numpy as np

from config import STT_MODEL, STT_KEYTERMS, WHISPER_PATH, WHISPER_MODEL, WHISPER_CMD_MODEL, TEMP_AUDIO

# Recognisers emit these for silence/noise; never treat them as speech
NOISE_TAGS     = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "okay"}


def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", NOISE_TAGS.sub("", text)).strip()
    if text.lower().strip(" .!?,") in HALLUCINATIONS:
        return ""
    return text


class MoonshineSTT:
    SIZES = ("tiny", "small", "medium")

    def __init__(self, size=STT_MODEL, keyterms=STT_KEYTERMS):
        from moonshine_voice import ModelArch, Transcriber, get_model_for_language
        if size not in self.SIZES:
            raise ValueError(f"JARVIS_STT_MODEL must be one of {', '.join(self.SIZES)}, not '{size}'")
        arch = getattr(ModelArch, f"{size.upper()}_STREAMING")   # streaming models support keyterms
        # Downloads only if missing from ~/.cache/moonshine_voice; offline afterwards
        path, arch = get_model_for_language("en", arch)
        self._tr = Transcriber(path, arch)
        if keyterms:
            self._tr.set_keyterms(list(keyterms))
        self.name = f"Moonshine {size}"

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        t0 = time.time()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0   # bytes silently give ''
        # Fade in the first ~0.16 s: a hard start (background noise cut mid-way) makes Moonshine
        # invent a leading "Yeah" on many recordings; the fade removes it without losing words.
        fade = min(len(audio), 2560)
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        try:
            transcript = self._tr.transcribe_without_streaming(audio, 16000)
        except Exception as e:
            print(f"[Voice] Moonshine failed: {type(e).__name__}: {e}")
            return ""
        # Long audio can come back as several lines
        text = clean_transcript(" ".join(line.text.strip() for line in transcript.lines))
        print(f"[Voice] STT {self.name} {time.time() - t0:.1f}s: '{text}'")
        return text


def prefetch(size=STT_MODEL):
    """Download the Moonshine model now (used by setup.sh) so the first command doesn't wait for it."""
    from moonshine_voice import ModelArch, get_model_for_language
    path, _ = get_model_for_language("en", getattr(ModelArch, f"{size.upper()}_STREAMING"))
    print(f"Moonshine {size} model ready in {path}")


class WhisperSTT:
    def __init__(self, model_path):
        self.model_path = model_path
        self.name = os.path.basename(model_path)

    @classmethod
    def if_available(cls, model_path):
        if os.path.exists(WHISPER_PATH) and os.path.exists(model_path):
            return cls(model_path)
        return None

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        with wave.open(TEMP_AUDIO, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        t0 = time.time()
        try:
            result = subprocess.run(
                [WHISPER_PATH, "-m", self.model_path, "-f", TEMP_AUDIO, "-nt", "-np", "-l", "en", "-t", "4"],
                capture_output=True, text=True, timeout=60
            )
        except Exception as e:
            print(f"[Voice] Whisper failed to run: {e}")
            return ""
        if result.returncode != 0:
            print(f"[Voice] Whisper error ({result.returncode}): {result.stderr.strip()[-300:]}")
            return ""
        text = clean_transcript(result.stdout)
        print(f"[Voice] STT {self.name} {time.time() - t0:.1f}s: '{text}'")
        return text


def load_command_stt(notice):
    """Moonshine if installed, else whisper.cpp base.en (the previous setup). None if neither works."""
    try:
        t0 = time.time()
        print(f"[Voice] Loading Moonshine {STT_MODEL} (downloads once if missing)...")
        recogniser = MoonshineSTT()
        print(f"[Voice] {recogniser.name} ready in {time.time() - t0:.1f}s")
        return recogniser
    except Exception as e:
        print(f"[Voice] Moonshine unavailable: {type(e).__name__}: {e}")
        reason = "not installed — run ./setup.sh" if isinstance(e, ImportError) else f"{type(e).__name__}: {e}"
    recogniser = WhisperSTT.if_available(WHISPER_CMD_MODEL) or WhisperSTT.if_available(WHISPER_MODEL)
    if recogniser:
        notice(f"Moonshine speech recognition unavailable ({reason}) — using {recogniser.name}.")
    return recogniser
