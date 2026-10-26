import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext
import queue
import time
import datetime


def _silence_alsa_errors():
    """PyAudio probes every ALSA device at startup and libasound prints dozens of
    harmless 'Unknown PCM' errors. Install a no-op handler to keep the terminal readable."""
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
        handler_t = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        global _alsa_handler   # must stay referenced or ctypes frees it
        _alsa_handler = handler_t(lambda *args: None)
        cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_handler)
    except Exception:
        pass


_silence_alsa_errors()

import speech_recognition as sr
from ollama import Client

from config import (
    WHISPER_PATH, WHISPER_MODEL, WHISPER_CMD_MODEL, TEMP_AUDIO,
    PIPER_PATH, PIPER_MODEL, DB_PATH,
    WAKE_WORDS, AGENT_NAME, OLLAMA_HOST, MODEL, OPTS_CHAT
)
from database import JarvisDB
from agents import AgentPipeline
from speech import Speaker

ollama_client = Client(host=OLLAMA_HOST)

# "hey jarvis, what time is it" -> match ends after "jarvis, "
WAKE_RE = re.compile(
    r"\b(?:(?:hey|hi|ok|okay)[\s,]+)?(?:" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b[\s,.!?]*",
    re.I,
)
# Whisper emits these for silence/noise; never treat them as speech
NOISE_TAGS     = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "okay"}


def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", NOISE_TAGS.sub("", text)).strip()
    if text.lower().strip(" .!?,") in HALLUCINATIONS:
        return ""
    return text


def after_wake_word(heard: str):
    """None if no wake word; otherwise whatever was said after it ('' if nothing)."""
    m = WAKE_RE.search(heard)
    if not m:
        return None
    return heard[m.end():].strip(" ,.!?")


class JarvisApp:
    def __init__(self, root):
        self.root     = root
        self.root.title("J.A.R.V.I.S")
        self.root.configure(bg="#0a0a14")
        # Delay geometry until after window is mapped so we can read real screen height
        self.root.update_idletasks()
        screen_h = self.root.winfo_screenheight()
        win_h = min(480, screen_h - 50)   # leave 50px for taskbar
        self.root.geometry(f"800x{win_h}")

        self.ui_queue = queue.Queue()
        # One query at a time — concurrent requests just queue inside Ollama anyway
        self.pipeline_lock = threading.Lock()
        self.ptt_event     = threading.Event()   # TALK button pressed
        self.running       = True
        self.voice_ready   = False
        self.cmd_model     = WHISPER_MODEL
        self._last_status  = None

        # Init DB, pipeline and speech output
        self.db       = JarvisDB(DB_PATH)
        self.pipeline = AgentPipeline(self.db)
        self.speaker  = Speaker()

        self.build_ui()
        self.check_queue_loop()

        voice_ok = self._startup_checks()

        # Pre-warm model in background so first query has no cold-start delay
        threading.Thread(target=self._warm_model, daemon=True).start()

        # Start wake word listener
        if voice_ok:
            threading.Thread(target=self.wake_word_listener, daemon=True).start()

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def build_ui(self):
        # Title bar
        title_bar = tk.Frame(self.root, bg="#0a0a14")
        title_bar.pack(fill=tk.X, padx=15, pady=(10, 0))

        tk.Label(
            title_bar, text="J.A.R.V.I.S",
            bg="#0a0a14", fg="#00d4ff",
            font=("Courier", 18, "bold")
        ).pack(side=tk.LEFT)

        self.time_lbl = tk.Label(
            title_bar, text="",
            bg="#0a0a14", fg="#4a9eff",
            font=("Courier", 12)
        )
        self.time_lbl.pack(side=tk.RIGHT)
        self.tick_clock()

        # Input row — packed BEFORE chat so it reserves space first
        row = tk.Frame(self.root, bg="#0a0a14")
        row.pack(fill=tk.X, padx=15, pady=(0, 8), side=tk.BOTTOM)

        self.entry = tk.Entry(
            row, bg="#161b22", fg="#c9d1d9",
            insertbackground="white",
            font=("Courier", 13), bd=0,
            highlightthickness=1, highlightbackground="#30363d"
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", self.on_text_submit)
        self.root.bind("<Escape>", lambda e: self.speaker.stop())

        # Tap to talk (skips the wake word); also stops Jarvis mid-sentence
        tk.Button(
            row, text="TALK", command=self.on_talk_pressed,
            bg="#1f6feb", fg="white", activebackground="#388bfd", activeforeground="white",
            font=("Courier", 12, "bold"), bd=0, padx=14, pady=6, highlightthickness=0
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.status_lbl = tk.Label(
            row, text="● STANDBY",
            bg="#0a0a14", fg="#00d4ff",
            font=("Courier", 11, "bold"), padx=10
        )
        self.status_lbl.pack(side=tk.RIGHT, ipady=6)

        # Chat display — packed after input row so expand=True fills remaining space
        self.chat = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD,
            bg="#0d1117", fg="#c9d1d9",
            insertbackground="white",
            font=("Courier", 12), bd=0,
            highlightthickness=1, highlightbackground="#21262d"
        )
        self.chat.pack(padx=15, pady=(0, 4), fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)

        # Color tags per message source
        self.chat.tag_configure("jarvis", foreground="#00d4ff", font=("Courier", 12, "bold"))
        self.chat.tag_configure("cache",  foreground="#89b4fa", font=("Courier", 12, "bold"))
        self.chat.tag_configure("tool",   foreground="#a6e3a1", font=("Courier", 12, "bold"))
        self.chat.tag_configure("block",  foreground="#f38ba8", font=("Courier", 12, "bold"))
        self.chat.tag_configure("user",   foreground="#7ee787", font=("Courier", 12, "bold"))
        self.chat.tag_configure("system", foreground="#8b949e", font=("Courier", 11, "italic"))

    def tick_clock(self):
        self.time_lbl.config(text=datetime.datetime.now().strftime("%H:%M:%S  %d %b %Y"))
        self.root.after(1000, self.tick_clock)

    # ── UI HELPERS ────────────────────────────────────────────────────────────
    def print_msg(self, sender, text, tag="jarvis"):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}: ", tag)
        self.chat.insert(tk.END, f"{text}\n\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

    def append_text(self, text, tag=None):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

    def check_queue_loop(self):
        try:
            while True:
                kind, *args = self.ui_queue.get_nowait()
                if kind == "msg":
                    self.print_msg(*args)
                elif kind == "start":            # streamed reply: label first, text follows
                    sender, tag = args
                    self.append_text(f"{sender}: ", tag)
                elif kind == "append":
                    self.append_text(args[0])
                elif kind == "status":
                    text, color = args
                    self.status_lbl.config(text=text, fg=color)
                self.ui_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue_loop)

    def set_status(self, text, color="#d29922"):
        if (text, color) != self._last_status:
            self._last_status = (text, color)
            self.ui_queue.put(("status", text, color))

    def system_msg(self, text):
        print(f"[Jarvis] {text}")
        self.ui_queue.put(("msg", "System", text, "system"))

    # ── STARTUP CHECKS ────────────────────────────────────────────────────────
    def _startup_checks(self) -> bool:
        """Report missing pieces in the GUI instead of failing silently. Returns True if voice can work."""
        if not (os.path.exists(PIPER_PATH) and os.path.exists(PIPER_MODEL)):
            self.system_msg("Piper voice not found in ~/piper — using the espeak-ng fallback voice.")

        if not os.path.exists(WHISPER_PATH):
            self.system_msg(
                "Voice input disabled: whisper-cli not found. Build it with: "
                "cd ~/whisper.cpp && cmake -B build && cmake --build build -j4 --config Release"
            )
            return False
        if not os.path.exists(WHISPER_MODEL):
            self.system_msg(
                "Voice input disabled: whisper model missing. Run: "
                "bash ~/whisper.cpp/models/download-ggml-model.sh tiny.en"
            )
            return False

        self.cmd_model = WHISPER_CMD_MODEL if os.path.exists(WHISPER_CMD_MODEL) else WHISPER_MODEL
        print(f"[Jarvis] Wake word model: {os.path.basename(WHISPER_MODEL)}, "
              f"command model: {os.path.basename(self.cmd_model)}")
        return True

    # ── MODEL WARM-UP ─────────────────────────────────────────────────────────
    def _warm_model(self):
        """Load the model into RAM at startup to eliminate cold-start lag."""
        # Hold the pipeline lock so queries wait for the load and the listener keeps the status
        with self.pipeline_lock:
            self.set_status("● WARMING UP", "#d29922")
            t0 = time.time()
            try:
                # Same num_ctx/num_thread as the chat call, otherwise Ollama reloads on the first query
                ollama_client.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": "ready"}],
                    think=False,
                    keep_alive=-1,
                    options={**OPTS_CHAT, "num_predict": 1}
                )
                print(f"[Jarvis] Model {MODEL} warm in {time.time() - t0:.1f}s.")
            except Exception as e:
                self.system_msg(f"Could not load {MODEL} from Ollama: {e}. "
                                f"Check 'sudo systemctl status ollama' and 'ollama list'.")
            self.set_status("● STANDBY", "#00d4ff")

    # ── KEYBOARD INPUT ────────────────────────────────────────────────────────
    def on_text_submit(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.speaker.stop()
        self.ui_queue.put(("msg", "You", text, "user"))
        threading.Thread(target=self.run_pipeline, args=(text,), daemon=True).start()

    def on_talk_pressed(self):
        self.speaker.stop()
        if not self.voice_ready:
            self.system_msg("Voice input isn't available — see the messages above. Use the text box.")
            return
        self.set_status("● LISTENING", "#f85149")
        self.ptt_event.set()

    # ── VOICE INPUT ───────────────────────────────────────────────────────────
    def wake_word_listener(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.7   # seconds of silence that end a phrase

        try:
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                self.voice_ready = True
                print(f"[Jarvis] Listening for wake word (energy threshold {recognizer.energy_threshold:.0f}).")
                while self.running:
                    try:
                        self._listen_once(recognizer, source)
                    except Exception as e:
                        print(f"[Jarvis] Voice loop error: {type(e).__name__}: {e}")
                        time.sleep(1)
        except Exception as e:
            self.system_msg(f"Voice input disabled: could not open the microphone ({e}). "
                            f"Check 'arecord -l'. You can still type.")

    def _listen_once(self, recognizer, source):
        if self.ptt_event.is_set():
            self.ptt_event.clear()
            self.capture_command(recognizer, source)
            return

        # Half-duplex: don't listen while Jarvis is talking, or it hears itself
        if self.speaker.is_speaking():
            self.set_status("● SPEAKING", "#00d4ff")
            time.sleep(0.1)
            return

        if not self.pipeline_lock.locked():
            self.set_status("● STANDBY", "#00d4ff")

        t0 = time.time()
        try:
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            return
        if self.speaker.active_since(t0) or self.ptt_event.is_set():
            return   # captured our own voice, or the TALK button took over

        heard = self.transcribe(audio, WHISPER_MODEL)
        command = after_wake_word(heard)
        if command is None:
            return
        print(f"[Jarvis] Wake word heard: '{heard}'")

        if command:
            # Said in one breath ("hey jarvis what time is it") — re-transcribe with the accurate model
            if self.cmd_model != WHISPER_MODEL:
                self.set_status("● TRANSCRIBING", "#d29922")
                precise  = self.transcribe(audio, self.cmd_model)
                stripped = after_wake_word(precise)
                if stripped:                       # accurate model also heard the wake word
                    command = stripped
                elif stripped is None and precise: # it transcribed the wake word differently
                    command = precise
            self.handle_voice_command(command)
        else:
            self.capture_command(recognizer, source)

    def capture_command(self, recognizer, source):
        self.set_status("● LISTENING", "#f85149")
        self.speaker.beep()
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
        except sr.WaitTimeoutError:
            self.system_msg("I didn't hear a command.")
            return

        self.set_status("● TRANSCRIBING", "#d29922")
        text = self.transcribe(audio, self.cmd_model)
        stripped = after_wake_word(text)   # drop a repeated "hey jarvis"
        if stripped is not None:
            text = stripped
        if text:
            self.handle_voice_command(text)
        else:
            self.system_msg("Sorry, I didn't catch that.")

    def handle_voice_command(self, text):
        self.ui_queue.put(("msg", "You (Voice)", text, "user"))
        self.run_pipeline(text)

    def transcribe(self, audio, model_path) -> str:
        """Runs whisper.cpp on the captured audio. Returns cleaned text ('' on failure/noise)."""
        with open(TEMP_AUDIO, "wb") as f:
            f.write(audio.get_wav_data(convert_rate=16000, convert_width=2))   # whisper's native format
        t0 = time.time()
        try:
            result = subprocess.run(
                [WHISPER_PATH, "-m", model_path, "-f", TEMP_AUDIO, "-nt", "-np", "-l", "en", "-t", "4"],
                capture_output=True, text=True, timeout=60
            )
        except Exception as e:
            print(f"[Jarvis] Whisper failed to run: {e}")
            return ""
        if result.returncode != 0:
            print(f"[Jarvis] Whisper error ({result.returncode}): {result.stderr.strip()[-300:]}")
            return ""
        text = clean_transcript(result.stdout)
        print(f"[Jarvis] STT {os.path.basename(model_path)} {time.time() - t0:.1f}s: '{text}'")
        return text

    # ── PIPELINE BRIDGE ───────────────────────────────────────────────────────
    def run_pipeline(self, user_input: str):
        with self.pipeline_lock:
            try:
                self._run_pipeline(user_input)
            except Exception as e:
                print(f"[Jarvis] Pipeline error: {type(e).__name__}: {e}")
                self.ui_queue.put(("msg", "System", f"Error: {e}", "system"))
            finally:
                self.set_status("● STANDBY", "#00d4ff")

    def _run_pipeline(self, user_input: str):
        streamed = []

        def on_sentence(sentence):
            # LLM replies arrive sentence by sentence: show and speak each immediately
            if not streamed:
                self.ui_queue.put(("start", AGENT_NAME, "jarvis"))
                self.set_status("● SPEAKING", "#00d4ff")
            streamed.append(sentence)
            self.ui_queue.put(("append", sentence + " "))
            self.speaker.say(sentence)

        response, source, info = self.pipeline.run(
            user_input, on_status=self.set_status, on_sentence=on_sentence
        )

        if streamed:
            self.ui_queue.put(("append", "\n\n"))
            return

        tag_map = {
            "tool":   "tool",
            "cache":  "cache",
            "memory": "tool",
            "error":  "block",
        }
        label_map = {
            "tool":   f"{AGENT_NAME} [instant]",
            "cache":  f"{AGENT_NAME} [cached]",
            "memory": f"{AGENT_NAME} [memory]",
            "error":  f"{AGENT_NAME} [error]",
        }
        self.ui_queue.put((
            "msg",
            label_map.get(source, AGENT_NAME),
            response,
            tag_map.get(source, "jarvis")
        ))
        self.speaker.say_text(response)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = JarvisApp(root)
    root.mainloop()
