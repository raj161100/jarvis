import re
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext
import queue
import time
import datetime

import speech_recognition as sr
from ollama import Client

from config import (
    WHISPER_PATH, WHISPER_MODEL, TEMP_AUDIO, TEMP_TTS,
    PIPER_PATH, PIPER_MODEL, DB_PATH,
    WAKE_WORDS, AGENT_NAME, OLLAMA_HOST, MODEL
)
from database import JarvisDB
from agents import AgentPipeline

ollama_client = Client(host=OLLAMA_HOST)


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
        self.tts_lock = threading.Lock()
        self.running  = True

        # Init DB and pipeline
        self.db       = JarvisDB(DB_PATH)
        self.pipeline = AgentPipeline(self.db)

        self.build_ui()
        self.check_queue_loop()

        # Pre-warm model in background so first query has no cold-start delay
        threading.Thread(target=self._warm_model, daemon=True).start()

        # Start wake word listener
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

    def check_queue_loop(self):
        try:
            while True:
                kind, *args = self.ui_queue.get_nowait()
                if kind == "msg":
                    self.print_msg(*args)
                elif kind == "status":
                    text, color = args
                    self.status_lbl.config(text=text, fg=color)
                self.ui_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue_loop)

    def set_status(self, text, color="#d29922"):
        self.ui_queue.put(("status", text, color))

    # ── TTS ───────────────────────────────────────────────────────────────────
    def speak(self, text):
        """Non-blocking — queues TTS via lock so only one plays at a time."""
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def _speak_blocking(self, text):
        with self.tts_lock:
            try:
                proc = subprocess.Popen(
                    [PIPER_PATH, "--model", PIPER_MODEL, "--output_file", TEMP_TTS],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                proc.communicate(input=text.encode())
                subprocess.run(["aplay", "-q", TEMP_TTS], stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                # Piper not set up yet — fall back to espeak-ng British voice
                subprocess.run(
                    ["espeak-ng", "-v", "en-gb", "-s", "145", "-p", "42", text],
                    stderr=subprocess.DEVNULL
                )

    def speak_sentences(self, text):
        """Split into sentences and speak each — user hears first sentence in ~3s."""
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            s = sentence.strip()
            if s:
                self._speak_blocking(s)

    # ── MODEL WARM-UP ─────────────────────────────────────────────────────────
    def _warm_model(self):
        """Load the model into RAM silently at startup to eliminate cold-start lag."""
        try:
            ollama_client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "ready"}],
                options={"num_predict": 1}
            )
            self.ui_queue.put(("status", "● STANDBY", "#00d4ff"))
        except Exception:
            pass

    # ── KEYBOARD INPUT ────────────────────────────────────────────────────────
    def on_text_submit(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.ui_queue.put(("msg", "You", text, "user"))
        threading.Thread(target=self.run_pipeline, args=(text,), daemon=True).start()

    # ── VOICE INPUT ───────────────────────────────────────────────────────────
    def wake_word_listener(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            mic = sr.Microphone()
        except Exception:
            self.ui_queue.put(("msg", "System", "No microphone detected. Voice input disabled. Use the text box.", "system"))
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                try:
                    self.set_status("● STANDBY", "#00d4ff")
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)

                    with open(TEMP_AUDIO, "wb") as f:
                        f.write(audio.get_wav_data())

                    result = subprocess.run(
                        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", TEMP_AUDIO, "-nt"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
                    )
                    heard = result.stdout.strip().lower()

                    if any(w in heard for w in WAKE_WORDS):
                        self.capture_command(recognizer, source)

                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    time.sleep(1)

    def capture_command(self, recognizer, source):
        self.set_status("● LISTENING", "#f85149")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            self.set_status("● PROCESSING", "#d29922")

            with open(TEMP_AUDIO, "wb") as f:
                f.write(audio.get_wav_data())

            result = subprocess.run(
                [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", TEMP_AUDIO, "-nt"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            text = result.stdout.strip()

            if text:
                self.ui_queue.put(("msg", "You (Voice)", text, "user"))
                self.run_pipeline(text)
            else:
                self.ui_queue.put(("msg", "System", "Wake word heard but command unclear.", "system"))

        except sr.WaitTimeoutError:
            self.ui_queue.put(("msg", "System", "Listening timed out.", "system"))

    # ── PIPELINE BRIDGE ───────────────────────────────────────────────────────
    def run_pipeline(self, user_input: str):
        def on_status(msg, color):
            self.set_status(msg, color)

        response, source, parsed = self.pipeline.run(user_input, on_status=on_status)

        tag_map = {
            "tool":    "tool",
            "cache":   "cache",
            "blocked": "block",
            "llm":     "jarvis",
        }
        label_map = {
            "tool":    f"{AGENT_NAME} [instant]",
            "cache":   f"{AGENT_NAME} [cached]",
            "blocked": f"{AGENT_NAME} [blocked]",
            "llm":     AGENT_NAME,
        }

        self.ui_queue.put((
            "msg",
            label_map.get(source, AGENT_NAME),
            response,
            tag_map.get(source, "jarvis")
        ))

        # Speak sentence-by-sentence to reduce perceived latency
        threading.Thread(
            target=self.speak_sentences, args=(response,), daemon=True
        ).start()

        self.set_status("● STANDBY", "#00d4ff")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = JarvisApp(root)
    root.mainloop()
