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

from ollama import Client

from config import DB_PATH, AGENT_NAME, OLLAMA_HOST, MODEL, OPTS_CHAT
from database import JarvisDB
from agents import AgentPipeline, FULL_SYSTEM_PROMPT
from speech import Speaker
from voice_input import VoiceInput

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
        # One query at a time — concurrent requests just queue inside Ollama anyway
        self.pipeline_lock = threading.Lock()
        self.muted         = False               # MUTE button: replies shown on screen only
        self._last_status  = None

        # Init DB, pipeline, speech output and voice input (each reports problems in the chat window)
        self.db       = JarvisDB(DB_PATH)
        self.pipeline = AgentPipeline(self.db)
        self.speaker  = Speaker(notice=self.system_msg)
        self.voice    = VoiceInput(self.speaker, on_command=self.handle_voice_command,
                                   on_status=self.set_status, notice=self.system_msg,
                                   is_busy=self.pipeline_lock.locked)

        self.build_ui()
        self.check_queue_loop()

        # Pre-warm model in background so first query has no cold-start delay
        threading.Thread(target=self._warm_model, daemon=True).start()

        # Wake word listener (loads the speech recogniser first)
        threading.Thread(target=self.voice.run, daemon=True).start()

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

        # Silences voice replies (typed questions are always answered on screen only)
        self.mute_btn = tk.Button(
            row, text="MUTE", command=self.on_mute_pressed,
            bg="#30363d", fg="white", activebackground="#484f58", activeforeground="white",
            font=("Courier", 12, "bold"), bd=0, padx=14, pady=6, highlightthickness=0
        )
        self.mute_btn.pack(side=tk.LEFT, padx=(0, 10))

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

    # ── MODEL WARM-UP ─────────────────────────────────────────────────────────
    def _warm_model(self):
        """Load the model into RAM at startup to eliminate cold-start lag."""
        # Hold the pipeline lock so queries wait for the load and the listener keeps the status
        with self.pipeline_lock:
            self.set_status("● WARMING UP", "#d29922")
            t0 = time.time()
            try:
                # Same num_ctx/num_thread as the chat call, otherwise Ollama reloads on the first query.
                # Sending the real system prompt (incl. profile) pre-processes it, so the first
                # question only pays for its own words.
                ollama_client.chat(
                    model=MODEL,
                    messages=[{"role": "system", "content": FULL_SYSTEM_PROMPT},
                              {"role": "user", "content": "ready"}],
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
        # Typed questions get on-screen answers only; voice is for spoken questions
        threading.Thread(target=self.run_pipeline, args=(text, False), daemon=True).start()

    def on_talk_pressed(self):
        self.speaker.stop()
        if not self.voice.ready:
            if self.voice.failed:
                self.system_msg("Voice input isn't available — see the messages above. Use the text box.")
            else:
                self.system_msg("Voice input is still starting up — try again in a moment.")
            return
        if not self.pipeline_lock.locked():   # otherwise it listens once the current answer is done
            self.set_status("● LISTENING", "#f85149")
        self.voice.ptt.set()

    def on_mute_pressed(self):
        self.muted = not self.muted
        if self.muted:
            self.speaker.stop()
            self.mute_btn.config(text="MUTED", bg="#da3633", activebackground="#f85149")
        else:
            self.mute_btn.config(text="MUTE", bg="#30363d", activebackground="#484f58")

    # ── VOICE INPUT ───────────────────────────────────────────────────────────
    def handle_voice_command(self, text):
        """Called from the voice thread with a transcribed command."""
        self.ui_queue.put(("msg", "You (Voice)", text, "user"))
        self.run_pipeline(text, speak=True)

    # ── PIPELINE BRIDGE ───────────────────────────────────────────────────────
    def run_pipeline(self, user_input: str, speak: bool):
        with self.pipeline_lock:
            try:
                self._run_pipeline(user_input, speak)
            except Exception as e:
                print(f"[Jarvis] Pipeline error: {type(e).__name__}: {e}")
                self.ui_queue.put(("msg", "System", f"Error: {e}", "system"))
            finally:
                self.set_status("● STANDBY", "#00d4ff")

    def _run_pipeline(self, user_input: str, speak: bool):
        streamed = []
        epoch = self.speaker.epoch    # changes when Escape/TALK/typing stops speech

        def speaking():
            return speak and not self.muted and self.speaker.epoch == epoch

        def on_sentence(sentence):
            # LLM replies arrive sentence by sentence: show (and speak) each immediately.
            # Checked per sentence so MUTE / Escape silence the rest of the reply.
            voice = speaking()
            if not streamed:
                self.ui_queue.put(("start", AGENT_NAME, "jarvis"))
                if voice:
                    self.set_status("● SPEAKING", "#00d4ff")
            streamed.append(sentence)
            self.ui_queue.put(("append", sentence + " "))
            if voice:
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
        if speaking():
            self.speaker.say_text(response)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = JarvisApp(root)
    root.mainloop()
