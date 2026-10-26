# J.A.R.V.I.S — Raspberry Pi 5 Offline AI Assistant

A fully offline AI assistant for Raspberry Pi 5 with a British accent, streaming voice replies, persistent memory, and live data tools. No cloud required — everything runs on-device.

---

## Features

| Feature | Detail |
|---|---|
| **LLM** | Ollama (qwen3.5:4b, thinking disabled) — one streaming call per turn |
| **Speech input** | Whisper.cpp — `tiny.en` scans for the wake word, `base.en` transcribes commands |
| **Wake word** | "Hey Jarvis …" in one breath, or "Jarvis" → beep → command |
| **Push-to-talk** | **TALK** button on the touchscreen skips the wake word |
| **Voice output** | Piper TTS — neural British voice (Alan, en_GB); espeak-ng fallback |
| **Reply mode** | Spoken questions get spoken replies; typed questions are answered on screen only; **MUTE** button silences voice |
| **Sentence-streaming TTS** | Each sentence is spoken as soon as the LLM finishes it |
| **Half-duplex mic** | Mic input is ignored while Jarvis is speaking, so it never hears itself |
| **Regex router** | Tool queries (time, weather, Wikipedia…) are routed instantly, no LLM call |
| **Persistent memory** | "Remember that …" / "What do you know about me" / "Forget that" |
| **Conversation history** | Last 10 turns sent to the model as context |
| **Semantic cache** | Reuses web search / Wikipedia answers for similar questions |
| **Security audit log** | Suspicious prompts are logged (never blocked — Jarvis has no risky tools) |
| **Live tools** | Weather, news, web search, Wikipedia |
| **Offline tools** | Time, date, CPU/RAM/temperature status, machine info |
| **800×480 GUI** | Jarvis-themed dark terminal UI with colour-coded sources |
| **Model warm-up** | Model is loaded and pinned in RAM at startup (`keep_alive=-1`) |
| **Pi 5 optimisations** | Flash attention, PCIe Gen3, swap disabled, 4-thread pinning, temp audio in `/dev/shm` |

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| **Board** | Raspberry Pi 5 (8 GB RAM minimum) |
| **Storage** | 64 GB SD card or faster M.2 SSD via HAT |
| **Microphone** | USB microphone (any ALSA-compatible device) |
| **Speaker** | USB speaker, Bluetooth speaker, HDMI audio, or 3.5 mm DAC HAT |
| **Cooling** | Active fan — mandatory; CPU reaches 75 °C under LLM load |
| **Display** | 800×480 screen connected via HDMI or DSI |

---

## Project Structure

```
jarvis/
├── assistant.py      # Main app — GUI, wake word, push-to-talk, STT, pipeline bridge
├── speech.py         # Speaker — Piper TTS synth + playback queue, beep, stop
├── agents.py         # Router, memory commands, streaming LLM orchestrator, pipeline
├── tools.py          # Tool handler — weather, news, search, Wikipedia, time, system
├── database.py       # SQLite wrapper — memory, history, cache, security log
├── semantic_cache.py # Sentence-transformer similarity cache
├── config.py         # Central config — paths, model, options, prompt
├── test_pipeline.py  # Text-only pipeline test (no GUI / mic needed)
├── requirements.txt  # Python dependencies
├── setup.sh          # One-shot Pi 5 setup script
└── README.md
```

---

## Quick Start

### 1. Clone and run setup

```bash
git clone https://github.com/raj161100/jarvis.git
cd jarvis
chmod +x setup.sh
./setup.sh
```

`setup.sh` handles everything:
- System packages (portaudio, espeak-ng, cmake, ffmpeg, python3-tk)
- Disables swap (mandatory for LLM performance on SD card)
- Enables PCIe Gen3 if you have an M.2 HAT
- Python virtual environment + pip packages
- Ollama install + Pi 5 systemd config (flash attention, keep-alive)
- Model pull: `qwen3.5:4b`
- Piper TTS binary + British voice (Alan, medium quality)
- Whisper.cpp build (`whisper-cli`) + `tiny.en` and `base.en` models

It is safe to re-run — finished steps are skipped.

### 2. Optional — set your NewsAPI key and home city

```bash
export NEWS_API_KEY=your_key_here
export JARVIS_LOCATION="New York"
```

Without a key the news tool falls back to BBC RSS.

### 3. Run

Run from a terminal on the Pi desktop (the GUI needs a display):

```bash
cd ~/jarvis
source ai_env/bin/activate
python assistant.py
```

Wait until the status changes from **WARMING UP** to **STANDBY** / **LISTENING** (the first load of the model takes about a minute), then:

- Say **"Hey Jarvis, what time is it?"** in one breath, or
- Say **"Jarvis"**, wait for the beep, then speak your command, or
- Tap **TALK**, wait for the beep, then speak, or
- Type in the text box and press Enter — typed questions are answered on screen only.

Spoken questions are answered on screen **and** out loud. Tap **MUTE** to keep voice replies on screen too (tap again to unmute). Press **Escape** to stop Jarvis talking.

### 4. Text-only test

To check the model and tools without the GUI or microphone:

```bash
python test_pipeline.py
```

---

## Memory Commands

These are handled instantly in code (no LLM call):

| Say | Effect |
|---|---|
| "Remember that my sister is Priya" | Saves the fact |
| "What do you know about me?" | Reads back saved facts |
| "Forget that" | Deletes the most recently saved fact |

Saved facts are also given to the model on every turn, so it can use them in normal answers.

---

## Personal Profile

For background Jarvis should always know (your name, where you live, your work, your devices), create `~/jarvis_profile.md` on the Pi:

```bash
nano ~/jarvis_profile.md
```

Write it as short plain sentences or bullet points. Anything inside `<!-- -->` is ignored, so you can leave notes to yourself. It is loaded at startup (restart Jarvis after editing) and pre-loaded during warm-up, so it doesn't slow down your first question.

- Keep it under ~1500 characters — it is included in every conversation, and longer means slower on the Pi.
- It lives outside the repo, so it is never pushed to GitHub. Use `JARVIS_PROFILE=/path/to/file` to keep it elsewhere.
- Anyone who talks to Jarvis can ask about what's in it — don't put passwords or secrets there.

---

## How to Change the Model

Set `JARVIS_MODEL` (no code change needed):

```bash
ollama pull qwen3.5:2b
JARVIS_MODEL=qwen3.5:2b python assistant.py
```

Or edit `MODEL` in `config.py`.

**Models for Pi 5 (8 GB), CPU only:**

| Model | Notes |
|---|---|
| `qwen3.5:4b` *(default)* | Best answers; ~3 tokens/s, so long replies take a while |
| `qwen3.5:2b` | Roughly twice as fast; a little less accurate |
| `gemma4:e2b` | Fast; weaker at following instructions |

Do not load two large models at once — the Pi 5 has 8 GB total and the OS + Piper take ~1.5 GB.

### Tune model options

`config.py` has a single option set used for every call:

```python
NUM_CTX    = 2048
NUM_THREAD = 4
OPTS_CHAT  = {"temperature": 0.7, "num_predict": 160, "num_ctx": NUM_CTX, "num_thread": NUM_THREAD}
```

- Keep the same `num_ctx` / `num_thread` everywhere — Ollama reloads the model whenever they change.
- Lower `num_predict` for shorter, faster replies.
- Raise `NUM_CTX` for more history; it makes every reply slower on the Pi.

---

## How to Add a New Tool / API

### Step 1 — Add the method to `tools.py`

Inside the `ToolHandler` class, add a `_my_tool` method. Return `None` if it found nothing — the question then falls through to the LLM.

```python
def _stock_price(self, entities: list) -> str:
    symbol = entities[0] if entities else "AAPL"
    try:
        r = requests.get(f"https://api.example.com/price/{symbol}", timeout=5)
        return f"The price of {symbol} is ${r.json()['price']:.2f}."
    except Exception:
        return None
```

### Step 2 — Register it in `TOOL_MAP` (`tools.py`)

```python
TOOL_MAP = {
    ...
    "stock_price": "_stock_price",
}
```

### Step 3 — Add a route in `agents.py`

Add a regex to `TOOL_ROUTES`. A named group `arg` becomes the tool's entity:

```python
(r"\b(?:stock|share) price (?:of|for) (?P<arg>\w+)", "stock_price"),
```

### Step 4 — Caching (optional)

Only tools listed in `CACHEABLE_TOOLS` (`config.py`) use the semantic cache. Leave real-time tools out.

---

## How to Add / Change Wake Words

In `config.py`:

```python
WAKE_WORDS = ["jarvis", "jarvas", "jervis", "javis", "aami"]
```

Whisper `tiny.en` often mishears names, so include common misspellings. An optional "hey / hi / ok" before the wake word is handled automatically.

---

## Pipeline Explained

```
User input (voice or text)
    │
    ├─ Memory command? ───────────────── instant → saved / recalled / forgotten
    │
    ├─ Regex router matches a tool?
    │       ├─ semantic cache (search/Wikipedia only) → cached answer
    │       ├─ tool answer ──────────────── instant / ~1s network
    │       └─ tool found nothing ───────── falls through ↓
    │
    ├─ OrchestratorAgent ─────────────── ONE streaming LLM call
    │       memory + history + tool result as context
    │       each sentence is spoken as soon as it is generated
    │
    └─ Persist to DB (errors are never saved or cached)
```

Per-stage timings are printed in the terminal as `[Pipeline] …` lines, and speech-to-text timings as `[Jarvis] STT …`.

---

## Database Schema

SQLite database at `~/jarvis.db` (path set in `config.py`).

| Table | Purpose |
|---|---|
| `memories` | Facts Jarvis remembers long-term |
| `conversations` | Chat history |
| `response_cache` | Exact-match cache (legacy) |
| `semantic_cache` | Similarity cache with stored embeddings |
| `security_log` | Audit trail of every query |

To inspect the database:

```bash
sqlite3 ~/jarvis.db ".tables"
sqlite3 ~/jarvis.db "SELECT * FROM memories ORDER BY id DESC LIMIT 10;"
```

To start with a clean history (keeps memories):

```bash
sqlite3 ~/jarvis.db "DELETE FROM conversations; DELETE FROM semantic_cache;"
```

---

## Semantic Cache Tuning

In `config.py`:

```python
SEMANTIC_THRESHOLD = 0.90           # raise to be stricter, lower to reuse more
CACHEABLE_TOOLS    = {"wikipedia", "web_search"}
```

To clear it: `sqlite3 ~/jarvis.db "DELETE FROM semantic_cache;"`

---

## Voice Output and Speakers

Jarvis speaks with the Piper neural voice and plays it through `aplay`, so it uses whatever the Pi's **default audio output** is — HDMI, a USB speaker, or a Bluetooth speaker.

**USB speaker:** plug it in, then pick it as the output from the volume icon on the Pi desktop.

**Bluetooth speaker:** pair it from the Bluetooth icon on the Pi desktop (or `bluetoothctl` → `scan on` → `pair <MAC>` → `connect <MAC>` → `trust <MAC>`), then pick it from the volume icon. Bluetooth adds a little delay before each sentence.

From a terminal:

```bash
wpctl status                 # list outputs (Sinks); * marks the default
wpctl set-default <id>       # make a sink the default
speaker-test -t wav -c 2     # test sound (Ctrl+C to stop)
```

### Change the voice

| Voice | File | Character |
|---|---|---|
| `en_GB-alan-medium` *(default)* | `en_GB-alan-medium.onnx` | Male, clear, neutral British |
| `en_GB-alan-low` | `en_GB-alan-low.onnx` | Same speaker, faster/smaller |
| `en_GB-jenny_dioco-medium` | `en_GB-jenny_dioco-medium.onnx` | Female British |
| `en_GB-cori-high` | `en_GB-cori-high.onnx` | Female, highest quality |
| `en_US-ryan-high` | `en_US-ryan-high.onnx` | American male, highest quality |

1. Download the `.onnx` and `.onnx.json` files from [HuggingFace rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `~/piper/`.
2. Edit `PIPER_MODEL` in `config.py`.

---

## Troubleshooting

### Startup messages
`assistant.py` checks its dependencies at startup and prints what is missing in the chat window (Piper, `whisper-cli`, Whisper models, Ollama model).

### Ollama not responding
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
ollama list   # check the model is pulled
ollama ps     # check it is loaded
```

### Voice is never recognised
- Watch the terminal for `[Jarvis] STT ggml-tiny.en.bin 0.8s: '…'` lines — they show exactly what Whisper heard.
- No STT lines at all → check the mic: `arecord -l` and
  `python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"`
- `whisper-cli not found` → build it: `cd ~/whisper.cpp && cmake -B build && cmake --build build -j4 --config Release`

### No audio output
```bash
aplay -l
speaker-test -t wav -c 2
```
See [Voice Output and Speakers](#voice-output-and-speakers).

### Replies are slow
- Check the `[Pipeline]` timings in the terminal.
- `ollama ps` should show the model loaded "Forever".
- Swap must be off: `free -h`.
- Try `JARVIS_MODEL=qwen3.5:2b`, or lower `num_predict` in `config.py`.

### CPU overheating (above 80 °C)
- `vcgencmd measure_temp`
- `vcgencmd get_throttled` (0x0 means no throttling)

### Semantic cache returning wrong answers
- Raise `SEMANTIC_THRESHOLD`, or clear it: `sqlite3 ~/jarvis.db "DELETE FROM semantic_cache;"`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JARVIS_MODEL` | `qwen3.5:4b` | Ollama model to use |
| `JARVIS_LOCATION` | `New York` | Default city for weather |
| `JARVIS_UNITS` | `F` | Weather temperature units: `F` or `C` |
| `JARVIS_PROFILE` | `~/jarvis_profile.md` | Personal profile file |
| `NEWS_API_KEY` | *(empty)* | NewsAPI.org key for headlines (falls back to BBC RSS) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |

To make them permanent:

```bash
echo 'export JARVIS_LOCATION="New York"' >> ~/.bashrc
source ~/.bashrc
```

---

## Licence

MIT — do whatever you like with it.
