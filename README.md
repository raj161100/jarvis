# J.A.R.V.I.S — Raspberry Pi 5 Offline AI Assistant

A fully offline AI assistant for Raspberry Pi 5 with a British accent, multi-agent pipeline, persistent memory, and live data tools. No cloud required — everything runs on-device.

---

## Features

| Feature | Detail |
|---|---|
| **LLM** | Ollama (qwen3.5:4b) — 9.4 t/s on Pi 5, thinking + tool support |
| **Speech input** | Whisper.cpp (tiny.en model) — offline STT |
| **Wake word** | Configurable list, detected by Whisper without a cloud service |
| **Voice output** | Piper TTS — neural British voice (Alan, en_GB); espeak-ng fallback |
| **Multi-agent pipeline** | Parse+Security → Orchestrator → Verification → Persist |
| **Persistent memory** | SQLite — remembers facts across reboots |
| **Conversation history** | Full chat history stored per session |
| **Semantic cache** | Sentence-transformers similarity search (cosine, threshold 0.90) — ~70ms lookup |
| **Exact hash cache** | MD5 lookup for identical queries — instant |
| **Security audit log** | Every query classified and logged with threat level |
| **Live tools** | Weather, news, web search, Wikipedia |
| **Offline tools** | Time, date, CPU/RAM/temperature/throttle status |
| **Sentence-streaming TTS** | First sentence spoken in ~3s while rest generates |
| **800×480 GUI** | Jarvis-themed dark terminal UI with colour-coded sources |
| **Model warm-up** | Dummy query at startup eliminates first-query cold-start |
| **Pi 5 optimisations** | Flash attention, PCIe Gen3, swap disabled, 4-thread pinning |

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| **Board** | Raspberry Pi 5 (8 GB RAM minimum) |
| **Storage** | 64 GB SD card or faster M.2 SSD via HAT |
| **Microphone** | USB microphone (any ALSA-compatible device) |
| **Speaker** | USB audio adapter + speaker, or 3.5 mm DAC HAT |
| **Cooling** | Active fan — mandatory; CPU reaches 75 °C under LLM load |
| **Display** | 800×480 screen connected via HDMI or DSI (optional for GUI mode) |

---

## Project Structure

```
jarvis/
├── assistant.py      # Main app — GUI, voice I/O, TTS, pipeline bridge
├── agents.py         # Four-agent pipeline (Parse+Security, Orchestrator, Verification)
├── tools.py          # Tool handler — weather, news, search, Wikipedia, time, system
├── database.py       # SQLite wrapper — memory, history, cache, security log
├── semantic_cache.py # Sentence-transformer similarity cache
├── config.py         # Central config — paths, model names, thresholds, prompts
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
- System packages (portaudio, espeak-ng, cmake, ffmpeg)
- Disables swap (mandatory for LLM performance on SD card)
- Enables PCIe Gen3 if you have an M.2 HAT
- Python virtual environment + pip packages
- Ollama install + Pi 5 systemd config (flash attention, keep-alive)
- Model pull: `qwen3.5:4b` and `nomic-embed-text`
- Piper TTS binary + British voice (Alan, medium quality)
- Whisper.cpp build from source

### 2. Optional — set your NewsAPI key

```bash
export NEWS_API_KEY=your_key_here
```

Without this the news tool falls back to BBC RSS (still works, no key needed).

### 3. Run

```bash
source ai_env/bin/activate
python assistant.py
```

Say **"Hey Jarvis"** or **"Jarvis"** to activate voice input. You can also type in the text box.

---

## How to Change or Add a Model

### Switch to a different model

1. Pull the model with Ollama:
   ```bash
   ollama pull mistral:7b
   ```

2. Edit `config.py`:
   ```python
   MODEL = "mistral:7b"   # was "qwen3.5:4b"
   ```

3. Restart the assistant. The new model loads automatically.

**Recommended models for Pi 5 (8 GB):**

| Model | RAM | Speed (Pi 5) | Notes |
|---|---|---|---|
| `qwen3.5:4b` *(default)* | 4.2 GB | ~9 t/s | Best quality/speed balance; thinking + tools |
| `llama3.2:3b` | 2.0 GB | ~12 t/s | Faster, lower quality |
| `phi3.5:3.8b` | 2.3 GB | ~11 t/s | Strong reasoning for its size |
| `gemma3:4b` | 3.5 GB | ~8 t/s | Good instruction following |
| `mistral:7b` | 4.7 GB | ~5 t/s | Slower but higher quality responses |

Do not load two large models simultaneously — the Pi 5 has 8 GB total and the OS + Piper take ~1.5 GB.

### Tune model options

In `config.py` there are three option sets:

```python
OPTS_FAST   = {"temperature": 0.0, "num_predict": 200, "num_ctx": 1024,  "num_thread": 4}
OPTS_MAIN   = {"temperature": 0.7, "num_predict": 180, "num_ctx": 4096,  "num_thread": 4}
OPTS_VERIFY = {"temperature": 0.1, "num_predict": 300, "num_ctx": 2048,  "num_thread": 4}
```

- `OPTS_FAST` — used by ParseAndSecureAgent (speed matters most)
- `OPTS_MAIN` — used by OrchestratorAgent (quality matters most)
- `OPTS_VERIFY` — used by VerificationAgent (accuracy matters most)
- Raise `num_predict` for longer responses; lower it to cut latency
- Raise `num_ctx` for longer memory windows; lower it to reduce prefill time

---

## How to Add a New Tool / API

### Step 1 — Add the method to `tools.py`

Inside the `ToolHandler` class, add a `_my_tool` method:

```python
def _my_tool(self, entities: list) -> str:
    """Example: fetch a stock price."""
    import requests
    symbol = entities[0] if entities else "AAPL"
    try:
        r = requests.get(f"https://api.example.com/price/{symbol}", timeout=5)
        data = r.json()
        return f"The price of {symbol} is ${data['price']:.2f}."
    except Exception as e:
        return f"Unable to fetch stock price: {e}"
```

### Step 2 — Register it in `TOOL_MAP`

Still in `tools.py`:

```python
TOOL_MAP = {
    "time_query":    self._time_query,
    "date_query":    self._date_query,
    "system_status": self._system_status,
    "weather":       self._weather,
    "news":          self._news,
    "web_search":    self._web_search,
    "wikipedia":     self._wikipedia,
    "stock_price":   self._my_tool,   # ← add this line
}
```

### Step 3 — Add it to the ParseAndSecureAgent routing prompt in `agents.py`

Find the `PROMPT` string inside `ParseAndSecureAgent` and add a line to the routing table:

```
  stock_price  → asking about a share price, stock, market value
```

### Step 4 — Optionally add a quick regex shortcut in `agents.py`

For queries with obvious patterns (avoids an LLM call entirely):

```python
QUICK_TOOL_PATTERNS = {
    ...
    r"\b(stock price|share price|how much is|ticker)\b": "stock_price",
}
```

### Step 5 — Prevent caching for real-time data

If your tool returns time-sensitive data, add the tool name to `NO_CACHE` in `config.py`:

```python
NO_CACHE = {"time_query", "date_query", "system_status", "stock_price"}
```

---

## How to Add / Change Wake Words

In `config.py`:

```python
WAKE_WORDS = ["jarvis", "hey jarvis", "ok jarvis", "computer"]
```

Add any phrase you want. Whisper.cpp transcribes the audio and checks if any wake word appears in the transcript. Shorter phrases are more reliable.

---

## Agent Pipeline Explained

```
User input
    │
    ├─ Quick regex check ──────────────── instant → tool result
    │
    ├─ Semantic cache lookup ──────────── ~70ms  → cached response
    │
    ├─ ParseAndSecureAgent ────────────── 2-3s
    │       classifies input + security check in one LLM call
    │       ├─ BLOCKED ────────────────────────── refusal message
    │       └─ tool detected → tool handler ────── instant result
    │
    ├─ OrchestratorAgent ──────────────── 5-15s
    │       main reasoning with memory + history
    │
    ├─ VerificationAgent (complex only) ── 2-3s
    │       checks voice-friendliness, brevity, relevance
    │
    └─ Persist to DB + semantic cache
```

**Agents:**

| Agent | File | Purpose |
|---|---|---|
| `ParseAndSecureAgent` | `agents.py` | Classifies input + security filter in one call |
| `OrchestratorAgent` | `agents.py` | Main reasoning, uses memory + history |
| `VerificationAgent` | `agents.py` | Quality check — voice-friendly, on-topic, brief |
| Tool handler | `tools.py` | Fetches real data for routed queries |

---

## Database Schema

SQLite database at `~/jarvis.db` (path set in `config.py`).

| Table | Columns | Purpose |
|---|---|---|
| `memories` | `id, content, category, session_id, timestamp` | Facts Jarvis remembers long-term |
| `conversations` | `id, role, content, category, session_id, timestamp` | Full chat history per session |
| `response_cache` | `id, input_hash, response, category, timestamp` | Exact MD5 hash cache (24 hr TTL) |
| `semantic_cache` | `id, query, response, category, embedding_json, timestamp` | Similarity cache with stored embeddings |
| `security_log` | `id, input_text, threat_level, block_reason, session_id, timestamp` | Security audit trail |

To inspect the database:

```bash
sqlite3 ~/jarvis.db ".tables"
sqlite3 ~/jarvis.db "SELECT * FROM memories ORDER BY timestamp DESC LIMIT 10;"
sqlite3 ~/jarvis.db "SELECT threat_level, input_text FROM security_log ORDER BY timestamp DESC;"
```

---

## Semantic Cache Tuning

The semantic cache stores embeddings of past queries and reuses responses when a new query is similar enough.

In `config.py`:

```python
SEMANTIC_THRESHOLD = 0.90  # cosine similarity — raise to be stricter, lower to reuse more
```

- `0.95` — near-identical queries only (very conservative)
- `0.90` — default; catches rephrased versions of the same question
- `0.80` — aggressive reuse; may return wrong cached answer for distinct questions

To clear the semantic cache:

```bash
sqlite3 ~/jarvis.db "DELETE FROM semantic_cache;"
```

The in-memory embedding matrix is rebuilt from SQLite on each startup.

---

## Voice Options (Piper TTS)

The default voice is **Alan** (en_GB, medium quality). Other available British voices:

| Voice | File | Character |
|---|---|---|
| `en_GB-alan-medium` *(default)* | `en_GB-alan-medium.onnx` | Male, clear, neutral British |
| `en_GB-alan-low` | `en_GB-alan-low.onnx` | Same speaker, faster/smaller |
| `en_GB-jenny-dioco-medium` | `en_GB-jenny_dioco-medium.onnx` | Female British |
| `en_GB-cori-high` | `en_GB-cori-high.onnx` | Female, highest quality |
| `en_US-ryan-high` | `en_US-ryan-high.onnx` | American male, highest quality |

To switch voice:

1. Download the `.onnx` and `.onnx.json` files from [HuggingFace rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `~/piper/`.
2. Edit `config.py`:
   ```python
   PIPER_MODEL = os.path.expanduser("~/piper/en_GB-cori-high.onnx")
   ```

---

## Troubleshooting

### Ollama not responding
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
ollama list   # check models are pulled
```

### No audio output
```bash
aplay -l                  # list audio devices
aplay /usr/share/sounds/alsa/Front_Left.wav   # test playback
```
Set the correct device in `~/.asoundrc` if needed.

### Microphone not detected
```bash
arecord -l                # list recording devices
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

### TTS too slow / stuttering
- Check Ollama is still loaded: `ollama ps`
- Ensure swap is off: `free -h` (Swap line should show 0)
- Lower `OPTS_MAIN["num_predict"]` in `config.py` for shorter responses

### CPU overheating (above 80 °C)
- Verify fan is running: `vcgencmd measure_temp`
- The Pi 5 throttles at 85 °C — responses will slow down
- Check throttle flags: `vcgencmd get_throttled` (0x0 means no throttling)

### Model responses in wrong format / errors
- The model occasionally returns malformed JSON — the pipeline catches this and returns a safe fallback
- Try a different model (see model table above)
- Check Ollama logs: `journalctl -u ollama -n 50`

### Semantic cache returning wrong answers
- Raise `SEMANTIC_THRESHOLD` in `config.py` (e.g. to `0.95`)
- Or clear it: `sqlite3 ~/jarvis.db "DELETE FROM semantic_cache;"`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEWS_API_KEY` | *(empty)* | NewsAPI.org key for headlines (falls back to BBC RSS without it) |
| `OLLAMA_HOST` | `http://localhost:11434` | Override if Ollama runs on another machine |

---

## Adding a NewsAPI Key (Permanent)

```bash
echo 'export NEWS_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

Get a free key at [newsapi.org](https://newsapi.org) — 100 requests/day on the free tier.

---

## Licence

MIT — do whatever you like with it.
