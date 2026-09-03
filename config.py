import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
WHISPER_PATH  = os.path.expanduser("~/whisper.cpp/main")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-tiny.en.bin")
TEMP_AUDIO    = "/tmp/jarvis_input.wav"
TEMP_TTS      = "/tmp/jarvis_tts.wav"
PIPER_PATH    = os.path.expanduser("~/piper/piper")
PIPER_MODEL   = os.path.expanduser("~/piper/en_GB-alan-medium.onnx")
DB_PATH       = os.path.expanduser("~/jarvis.db")

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL       = "qwen3.5:4b"       # ~7-8 t/s on Pi5, 4.16GB RAM
EMBED_MODEL = "all-MiniLM-L6-v2" # 90MB RAM, ~50ms/embed on Pi5
OLLAMA_HOST = "http://localhost:11434"

# ── IDENTITY ──────────────────────────────────────────────────────────────────
AGENT_NAME  = "Jarvis"
WAKE_WORDS  = ["hey jarvis", "hey aami", "jarvis", "aami"]

# ── MEMORY / CACHE ────────────────────────────────────────────────────────────
MAX_HISTORY        = 100
CACHE_TTL_HOURS    = 24
SEMANTIC_THRESHOLD = 0.90

# Categories that are time-sensitive — never cache these
NO_CACHE = {"time_query", "date_query", "system_status"}

# ── OLLAMA OPTIONS (tuned for Pi 5) ───────────────────────────────────────────
# Fast agents: classify and security — short output, small context
OPTS_FAST = {
    "temperature": 0.0,
    "num_predict": 400,   # enough for JSON output without thinking overhead
    "num_ctx":    1024,
    "num_thread":    4,
}
# Main agent: full reasoning
OPTS_MAIN = {
    "temperature": 0.7,
    "num_predict": 300,   # 2-3 spoken sentences
    "num_ctx":    4096,
    "num_thread":    4,
}
# Verification: moderate context, low temp
OPTS_VERIFY = {
    "temperature": 0.1,
    "num_predict": 400,
    "num_ctx":    2048,
    "num_thread":    4,
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Jarvis, a highly intelligent AI assistant with dry British wit.
Speak concisely and precisely. Occasionally address the user as 'sir' or 'ma'am'.
Responses will be read aloud — no markdown, no bullet points, no code blocks. Natural spoken sentences only.
Maximum 2-3 sentences. If more detail is needed, summarise and offer to elaborate.
If the user asks you to remember something, capture it in the 'remember' field.
Return ONLY valid JSON: {"response": "<your spoken reply>", "remember": "<fact or empty>"}"""

# ── API KEYS (set as env vars, not hardcoded) ─────────────────────────────────
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
