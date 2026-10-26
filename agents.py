import re
import time

from ollama import Client

from config import (
    MODEL, OLLAMA_HOST, SYSTEM_PROMPT, OPTS_CHAT,
    MAX_HISTORY, MAX_MEMORIES, CACHEABLE_TOOLS
)
from database import JarvisDB
from tools import ToolHandler
from semantic_cache import SemanticCache

ollama_client = Client(host=OLLAMA_HOST)


# ── ROUTER: regex → tool, no LLM call ─────────────────────────────────────────
# Checked in order against the lowercased input (trailing punctuation stripped).
# A named group 'arg' becomes the tool's entity (city, topic, search query).
TOOL_ROUTES = [
    (r"\b(what time is it|what'?s the time|tell me the time|current time)\b",                    "time_query"),
    (r"\b(what'?s the date|what is the date|today'?s date|what day is (it|today))\b",            "date_query"),
    (r"\b(system status|cpu (usage|load|temp\w*)|(ram|memory) usage|how hot are you"
     r"|your temperature|(pi|core|cpu) temperature)\b",                                          "system_status"),
    (r"\b(machine|hardware|device|computer)\b.*\byou(?:'re| are)? (?:running|run) on\b"
     r"|\bwhat are you running on\b",                                                            "machine_info"),
    (r"\b(weather|forecast|temperature(?= (?:outside|in)\b)|is it (?:going to )?rain\w*)\b"
     r"(?:.*?\b(?:in|for|at)\s+(?P<arg>[a-z][a-z .'-]+))?",                                      "weather"),
    (r"\b(news|headlines)\b(?:.*?\b(?:about|on)\s+(?P<arg>.+))?",                                "news"),
    (r"^(?:please )?(?:search(?: the web| online)? for|look up|google)\s+(?P<arg>.+)",           "web_search"),
    (r"^(?:who (?:is|was)|tell me about|what (?:is|was) an?)\s+"
     r"(?!(?:my|your|you|yourself|me|it|that|this)\b)(?P<arg>.+)",                               "wikipedia"),
]

# Tools whose raw output reads badly aloud — the LLM turns it into a spoken summary
SUMMARISE_TOOLS = {"web_search"}

FILLER_WORDS = r"\b(today|tomorrow|tonight|right now|now|please|like|currently)\b"


def route(text: str):
    """Returns (tool_name, entities) or (None, [])."""
    original = re.sub(r"[?.!]+$", "", text.strip())
    t = original.lower()
    for pattern, tool_name in TOOL_ROUTES:
        m = re.search(pattern, t)
        if m:
            arg = ""
            if m.groupdict().get("arg"):
                # Match on lowercase, but keep the original casing (Wikipedia titles are case-sensitive)
                arg = original[m.start("arg"):m.end("arg")]
                arg = re.sub(FILLER_WORDS, "", arg, flags=re.I).strip(" ,'")
            if arg and tool_name == "weather":
                arg = arg.title()
            elif arg and tool_name == "wikipedia":
                arg = arg[0].upper() + arg[1:]
            return tool_name, [arg] if arg else []
    return None, []


# ── GUARD: log-only, never blocks ─────────────────────────────────────────────
# Jarvis has no tools with side effects, so an LLM security classifier only added
# latency and wrongly blocked normal questions. Suspicious input is just logged.
SUSPICIOUS = re.compile(
    r"ignore (?:all |any |your |the )?(?:previous|prior|above) (?:instructions|prompts)"
    r"|disregard (?:your|the) (?:rules|instructions)|system prompt|jailbreak|developer mode",
    re.I,
)


def threat_level(text: str) -> str:
    return "medium" if SUSPICIOUS.search(text) else "safe"


# ── MEMORY COMMANDS: handled in code, no LLM call ─────────────────────────────
REMEMBER_RE = re.compile(r"^(?:please )?(?:remember|note|don'?t forget)(?: that)?\s+(?P<fact>.+)", re.I)
RECALL_RE   = re.compile(r"\bwhat (?:do|did) you (?:know|remember) about me\b|\bwhat have you remembered\b", re.I)
FORGET_RE   = re.compile(r"^(?:please )?forget (?:that|the last thing|what i (?:just )?said)\b", re.I)


def to_second_person(text: str) -> str:
    """'my sister is Priya' -> 'your sister is Priya' for spoken confirmations."""
    swaps = {r"\bi am\b": "you are", r"\bi'm\b": "you're", r"\bi\b": "you", r"\bmy\b": "your",
             r"\bmine\b": "yours", r"\bme\b": "you", r"\bmyself\b": "yourself"}
    for pattern, repl in swaps.items():
        text = re.sub(pattern, repl, text, flags=re.I)
    return text


# ── STREAMING SENTENCE SPLITTER ───────────────────────────────────────────────
def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Feeds LLM tokens in, emits complete sentences as soon as they end."""

    BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+(?=\S)|\n+")

    def __init__(self, emit):
        self.emit = emit
        self.buf  = ""

    def feed(self, text: str):
        self.buf += text
        while True:
            m = self.BOUNDARY.search(self.buf)
            if not m:
                return
            sentence, self.buf = self.buf[:m.end()], self.buf[m.end():]
            self._emit(sentence)

    def flush(self):
        self._emit(self.buf)
        self.buf = ""

    def _emit(self, sentence):
        s = clean_for_speech(sentence)
        if s:
            self.emit(s)


# ── ORCHESTRATOR: one streaming LLM call ──────────────────────────────────────
class OrchestratorAgent:
    """
    Single streaming call per turn: plain text out, emitted sentence by sentence
    so TTS can start speaking while the rest is still generating.
    """

    def run(self, user_input: str, memories: list, history: list,
            tool_result: str = None, on_sentence=None) -> str:
        # Memories and tool output go in the LAST message only, keeping the
        # system prompt + history prefix stable so Ollama can reuse it.
        context = ""
        if memories:
            context += "<memory>Facts the user has told me: " + " | ".join(memories) + "</memory>\n"
        if tool_result:
            context += f"<tool_result>{tool_result}</tool_result>\n"

        sentences = []

        def emit(s):
            sentences.append(s)
            if on_sentence:
                on_sentence(s)

        splitter = SentenceSplitter(emit)
        pending  = ""
        stream = ollama_client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": f"{context}{user_input}"},
            ],
            options=OPTS_CHAT,
            think=False,
            stream=True,
            keep_alive=-1,
        )
        for chunk in stream:
            pending += chunk.message.content or ""
            # Drop any <think>…</think> block; wait while one is still open
            if "<think>" in pending:
                if "</think>" not in pending:
                    continue
                pending = re.sub(r"<think>.*?</think>", "", pending, flags=re.DOTALL)
            splitter.feed(pending)
            pending = ""
        splitter.feed(pending)
        splitter.flush()
        return " ".join(sentences)


# ── PIPELINE COORDINATOR ──────────────────────────────────────────────────────
class AgentPipeline:
    """
    Short-circuit paths, cheapest first:

    Memory command (remember / what do you know / forget) → instant
    Regex router → tool handler (instant, or semantic cache for lookups)
                   → tool found nothing → falls through to the LLM
    Orchestrator: ONE streaming LLM call → sentences spoken as they arrive
    Persist to DB (errors are never saved or cached)

    run() returns (response, source, info). source is one of:
      "memory", "tool", "cache" — whole answer, not streamed
      "llm"                     — already delivered sentence-by-sentence via on_sentence
      "error"                   — nothing was streamed
    """

    def __init__(self, db: JarvisDB):
        self.db        = db
        self.tools     = ToolHandler()
        self.orch      = OrchestratorAgent()
        self.sem_cache = SemanticCache(db)

    def run(self, user_input: str, on_status=None, on_sentence=None):
        def status(msg, color="#d29922"):
            if on_status:
                on_status(msg, color)

        # Per-stage timing printed to the terminal — shows where latency comes from
        t_start = t_last = time.time()
        def timing(stage):
            nonlocal t_last
            now = time.time()
            print(f"[Pipeline] {stage:<22} {now - t_last:6.1f}s  (total {now - t_start:.1f}s)")
            t_last = now

        self.db.log_security(user_input, threat_level(user_input), "")

        # ── 1. MEMORY COMMANDS ────────────────────────────────────────────
        answer = self._memory_command(user_input)
        if answer:
            self._save(user_input, answer, "memory")
            timing("memory command")
            return answer, "memory", {"tool_name": "memory"}

        # ── 2. ROUTER + TOOLS ─────────────────────────────────────────────
        tool_name, entities = route(user_input)
        tool_result = None
        if tool_name:
            if tool_name in CACHEABLE_TOOLS:
                status("● SEARCHING CACHE", "#89b4fa")
                cached, score = self.sem_cache.lookup(user_input)
                timing("cache lookup")
                if cached:
                    self._save(user_input, cached, tool_name)
                    return cached, "cache", {"tool_name": tool_name, "cache_score": score}

            status(f"● {tool_name.replace('_', ' ').upper()}", "#a6e3a1")
            answer = self.tools.handle(tool_name, entities)
            timing(f"tool:{tool_name}")
            if answer and tool_name not in SUMMARISE_TOOLS:
                self._save(user_input, answer, tool_name)
                if tool_name in CACHEABLE_TOOLS:
                    self.sem_cache.store(user_input, answer, tool_name)
                return answer, "tool", {"tool_name": tool_name, "entities": entities}
            tool_result = answer   # None (tool found nothing) or text for the LLM to summarise

        # ── 3. ORCHESTRATOR (one streaming LLM call) ──────────────────────
        status("● THINKING", "#d29922")
        memories = self.db.get_memories(limit=MAX_MEMORIES)
        history  = self.db.get_history(limit=MAX_HISTORY)
        timing("context")

        first = [True]
        def on_sent(s):
            if first[0]:
                timing("LLM first sentence")
                first[0] = False
            if on_sentence:
                on_sentence(s)

        try:
            response = self.orch.run(user_input, memories, history, tool_result, on_sentence=on_sent)
        except Exception as e:
            print(f"[Pipeline] LLM error: {type(e).__name__}: {e}")
            if first[0]:   # nothing streamed yet
                return (f"I'm having trouble reaching my language model, sir. "
                        f"Check that Ollama is running."), "error", {"error": str(e)}
            response = ""
        timing(f"LLM done ({len(history)} hist)")

        if not response.strip():
            return "I'm afraid I lost my train of thought, sir.", "error", {}

        # ── 4. PERSIST ────────────────────────────────────────────────────
        self._save(user_input, response, tool_name or "chat")
        if tool_name in CACHEABLE_TOOLS and tool_result:
            self.sem_cache.store(user_input, response, tool_name)

        return response, "llm", {"tool_name": tool_name or "none"}

    def _save(self, user_input, answer, category):
        self.db.save_turn("user", user_input, category)
        self.db.save_turn("assistant", answer, category)

    def _memory_command(self, text: str):
        text = text.strip()
        m = REMEMBER_RE.match(text)
        if m:
            fact = m.group("fact").strip().rstrip(".")
            self.db.save_memory(fact)
            return f"Noted, sir. I'll remember that {to_second_person(fact)}."
        if RECALL_RE.search(text):
            facts = self.db.get_memories(limit=20)
            if not facts:
                return "You haven't asked me to remember anything yet, sir."
            return "Here is what I know: " + ". ".join(to_second_person(f) for f in facts) + "."
        if FORGET_RE.match(text):
            fact = self.db.delete_last_memory()
            if fact:
                return f"Done. I've forgotten that {to_second_person(fact)}."
            return "There's nothing for me to forget, sir."
        return None
