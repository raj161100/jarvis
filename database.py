import sqlite3
import hashlib
import threading
import uuid
import json

from config import CACHE_TTL_HOURS, NO_CACHE, MAX_HISTORY, CACHEABLE_TOOLS

SESSION_ID = str(uuid.uuid4())[:8]


class JarvisDB:
    """
    SQLite-backed persistence layer.
    Tables:
        memories          — long-term facts Jarvis remembers about the user
        conversations     — full session history across reboots
        response_cache    — exact-match response cache with TTL
        semantic_cache    — embedding vectors for semantic similarity cache
        security_log      — audit trail of all security checks
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    fact      TEXT NOT NULL,
                    category  TEXT DEFAULT 'general',
                    importance INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role       TEXT,
                    content    TEXT,
                    category   TEXT
                );

                CREATE TABLE IF NOT EXISTS response_cache (
                    input_hash TEXT PRIMARY KEY,
                    input_text TEXT,
                    response   TEXT,
                    category   TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    hit_count  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    query      TEXT,
                    response   TEXT,
                    category   TEXT,
                    embedding  TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    hit_count  INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS security_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id   TEXT,
                    input_text   TEXT,
                    threat_level TEXT,
                    reason       TEXT
                );
            """)

    # ── MEMORY ────────────────────────────────────────────────────────────────
    def save_memory(self, fact, category="general", importance=1):
        with self.lock:
            self.conn.execute(
                "INSERT INTO memories (fact, category, importance) VALUES (?, ?, ?)",
                (fact, category, importance)
            )
            self.conn.commit()

    def get_memories(self, limit=8):
        with self.lock:
            rows = self.conn.execute(
                "SELECT fact FROM memories ORDER BY importance DESC, timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [r[0] for r in rows]

    def delete_last_memory(self):
        """Removes the most recently saved fact. Returns its text, or None if there were none."""
        with self.lock:
            row = self.conn.execute("SELECT id, fact FROM memories ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            self.conn.execute("DELETE FROM memories WHERE id = ?", (row[0],))
            self.conn.commit()
        return row[1]

    # ── CONVERSATION HISTORY ──────────────────────────────────────────────────
    def save_turn(self, role, content, category="general"):
        with self.lock:
            self.conn.execute(
                "INSERT INTO conversations (session_id, role, content, category) VALUES (?, ?, ?, ?)",
                (SESSION_ID, role, content, category)
            )
            self.conn.commit()

    def get_history(self, limit=MAX_HISTORY):
        """Returns last N turns as ollama-compatible dicts, oldest first."""
        with self.lock:
            # Order by id, not timestamp — timestamps have 1s resolution, so a
            # user/assistant pair saved in the same second would come back shuffled
            rows = self.conn.execute(
                "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    # ── EXACT-MATCH CACHE ─────────────────────────────────────────────────────
    def get_cached(self, text):
        h = self._hash(text)
        with self.lock:
            row = self.conn.execute(
                """SELECT response FROM response_cache
                   WHERE input_hash = ?
                   AND datetime(created_at, ? || ' hours') > datetime('now')""",
                (h, f"+{CACHE_TTL_HOURS}")
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE response_cache SET hit_count = hit_count + 1 WHERE input_hash = ?",
                    (h,)
                )
                self.conn.commit()
                return row[0]
        return None

    def cache_response(self, text, response, category):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO response_cache (input_hash, input_text, response, category) VALUES (?,?,?,?)",
                (self._hash(text), text, response, category)
            )
            self.conn.commit()

    # ── SEMANTIC CACHE ────────────────────────────────────────────────────────
    def get_all_semantic_cache(self):
        """Load stored embeddings on startup for in-memory matrix.
        Only context-free lookups — older versions also cached chat answers and errors."""
        placeholders = ",".join("?" * len(CACHEABLE_TOOLS))
        with self.lock:
            return self.conn.execute(
                f"SELECT query, response, embedding FROM semantic_cache WHERE category IN ({placeholders})",
                tuple(CACHEABLE_TOOLS)
            ).fetchall()

    def store_semantic_cache(self, query, response, category, embedding_json):
        with self.lock:
            self.conn.execute(
                "INSERT INTO semantic_cache (query, response, category, embedding) VALUES (?,?,?,?)",
                (query, response, category, embedding_json)
            )
            self.conn.commit()

    # ── SECURITY LOG ──────────────────────────────────────────────────────────
    def log_security(self, text, threat_level, reason):
        with self.lock:
            self.conn.execute(
                "INSERT INTO security_log (session_id, input_text, threat_level, reason) VALUES (?,?,?,?)",
                (SESSION_ID, text, threat_level, reason)
            )
            self.conn.commit()

    # ── UTIL ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _hash(text):
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
