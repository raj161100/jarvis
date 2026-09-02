import json
import numpy as np

from config import SEMANTIC_THRESHOLD, EMBED_MODEL, NO_CACHE

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class SemanticCache:
    """
    Semantic similarity cache using sentence embeddings.

    How it works:
      1. Every query is encoded into a 384-dimensional vector (embedding)
         that captures its *meaning*, not just its exact words.
      2. On lookup: compute cosine similarity between the new query and all
         stored embeddings using a single matrix multiply (fast even for
         thousands of entries).
      3. If best similarity >= threshold, return the cached response.
      4. On store: encode and persist the new (query, response, embedding)
         to both memory and SQLite so it survives reboots.

    Cosine similarity intuition:
      "what time is it"    ~= "what's the time"          → sim ~0.97 → HIT
      "who is Einstein"    ~= "tell me about Einstein"   → sim ~0.94 → HIT
      "play jazz music"    vs "what is quantum physics"  → sim ~0.12 → MISS

    Falls back to no-op (always miss) if sentence-transformers not installed.
    """

    def __init__(self, db, threshold=SEMANTIC_THRESHOLD):
        self.db        = db
        self.threshold = threshold
        self._queries   = []   # original query strings
        self._responses = []   # corresponding responses
        self._matrix    = None # (N x 384) float32 numpy matrix; None until first entry

        if not ST_AVAILABLE:
            print("[SemanticCache] sentence-transformers not installed — cache disabled.")
            print("[SemanticCache] Install with: pip install sentence-transformers")
            self.embedder = None
            return

        print(f"[Jarvis] Loading embedding model ({EMBED_MODEL})...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("[Jarvis] Embedding model ready.")
        self._load_from_db()

    def _load_from_db(self):
        """Restore cached embeddings from SQLite on startup."""
        rows = self.db.get_all_semantic_cache()
        embeddings = []
        for query, response, emb_json in rows:
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            self._queries.append(query)
            self._responses.append(response)
            embeddings.append(emb)
        if embeddings:
            self._matrix = np.stack(embeddings)
        print(f"[Jarvis] Loaded {len(self._queries)} semantic cache entries.")

    def lookup(self, query: str):
        """
        Returns (cached_response, similarity_score) on hit.
        Returns (None, 0.0) on miss or if embedder unavailable.

        Time: ~50ms (embedding) + <1ms (matrix multiply for 1000 entries).
        """
        if self.embedder is None or self._matrix is None:
            return None, 0.0

        # normalize_embeddings=True → L2 norm = 1 → dot product == cosine similarity
        query_emb = self.embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # One matrix multiply gives cosine similarity to every cached entry
        scores     = self._matrix @ query_emb   # shape: (N,)
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            return self._responses[best_idx], best_score

        return None, best_score

    def store(self, query: str, response: str, category: str):
        """
        Embed and persist a new (query, response) pair.
        Skips time-sensitive categories that should never be cached.
        """
        if self.embedder is None or category in NO_CACHE:
            return

        emb = self.embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        # Update in-memory structures
        self._queries.append(query)
        self._responses.append(response)

        if self._matrix is None:
            self._matrix = emb.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, emb])

        # Persist to SQLite for next boot
        self.db.store_semantic_cache(
            query, response, category,
            json.dumps(emb.tolist())
        )
