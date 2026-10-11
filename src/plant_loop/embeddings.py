"""Embedders + the vector index behind the Retriever port.

Two embedders:

- :class:`HashingEmbedder` — deterministic, offline, dependency-free:
  token hashing into a fixed-width vector. A stand-in with honest
  limits — it catches keyword-shaped similarity, which is what the
  SOP/history retrieval needs offline; it is not semantic search.
- :class:`ProviderEmbedder` — any OpenAI-compatible ``/embeddings``
  endpoint (including Meta Model API's surface where offered, or a
  local server), used when ``EMBEDDER=provider``.

Two indexes with one interface: :class:`PgVectorIndex` (production —
the same PostgreSQL as the records) and :class:`InMemoryVectorIndex`
(the test double). Selection happens in :func:`get_vector_index`;
nothing else in the package knows which one it holds.
"""

from __future__ import annotations

import hashlib
import math
import re

from .config import env_str, load_dotenv
from .errors import not_configured_error, translate_provider_error

_TOKEN = re.compile(r"[a-z0-9]+")
_DIMENSIONS = 256


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class HashingEmbedder:
    """Offline stand-in: hashed token counts, L2-normalised."""

    name = "hashing"
    model = "hashing-256"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * _DIMENSIONS
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % _DIMENSIONS
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class ProviderEmbedder:
    """Embeddings from an OpenAI-compatible endpoint."""

    name = "provider"

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.model = model or env_str("EMBEDDING_MODEL", "text-embedding-3-small")
        self._base_url = env_str("OPENAI_BASE_URL")
        self._api_key = env_str("OPENAI_API_KEY")
        if not self._api_key and not self._base_url:
            raise not_configured_error(
                "provider embedder",
                "OPENAI_API_KEY",
                "Or set EMBEDDER=hashing to use the offline stand-in.",
            )

    def embed(self, text: str) -> list[float]:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "the provider embedder needs the openai SDK, which is not installed. "
                "Install the optional llm extra: uv sync --extra llm "
                "(pip fallback: pip install -e \".[llm]\"), then re-run."
            ) from exc
        try:
            client = openai.OpenAI(api_key=self._api_key or "not-needed", base_url=self._base_url)
            response = client.embeddings.create(model=self.model, input=text[:8000])
        except Exception as exc:
            raise translate_provider_error(exc, backend="embeddings", base_url=self._base_url) from exc
        return list(response.data[0].embedding)


def get_embedder() -> HashingEmbedder | ProviderEmbedder:
    load_dotenv()
    if (env_str("EMBEDDER", "hashing") or "hashing").strip().lower() == "provider":
        return ProviderEmbedder()
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorIndex:
    """The test double: vectors in a dict, brute-force cosine."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], str]] = {}

    def add(self, chunk_id: str, vector: list[float], model: str) -> None:
        self._vectors[chunk_id] = (vector, model)

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 5
    ) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, cosine(vector, stored))
            for chunk_id, (stored, _model) in self._vectors.items()
            if chunk_id != exclude_id
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


class PgVectorIndex:
    """Production index: KB chunk vectors in pgvector."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self._url)

    @staticmethod
    def _literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    def add(self, chunk_id: str, vector: list[float], model: str) -> None:
        from .migrations import run_migrations

        with self._connect() as conn:
            run_migrations(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO kb_embeddings (chunk_id, model, embedding) VALUES (%s, %s, %s::vector) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET model = EXCLUDED.model, embedding = EXCLUDED.embedding",
                    (chunk_id, model, self._literal(vector)),
                )
            conn.commit()

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 5
    ) -> list[tuple[str, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, 1 - (embedding <=> %s::vector) AS similarity "
                    "FROM kb_embeddings WHERE chunk_id <> %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (self._literal(vector), exclude_id or "", self._literal(vector), limit),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]


def get_vector_index() -> InMemoryVectorIndex | PgVectorIndex:
    load_dotenv()
    url = env_str("DATABASE_URL", "")
    if url:
        return PgVectorIndex(url)
    return InMemoryVectorIndex()
