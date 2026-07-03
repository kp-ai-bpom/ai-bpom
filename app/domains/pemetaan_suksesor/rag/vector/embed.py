import logging

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# Module-level embedding cache for pre-computation optimisation.
# Keyed by query string, stores the embedding vector.
_embedding_cache: dict[str, list[float]] = {}


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        client_kwargs: dict = {
            "api_key": settings.OPENAI_API_KEY,
            "default_headers": {"User-Agent": "ai-bpom/1.0"},
        }
        if settings.AI_BASE_URL:
            client_kwargs["base_url"] = settings.AI_BASE_URL
        _client = AsyncOpenAI(**client_kwargs)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.RAG_EMBEDDING_MODEL,
        input=texts,
    )
    return [d.embedding for d in response.data]


async def embed_single(text: str) -> list[float]:
    results = await embed_texts([text])
    return results[0]


# ── Embedding cache functions ──────────────────────────────────────────


def get_cached_embedding(query: str) -> list[float] | None:
    """Return the cached embedding for *query*, or ``None`` on cache miss."""
    return _embedding_cache.get(query)


def cache_embedding(query: str, embedding: list[float]) -> None:
    """Store a single query → embedding pair in the module-level cache."""
    _embedding_cache[query] = embedding


async def precompute_embeddings(queries: list[str]) -> int:
    """Batch-embed queries that are not yet cached.

    Returns the number of newly embedded queries.
    """
    to_embed = [q for q in queries if q not in _embedding_cache]
    if not to_embed:
        return 0
    try:
        embeddings = await embed_texts(to_embed)
        for query, embedding in zip(to_embed, embeddings):
            _embedding_cache[query] = embedding
        logger.info(
            "Pre-computed %d embedding(s); cache size now %d",
            len(to_embed),
            len(_embedding_cache),
        )
        return len(to_embed)
    except Exception:
        logger.warning("Pre-computation of embeddings failed", exc_info=True)
        return 0


def clear_embedding_cache() -> None:
    """Remove all entries from the embedding cache."""
    _embedding_cache.clear()