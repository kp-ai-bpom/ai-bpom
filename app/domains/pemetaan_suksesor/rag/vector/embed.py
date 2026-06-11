from openai import AsyncOpenAI

from app.core.config import settings

_client: AsyncOpenAI | None = None


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