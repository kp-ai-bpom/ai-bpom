"""Konfigurasi shared lintas-domain.

File ini HANYA berisi setting yang dipakai oleh lebih dari satu domain
(``chatbot``, ``pemetaan_suksesor``, ``penilaian_suksesor``):
  - Env / DB connection (``ENV``, ``POSTGRES_URI``)
  - LLM provider base (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` /
    ``AI_BASE_URL`` / ``LLM_PROVIDER`` / model names / embeddings model)
  - Default LLM temperature (``LLM_DEFAULT_TEMPERATURE``)

Setting domain-spesifik diletakkan di domain-nya masing-masing:
  - chatbot          → ``app/domains/chatbot/core/config.py``
  - pemetaan_suksesor → ``app/domains/pemetaan_suksesor/core/config.py``

Aturan: jangan tambahkan field domain-spesifik di file ini. Jika field
hanya dipakai oleh satu domain, taruh di ``core/config.py`` domain
tersebut supaya developer domain lain tidak ikut terbeban.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")

    # PostgreSQL Configuration (shared)
    POSTGRES_URI: str = os.getenv(
        "POSTGRES_URI", "postgresql+asyncpg://user:password@host:port/dbname"
    ).rstrip("/")

    # LLM Provider Configuration (shared base — model names dipakai semua domain)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AI_BASE_URL: Optional[str] = os.getenv("AI_BASE_URL", None)
    AI_INSTRUCT_MODEL_NAME: str = os.getenv("AI_INSTRUCT_MODEL_NAME", "gpt-4o-mini")
    AI_THINK_MODEL_NAME: str = os.getenv("AI_THINK_MODEL_NAME", "gpt-4o-mini")
    AI_DEEP_THINK_MODEL_NAME: str = os.getenv("AI_DEEP_THINK_MODEL_NAME", "gpt-4o-mini")
    AI_EMBEDDINGS_MODEL_NAME: str = os.getenv(
        "AI_EMBEDDINGS_MODEL_NAME", "text-embedding-large"
    )

    # Cross-model comparison: explicit provider override.
    # "openai" / "anthropic" → eksplisit; "" (default) → auto-detect.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")

    # LLM Adapter Configuration
    LLM_PROVIDER_PRIORITY: str = os.getenv("LLM_PROVIDER_PRIORITY", "openai,anthropic")
    LLM_FALLBACK_ENABLED: bool = (
        os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    )

    # Default temperature untuk LLMManager singleton (dipakai semua domain).
    # Default 0.7 = perilaku semula. Per-stage temperature override dilakukan
    # di config domain masing-masing (mis. ``_resolve_stage_temperature`` di
    # ``app/domains/chatbot/core/config.py``).
    LLM_DEFAULT_TEMPERATURE: float = float(
        os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7")
    )

    # ── Neo4j ──────────────────────────────────────────────────────
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # ── MinIO ──────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "password123")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "bpom-documents")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # ── RAG ────────────────────────────────────────────────────────
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    RAG_EMBEDDING_DIMENSIONS: int = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536"))
    RAG_VECTOR_TOP_K: int = int(os.getenv("RAG_VECTOR_TOP_K", "5"))
    RAG_GRAPH_DEPTH: int = int(os.getenv("RAG_GRAPH_DEPTH", "1"))
    RAG_GRAPH_LIMIT: int = int(os.getenv("RAG_GRAPH_LIMIT", "50"))

    @field_validator("POSTGRES_URI", mode="before")
    @classmethod
    def _normalize_postgres_uri(cls, value: str) -> str:
        if isinstance(value, str):
            return value.rstrip("/")
        return value


settings = Settings()
