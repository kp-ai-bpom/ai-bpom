import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")

    # PostgreSQL Configuration
    POSTGRES_URI: str = os.getenv(
        "POSTGRES_URI", "postgresql+asyncpg://user:password@host:port/dbname"
    ).rstrip("/")

    # LLM Provider Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AI_BASE_URL: Optional[str] = os.getenv("AI_BASE_URL", None)
    AI_INSTRUCT_MODEL_NAME: str = os.getenv("AI_INSTRUCT_MODEL_NAME", "gpt-4o-mini")
    AI_THINK_MODEL_NAME: str = os.getenv("AI_THINK_MODEL_NAME", "gpt-4o-mini")
    AI_DEEP_THINK_MODEL_NAME: str = os.getenv("AI_DEEP_THINK_MODEL_NAME", "gpt-4o-mini")
    AI_EMBEDDINGS_MODEL_NAME: str = os.getenv(
        "AI_EMBEDDINGS_MODEL_NAME", "text-embedding-large"
    )

    # LLM Adapter Configuration
    LLM_PROVIDER_PRIORITY: str = os.getenv("LLM_PROVIDER_PRIORITY", "openai,anthropic")
    LLM_FALLBACK_ENABLED: bool = (
        os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    )

    # Chatbot Semantic Memory Pipeline Configuration
    CHATBOT_VECTOR_TABLE: str = os.getenv("CHATBOT_VECTOR_TABLE", "knowledge_entities")
    CHATBOT_SQL_TIMEOUT_MS: int = int(os.getenv("CHATBOT_SQL_TIMEOUT_MS", "8000"))
    CHATBOT_TOP_K_PER_KEYWORD: int = int(os.getenv("CHATBOT_TOP_K_PER_KEYWORD", "15"))
    CHATBOT_MAX_RETRIEVED_TABLES: int = int(os.getenv("CHATBOT_MAX_RETRIEVED_TABLES", "5"))
    CHATBOT_TABLE_WEIGHT: float = float(os.getenv("CHATBOT_TABLE_WEIGHT", "1.5"))
    CHATBOT_COLUMN_WEIGHT: float = float(os.getenv("CHATBOT_COLUMN_WEIGHT", "1.0"))
    CHATBOT_RETRIEVAL_THRESHOLD: float = float(
        os.getenv("CHATBOT_RETRIEVAL_THRESHOLD", "0.3")
    )
    CHATBOT_COLUMN_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_COLUMN_SIMILARITY_THRESHOLD", "0.25")
    )
    CHATBOT_MAX_CONTEXT_CHARS: int = int(os.getenv("CHATBOT_MAX_CONTEXT_CHARS", "40000"))
    CHATBOT_SQL_GENERATION_RETRIES: int = int(
        os.getenv("CHATBOT_SQL_GENERATION_RETRIES", "4")
    )
    CHATBOT_SQL_GENERATOR_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_SQL_GENERATOR_MAX_TOKENS", "3072")
    )
    CHATBOT_SQL_DEFAULT_SCHEMA: str = os.getenv(
        "CHATBOT_SQL_DEFAULT_SCHEMA", "public"
    )
    CHATBOT_KEYWORD_RETRIES: int = int(os.getenv("CHATBOT_KEYWORD_RETRIES", "3"))
    CHATBOT_SAMPLE_ROWS_PER_TABLE: int = int(
        os.getenv("CHATBOT_SAMPLE_ROWS_PER_TABLE", "3")
    )
    CHATBOT_ALLOWED_TABLES_JSON: str = os.getenv(
        "CHATBOT_ALLOWED_TABLES_JSON",
        '{"public":["propinsi_tm","kabupaten_tm","kecamatan_tm","pangkat_tm","tipepegawai_tm","eselon_tm","pegawai_tm","disabilitas_tm","riwayatjabatan_th","SIAP_SATKER_TOP","jabatan_tm","sk_pegawai_v"],"siap":["R_FUNGSI","T_RIWAYAT_MUTASI","V_PENDIDIKAN_TERAKHIR"]}',
    )

    # Chatbot Question Rewriting Configuration
    CHATBOT_REWRITE_ENABLED: bool = (
        os.getenv("CHATBOT_REWRITE_ENABLED", "true").lower() == "true"
    )
    CHATBOT_REWRITE_WORKING_MEMORY_WINDOW: int = int(
        os.getenv("CHATBOT_REWRITE_WORKING_MEMORY_WINDOW", "4")
    )
    CHATBOT_REWRITE_MAX_EPISODIC_MATCHES: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_EPISODIC_MATCHES", "3")
    )
    CHATBOT_REWRITE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_REWRITE_SIMILARITY_THRESHOLD", "0.3")
    )
    CHATBOT_REWRITE_MAX_EPISODIC_SNIPPET_CHARS: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_EPISODIC_SNIPPET_CHARS", "1500")
    )
    CHATBOT_REWRITE_MAX_WORKING_SNIPPET_CHARS: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_WORKING_SNIPPET_CHARS", "1000")
    )
    CHATBOT_REWRITE_LLM_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_REWRITE_LLM_MAX_TOKENS", "1200")
    )
    CHATBOT_REWRITE_SOURCE: str = os.getenv(
        "CHATBOT_REWRITE_SOURCE", "chatbot_api"
    )

    @field_validator("POSTGRES_URI", mode="before")
    @classmethod
    def _normalize_postgres_uri(cls, value: str) -> str:
        if isinstance(value, str):
            return value.rstrip("/")
        return value


settings = Settings()
