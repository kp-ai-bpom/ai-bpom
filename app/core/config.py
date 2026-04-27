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
        '{"public":["propinsi_tm","kabupaten_tm","kecamatan_tm","pangkat_tm","tipepegawai_tm","eselon_tm","pegawai_tm","disabilitas_tm","riwayatjabatan_th","SIAP_SATKER_TOP","jabatan_tm","sk_pegawai_v"],"siap":["R_FUNGSI","T_RIWAYAT_MUTASI","V_PENDIDIKAN_TERAKHIR"],"mantel":["period_employees","periods"]}',
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

    # Chatbot Ambiguity Handling Configuration
    CHATBOT_AMBIGUITY_ENABLED: bool = (
        os.getenv("CHATBOT_AMBIGUITY_ENABLED", "true").lower() == "true"
    )
    CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS", "1024")
    )
    CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS", "512")
    )
    CHATBOT_AMBIGUITY_TEMPERATURE: float = float(
        os.getenv("CHATBOT_AMBIGUITY_TEMPERATURE", "0.1")
    )
    CHATBOT_AMBIGUITY_TIMEOUT_SECONDS: float = float(
        os.getenv("CHATBOT_AMBIGUITY_TIMEOUT_SECONDS", "10")
    )
    CHATBOT_AMBIGUITY_RATE_LIMIT_RETRIES: int = int(
        os.getenv("CHATBOT_AMBIGUITY_RATE_LIMIT_RETRIES", "3")
    )
    CHATBOT_AMBIGUITY_SESSION_TTL_SECONDS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_SESSION_TTL_SECONDS", "300")
    )
    CHATBOT_AMBIGUITY_AUTO_RESOLVE_WINDOW: int = int(
        os.getenv("CHATBOT_AMBIGUITY_AUTO_RESOLVE_WINDOW", "3")
    )
    CHATBOT_AMBIGUITY_MAX_HISTORY_TURNS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_MAX_HISTORY_TURNS", "3")
    )
    CHATBOT_USE_TOON: bool = (
        os.getenv("CHATBOT_USE_TOON", "true").lower() == "true"
    )
    CHATBOT_PROCEDURAL_ENABLED: bool = (
        os.getenv("CHATBOT_PROCEDURAL_ENABLED", "true").lower() == "true"
    )
    CHATBOT_PROCEDURAL_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_PROCEDURAL_SIMILARITY_THRESHOLD", "0.85")
    )
    CHATBOT_PROCEDURAL_TTL_DAYS: int = int(
        os.getenv("CHATBOT_PROCEDURAL_TTL_DAYS", "90")
    )
    CHATBOT_PROCEDURAL_RESET_KEYWORDS: str = os.getenv(
        "CHATBOT_PROCEDURAL_RESET_KEYWORDS",
        "reset preferensi,lupakan preferensi,lupakan aturan,hapus preferensi",
    )

    # Chatbot SQL Validation Pipeline Configuration (Tahap 5)
    # CHATBOT_VALIDATION_LEVEL: "full" (default; execution + semantic),
    # "execution_only" (lewati judge), atau "none" (matikan total — perilaku
    # legacy persis seperti sebelum Tahap 5).
    CHATBOT_VALIDATION_LEVEL: str = os.getenv("CHATBOT_VALIDATION_LEVEL", "full")
    CHATBOT_VALIDATION_MAX_EXECUTION_ITERATIONS: int = int(
        os.getenv("CHATBOT_VALIDATION_MAX_EXECUTION_ITERATIONS", "3")
    )
    # CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS: di bawah kebijakan PARTIAL
    # fixed-flow saat ini (judge#1 → revise → re-execute → judge#2 wajib),
    # nilai variabel ini berfungsi sebagai sakelar boolean: 0 = nonaktifkan
    # loop semantik (judge tidak dipanggil sama sekali), >= 1 = aktifkan satu
    # putaran lengkap revise+rejudge. Nilai > 1 saat ini setara dengan 1.
    CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS: int = int(
        os.getenv("CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS", "1")
    )
    CHATBOT_VALIDATION_TIMEOUT_SECONDS: float = float(
        os.getenv("CHATBOT_VALIDATION_TIMEOUT_SECONDS", "30")
    )
    CHATBOT_VALIDATION_TRIVIAL_SKIP: bool = (
        os.getenv("CHATBOT_VALIDATION_TRIVIAL_SKIP", "true").lower() == "true"
    )
    CHATBOT_VALIDATION_REFINER_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_VALIDATION_REFINER_MAX_TOKENS", "2048")
    )
    CHATBOT_VALIDATION_JUDGE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_VALIDATION_JUDGE_MAX_TOKENS", "1536")
    )
    CHATBOT_VALIDATION_EXECUTION_TIMEOUT_MS: int = int(
        os.getenv("CHATBOT_VALIDATION_EXECUTION_TIMEOUT_MS", "5000")
    )

    @field_validator("POSTGRES_URI", mode="before")
    @classmethod
    def _normalize_postgres_uri(cls, value: str) -> str:
        if isinstance(value, str):
            return value.rstrip("/")
        return value


settings = Settings()
