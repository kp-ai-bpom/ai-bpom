import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")

    # PostgreSQL Configuration
    POSTGRES_URI: str = os.getenv(
        "POSTGRES_URI", "postgresql+asyncpg://user:password@host:port/dbname"
    )

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


settings = Settings()
