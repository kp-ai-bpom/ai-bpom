from dataclasses import dataclass
from functools import lru_cache

from app.domains.chatbot.core.config import (
    _resolve_stage_temperature,
    chatbot_settings as settings,
)


@dataclass(frozen=True)
class SQLGeneratorConfig:
    retries: int
    max_tokens: int
    default_schema: str
    llm_temperature: float


@lru_cache(maxsize=1)
def get_sql_generator_config() -> SQLGeneratorConfig:
    return SQLGeneratorConfig(
        retries=max(1, settings.CHATBOT_SQL_GENERATION_RETRIES),
        max_tokens=max(256, settings.CHATBOT_SQL_GENERATOR_MAX_TOKENS),
        default_schema=settings.CHATBOT_SQL_DEFAULT_SCHEMA or "public",
        llm_temperature=_resolve_stage_temperature(
            "CHATBOT_SQL_GENERATOR_LLM_TEMPERATURE"
        ),
    )
