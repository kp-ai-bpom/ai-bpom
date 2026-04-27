from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings


@dataclass(frozen=True)
class AmbiguityConfig:
    enabled: bool
    detection_max_tokens: int
    refine_max_tokens: int
    temperature: float
    timeout_seconds: float
    rate_limit_retries: int
    session_ttl_seconds: int
    auto_resolve_window: int
    max_history_turns: int
    procedural_enabled: bool
    procedural_similarity_threshold: float
    procedural_ttl_days: int
    procedural_reset_keywords: tuple[str, ...]


@lru_cache(maxsize=1)
def get_ambiguity_config() -> AmbiguityConfig:
    raw_keywords = settings.CHATBOT_PROCEDURAL_RESET_KEYWORDS or ""
    reset_keywords = tuple(
        kw.strip().lower()
        for kw in raw_keywords.split(",")
        if kw and kw.strip()
    )
    return AmbiguityConfig(
        enabled=settings.CHATBOT_AMBIGUITY_ENABLED,
        detection_max_tokens=max(256, settings.CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS),
        refine_max_tokens=max(128, settings.CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS),
        temperature=max(0.0, min(1.0, settings.CHATBOT_AMBIGUITY_TEMPERATURE)),
        timeout_seconds=max(1.0, float(settings.CHATBOT_AMBIGUITY_TIMEOUT_SECONDS)),
        rate_limit_retries=max(0, settings.CHATBOT_AMBIGUITY_RATE_LIMIT_RETRIES),
        session_ttl_seconds=max(60, settings.CHATBOT_AMBIGUITY_SESSION_TTL_SECONDS),
        auto_resolve_window=max(1, settings.CHATBOT_AMBIGUITY_AUTO_RESOLVE_WINDOW),
        max_history_turns=max(1, settings.CHATBOT_AMBIGUITY_MAX_HISTORY_TURNS),
        procedural_enabled=settings.CHATBOT_PROCEDURAL_ENABLED,
        procedural_similarity_threshold=max(
            0.0, min(1.0, settings.CHATBOT_PROCEDURAL_SIMILARITY_THRESHOLD)
        ),
        procedural_ttl_days=max(1, settings.CHATBOT_PROCEDURAL_TTL_DAYS),
        procedural_reset_keywords=reset_keywords,
    )
