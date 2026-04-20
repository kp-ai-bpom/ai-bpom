from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings


@dataclass(frozen=True)
class QuestionRewritingConfig:
    enabled: bool
    working_memory_window: int
    max_episodic_matches: int
    episodic_similarity_threshold: float
    max_episodic_snippet_chars: int
    max_working_snippet_chars: int
    llm_max_tokens: int
    source: str


@lru_cache(maxsize=1)
def get_question_rewriting_config() -> QuestionRewritingConfig:
    return QuestionRewritingConfig(
        enabled=settings.CHATBOT_REWRITE_ENABLED,
        working_memory_window=max(1, settings.CHATBOT_REWRITE_WORKING_MEMORY_WINDOW),
        max_episodic_matches=max(1, settings.CHATBOT_REWRITE_MAX_EPISODIC_MATCHES),
        episodic_similarity_threshold=max(
            0.0, min(1.0, settings.CHATBOT_REWRITE_SIMILARITY_THRESHOLD)
        ),
        max_episodic_snippet_chars=max(
            300, settings.CHATBOT_REWRITE_MAX_EPISODIC_SNIPPET_CHARS
        ),
        max_working_snippet_chars=max(
            200, settings.CHATBOT_REWRITE_MAX_WORKING_SNIPPET_CHARS
        ),
        llm_max_tokens=max(256, settings.CHATBOT_REWRITE_LLM_MAX_TOKENS),
        source=settings.CHATBOT_REWRITE_SOURCE,
    )
