from dataclasses import dataclass
from functools import lru_cache

from app.domains.chatbot.core.config import (
    _resolve_stage_temperature,
    chatbot_settings as settings,
)

from .types import ValidationLevel


@dataclass(frozen=True)
class SQLValidationConfig:
    level: ValidationLevel
    max_execution_iterations: int
    # Catatan operator: di bawah kebijakan PARTIAL fixed-flow saat ini
    # (judge#1 → revise → re-execute → judge#2 wajib), parameter ini
    # berperilaku sebagai sakelar boolean — nilai 0 menonaktifkan loop
    # semantik sepenuhnya, nilai >= 1 mengaktifkan satu putaran lengkap
    # revise+re-judge. Tidak ada efek tambahan untuk nilai > 1.
    max_semantic_iterations: int
    timeout_seconds: float
    trivial_skip: bool
    refiner_max_tokens: int
    judge_max_tokens: int
    refiner_llm_temperature: float
    judge_llm_temperature: float
    execution_timeout_ms: int


@lru_cache(maxsize=1)
def get_sql_validation_config() -> SQLValidationConfig:
    return SQLValidationConfig(
        level=ValidationLevel.from_string(settings.CHATBOT_VALIDATION_LEVEL),
        max_execution_iterations=max(
            1, settings.CHATBOT_VALIDATION_MAX_EXECUTION_ITERATIONS
        ),
        max_semantic_iterations=max(
            0, settings.CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS
        ),
        timeout_seconds=max(1.0, float(settings.CHATBOT_VALIDATION_TIMEOUT_SECONDS)),
        trivial_skip=settings.CHATBOT_VALIDATION_TRIVIAL_SKIP,
        refiner_max_tokens=max(512, settings.CHATBOT_VALIDATION_REFINER_MAX_TOKENS),
        judge_max_tokens=max(512, settings.CHATBOT_VALIDATION_JUDGE_MAX_TOKENS),
        refiner_llm_temperature=_resolve_stage_temperature(
            "CHATBOT_VALIDATION_REFINER_LLM_TEMPERATURE"
        ),
        judge_llm_temperature=_resolve_stage_temperature(
            "CHATBOT_VALIDATION_JUDGE_LLM_TEMPERATURE"
        ),
        execution_timeout_ms=max(
            500, settings.CHATBOT_VALIDATION_EXECUTION_TIMEOUT_MS
        ),
    )
