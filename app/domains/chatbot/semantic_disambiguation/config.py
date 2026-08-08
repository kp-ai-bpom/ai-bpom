from dataclasses import dataclass
from functools import lru_cache

from app.domains.chatbot.core.config import (
    _resolve_stage_temperature,
    chatbot_settings as settings,
)


@dataclass(frozen=True)
class AmbiguityConfig:
    enabled: bool
    detection_max_tokens: int
    refine_max_tokens: int
    # Legacy field — dipertahankan untuk kompatibilitas dengan kode/eval lama
    # yang membaca ``config.temperature`` langsung. Untuk callsite LLM gunakan
    # ``llm_temperature`` yang sudah mengikuti hierarki resolver per-stage.
    temperature: float
    llm_temperature: float
    timeout_seconds: float
    rate_limit_retries: int
    session_ttl_seconds: int
    auto_resolve_window: int
    max_history_turns: int
    procedural_enabled: bool
    procedural_similarity_threshold: float
    procedural_ttl_days: int
    procedural_reset_keywords: tuple[str, ...]
    # ── Stage 1 UQ (Uncertainty Quantification) ─────────────────────────────
    # Konstanta FROZEN dari kalibrasi 60 testcase contextual (ROC AUC=0.8928,
    # F1=0.873). Lihat ``ambiguity/uq.py`` dan skripsi §3.1.4. Env vars
    # disediakan sebagai escape hatch operator, BUKAN untuk re-tuning tanpa
    # re-kalibrasi.
    uq_enabled: bool
    uq_m_sampling: int
    uq_t_sampling: float
    uq_tau_cluster: float
    uq_tau_u: float
    uq_sample_max_tokens: int
    uq_embedding_model: str


@lru_cache(maxsize=1)
def get_ambiguity_config() -> AmbiguityConfig:
    raw_keywords = settings.CHATBOT_PROCEDURAL_RESET_KEYWORDS or ""
    reset_keywords = tuple(
        kw.strip().lower()
        for kw in raw_keywords.split(",")
        if kw and kw.strip()
    )
    legacy_ambiguity_temp = max(
        0.0, min(1.0, settings.CHATBOT_AMBIGUITY_TEMPERATURE)
    )
    return AmbiguityConfig(
        enabled=settings.CHATBOT_AMBIGUITY_ENABLED,
        detection_max_tokens=max(256, settings.CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS),
        refine_max_tokens=max(128, settings.CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS),
        temperature=legacy_ambiguity_temp,
        llm_temperature=_resolve_stage_temperature(
            "CHATBOT_AMBIGUITY_LLM_TEMPERATURE",
            legacy_default=legacy_ambiguity_temp,
        ),
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
        # FROZEN constants — di-clamp defensif tapi default mengikuti kalibrasi.
        uq_enabled=settings.CHATBOT_AMBIGUITY_UQ_ENABLED,
        uq_m_sampling=max(2, settings.CHATBOT_AMBIGUITY_UQ_M_SAMPLING),
        uq_t_sampling=max(0.0, min(2.0, settings.CHATBOT_AMBIGUITY_UQ_T_SAMPLING)),
        uq_tau_cluster=max(0.0, min(1.0, settings.CHATBOT_AMBIGUITY_UQ_TAU_CLUSTER)),
        uq_tau_u=max(0.0, min(1.0, settings.CHATBOT_AMBIGUITY_UQ_TAU_U)),
        uq_sample_max_tokens=max(128, settings.CHATBOT_AMBIGUITY_UQ_SAMPLE_MAX_TOKENS),
        uq_embedding_model=(
            (settings.CHATBOT_AMBIGUITY_EMBEDDING_MODEL or "").strip()
            or "text-embedding-3-small"
        ),
    )
