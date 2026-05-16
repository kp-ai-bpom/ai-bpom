from dataclasses import dataclass
from functools import lru_cache

from app.domains.chatbot.core.config import (
    _resolve_stage_temperature,
    chatbot_settings as settings,
)


@dataclass(frozen=True)
class QuestionRewritingConfig:
    enabled: bool
    working_memory_window: int
    max_episodic_matches: int
    episodic_similarity_threshold: float
    max_episodic_snippet_chars: int
    max_working_snippet_chars: int
    llm_max_tokens: int
    llm_temperature: float
    source: str
    # Opsi B — embedding filter on WorkingMemoryService (advisor concern, §6.3.5)
    wm_embedding_filter_enabled: bool
    wm_embedding_threshold: float
    # Stage 1 UQ (M-sampling rewriter NL→NL). Mirror kalibrasi
    # tests/stage1/scripts/kalibrasi_stage1.py. Saat uq_enabled=True,
    # rewrite() melakukan M LLM call paralel @ uq_t_sampling, cluster
    # cosine single-link @ uq_tau_cluster, dan menghitung H_norm.
    # Kalibrasi script override uq_enabled=False supaya outer M-loop
    # kalibrasi tidak diperbanyak menjadi M×M.
    uq_enabled: bool
    uq_m_sampling: int
    uq_t_sampling: float
    uq_tau_cluster: float
    uq_tau_u: float
    uq_sample_max_tokens: int


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
        llm_temperature=_resolve_stage_temperature(
            "CHATBOT_REWRITE_LLM_TEMPERATURE"
        ),
        source=settings.CHATBOT_REWRITE_SOURCE,
        wm_embedding_filter_enabled=settings.CHATBOT_REWRITE_WM_EMBEDDING_FILTER_ENABLED,
        wm_embedding_threshold=max(
            0.0, min(1.0, settings.CHATBOT_REWRITE_WM_EMBEDDING_THRESHOLD)
        ),
        uq_enabled=settings.CHATBOT_REWRITE_UQ_ENABLED,
        uq_m_sampling=max(2, settings.CHATBOT_REWRITE_UQ_M_SAMPLING),
        uq_t_sampling=max(0.0, min(2.0, settings.CHATBOT_REWRITE_UQ_T_SAMPLING)),
        uq_tau_cluster=max(0.0, min(1.0, settings.CHATBOT_REWRITE_UQ_TAU_CLUSTER)),
        uq_tau_u=max(0.0, min(1.0, settings.CHATBOT_REWRITE_UQ_TAU_U)),
        uq_sample_max_tokens=max(256, settings.CHATBOT_REWRITE_UQ_SAMPLE_MAX_TOKENS),
    )
