from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodicMatch:
    episode_id: int
    session_id: str
    similarity: float
    summary: str


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    episodic_matches_count: int
    top_similarity: float
    episodic_details: list[EpisodicMatch]
    # Sinyal UQ Stage 1 (M-sampling rewriter NL→NL). None bila UQ
    # dimatikan via config atau bila tidak ada konteks untuk rewrite
    # (turn pertama). Bila ada, dict berisi: h_norm, tau_u, verdict,
    # m_total, m_valid, unique_outcomes, majority_ratio,
    # mean_intra_cosine, n_error, fingerprints, samples,
    # majority_cluster_representative.
    uncertainty: dict[str, Any] | None = field(default=None)
