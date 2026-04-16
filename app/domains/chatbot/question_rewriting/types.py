from dataclasses import dataclass


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
