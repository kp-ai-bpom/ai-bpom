from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedTable:
    schema: str
    table: str
    score: float
    column_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    keywords: list[str]
    predicted_tables: dict[str, RetrievedTable]
    context: str
    sql: str
    explanation: str
    executed: bool
    execution_error: str | None = None
    rows: list[dict[str, Any]] | None = None
