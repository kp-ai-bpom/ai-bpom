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
    # Field jejak validasi SQL (Tahap 5). Sengaja dijaga optional dan default
    # None agar konsumen lama tidak terpengaruh sebelum lapisan API/UI siap.
    validation_status: str | None = None
    validation_iterations: dict[str, int] | None = None
    validation_rubric: dict[str, dict[str, str]] | None = None
    validation_revisions: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class PreparedSchemaContext:
    keywords: list[str]
    predicted_tables: dict[str, RetrievedTable]
    context: str
    schema_tables: list[dict[str, Any]]
