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
    # Subset of FK edges whose both endpoints are in ``predicted_tables`` —
    # this is the join graph surfaced through the Stage 2 trace output and
    # the UI so users can see how retrieved tables relate. Each dict has
    # the shape produced by :py:meth:`FKEdge.to_dict`. Empty list when no
    # FK relations apply to the retrieved tables.
    join_graph: list[dict[str, str]] = field(default_factory=list)
    # PK / FK / display key map ``{table_key: {column_name: role}}`` for
    # the retrieved tables, used by the Stage 2 trace builder to annotate
    # each column with its semantic role. Role values:
    # - ``"PK"`` for primary-key columns
    # - ``"FK→<target_table>.<target_col>"`` for foreign keys
    # - ``"display"`` for whitelisted display columns auto-injected by the
    #   retriever (Task #50 / #58 lineage)
    column_roles: dict[str, dict[str, str]] = field(default_factory=dict)
