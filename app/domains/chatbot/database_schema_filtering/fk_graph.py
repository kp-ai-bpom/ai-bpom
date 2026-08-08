"""FK (foreign-key) graph extractor for the schema retrieval stage.

Builds a hybrid FK map used to surface JOIN paths in the Stage 2 trace output
so that downstream Stage 3 (SQL generation) and the UI both have an
interpretable picture of how the retrieved tables relate to each other.

Two sources are merged:

1. **Documented edges** — parsed from the ``entity_text`` field of the
   base-knowledge CSV that backs the vector store. The canonical relation
   sentence follows the pattern
   ``"Relasi: kolom <col> pada tabel <A> mengacu pada <target_col> di tabel <B>"``.
   Edges from this source are tagged ``source="documented"`` because they
   come from the semantic description that is also embedded for retrieval.

2. **Inferred edges** — read from the live PostgreSQL
   ``information_schema.referential_constraints`` /
   ``key_column_usage`` views. These represent the structural truth of the
   schema (actual FK constraints) and are tagged ``source="inferred"``.

The documented map takes priority when both sources describe the same
``(table, column)`` origin — the entity_text wording is what gets embedded
for retrieval so it is the more faithful representation of what RASL "knows"
about the relation. The inferred map fills the gaps left by columns whose
entity_text does not (yet) include a relation sentence.

The merged map is exposed through :class:`FKGraph`, which is built once per
process and cached. Callers (the Stage 2 trace builder) ask for the subset
of edges whose both endpoints live in a given set of predicted-table keys —
that subset is the "join graph" for a single query.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


_BASE_KNOWLEDGE_CSV_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "base_knowledge_rasl.csv"
)


_RELATION_PATTERN = re.compile(
    r"kolom\s+(?P<src_col>\w+)\s+pada\s+(?:tabel|view)\s+(?P<src_tbl>[\w.]+)\s+"
    r"mengacu\s+pada\s+(?P<tgt_col>\w+)\s+(?:di|pada)\s+(?:tabel|view)\s+(?P<tgt_tbl>[\w.]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FKEdge:
    """A single foreign-key relation between two columns.

    ``source`` is ``"documented"`` when the edge was parsed from the
    ``entity_text`` description (the same text embedded for retrieval), and
    ``"inferred"`` when it was read from the live PostgreSQL FK constraints.
    The label is propagated all the way to the UI so the user can see
    whether the JOIN path comes from the curated knowledge base or only
    from structural metadata.
    """

    from_table: str  # "schema.table"
    from_column: str
    to_table: str  # "schema.table"
    to_column: str
    source: str  # "documented" | "inferred"

    def to_dict(self) -> dict[str, str]:
        return {
            "from": f"{self.from_table}.{self.from_column}",
            "to": f"{self.to_table}.{self.to_column}",
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "source": self.source,
        }


def _normalize_table_key(
    raw_table: str,
    allowed_tables: dict[str, list[str]] | None,
) -> str | None:
    """Resolve a raw table reference to a canonical ``schema.table`` key.

    The entity_text sentences are inconsistent about schema prefixes — most
    say ``"tabel pegawai_tm"`` (no schema) while a few say
    ``"public.pegawai_tm"``. We resolve by:

    1. If the raw value already contains a dot, keep it as-is (lowercased
       schema portion to match how the retriever builds keys).
    2. Otherwise, look up the bare table name across ``allowed_tables`` and
       attach the matching schema. If no allowed table matches, return None
       so the edge is dropped rather than emitted with a guessed prefix.
    """
    if not raw_table:
        return None
    # Strip trailing/leading dots and whitespace: the relation regex
    # captures with ``[\w.]+`` which greedily eats sentence-final periods
    # (``"di view V_PENDIDIKAN_TERAKHIR."`` → ``"V_PENDIDIKAN_TERAKHIR."``).
    # Without this strip the captured key has a phantom trailing dot that
    # never matches the canonical ``schema.table`` form the rest of the
    # pipeline uses, silently breaking FK closure and edges_within.
    cleaned = raw_table.strip().strip(".")
    if "." in cleaned:
        return cleaned

    if not allowed_tables:
        return None

    for schema_name, tables in allowed_tables.items():
        for table_name in tables:
            if table_name == cleaned or table_name.lower() == cleaned.lower():
                return f"{schema_name}.{table_name}"
    return None


def _parse_documented_edges(
    csv_path: Path,
    allowed_tables: dict[str, list[str]] | None,
) -> list[FKEdge]:
    """Parse ``Relasi: ...`` sentences from base-knowledge entity_text."""
    if not csv_path.exists():
        return []

    edges: list[FKEdge] = []
    seen: set[tuple[str, str, str, str]] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                column_name = (row.get("column_name") or "").strip()
                if not column_name:
                    continue
                entity_text = row.get("entity_text") or ""
                if not entity_text:
                    continue

                match = _RELATION_PATTERN.search(entity_text)
                if not match:
                    continue

                raw_src_tbl = (row.get("table_name") or match.group("src_tbl") or "").strip()
                src_tbl = _normalize_table_key(raw_src_tbl, allowed_tables)
                if not src_tbl:
                    continue
                tgt_tbl = _normalize_table_key(match.group("tgt_tbl"), allowed_tables)
                if not tgt_tbl:
                    continue

                src_col = match.group("src_col").strip()
                tgt_col = match.group("tgt_col").strip()
                if not src_col or not tgt_col:
                    continue
                if src_col.lower() != column_name.lower():
                    # Pattern matched a different column than the row's own
                    # column_name — skip rather than emit a misleading edge.
                    continue

                dedupe_key = (src_tbl, src_col, tgt_tbl, tgt_col)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                edges.append(
                    FKEdge(
                        from_table=src_tbl,
                        from_column=src_col,
                        to_table=tgt_tbl,
                        to_column=tgt_col,
                        source="documented",
                    )
                )
    except Exception:
        return []

    return edges


class FKGraph:
    """Merged FK map keyed by ``(table_key, column_name)``.

    Documented edges (from entity_text) take priority over inferred edges
    (from information_schema) for the same origin column, because the
    documented wording is what gets embedded and surfaced through RASL.
    Inferred edges fill the gaps for columns whose entity_text does not
    yet include a relation sentence.
    """

    def __init__(self, edges: Iterable[FKEdge]):
        self._by_origin: dict[tuple[str, str], FKEdge] = {}
        for edge in edges:
            key = (edge.from_table, edge.from_column)
            existing = self._by_origin.get(key)
            if existing is None:
                self._by_origin[key] = edge
                continue
            # "documented" wins over "inferred" for the same origin column.
            if existing.source != "documented" and edge.source == "documented":
                self._by_origin[key] = edge

    def role_for(self, table_key: str, column_name: str) -> str | None:
        """Return ``"FK→<target_table>.<target_col>"`` if the column is a
        documented or inferred FK origin, else ``None``."""
        edge = self._by_origin.get((table_key, column_name))
        if edge is None:
            return None
        return f"FK→{edge.to_table}.{edge.to_column}"

    def iter_edges(self) -> Iterable[FKEdge]:
        """Iterate over every merged FK edge in the graph (documented +
        inferred). Used by FK-closure expansion to discover neighbors of a
        partially-retrieved table set."""
        return iter(self._by_origin.values())

    def edges_within(self, predicted_table_keys: set[str]) -> list[FKEdge]:
        """Return the subset of edges whose both endpoints belong to the
        given set of predicted table keys — i.e. the JOIN graph for one
        query's retrieval result.

        Sorted deterministically (by from_table, from_column) so the UI
        gets a stable ordering across runs and the snapshot is diff-friendly.
        """
        if not predicted_table_keys:
            return []
        within = [
            edge
            for edge in self._by_origin.values()
            if edge.from_table in predicted_table_keys
            and edge.to_table in predicted_table_keys
        ]
        within.sort(key=lambda e: (e.from_table, e.from_column, e.to_table, e.to_column))
        return within

    def __len__(self) -> int:
        return len(self._by_origin)


@lru_cache(maxsize=1)
def _documented_edges_cached(
    allowed_tables_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[FKEdge, ...]:
    """Cache the entity_text parse — the CSV is read-only at runtime."""
    allowed_tables = {schema: list(tables) for schema, tables in allowed_tables_key}
    return tuple(_parse_documented_edges(_BASE_KNOWLEDGE_CSV_PATH, allowed_tables))


def _allowed_tables_to_cache_key(
    allowed_tables: dict[str, list[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not allowed_tables:
        return tuple()
    return tuple(
        (schema, tuple(sorted(tables))) for schema, tables in sorted(allowed_tables.items())
    )


def build_fk_graph(
    *,
    allowed_tables: dict[str, list[str]] | None,
    inferred_edges: Iterable[dict[str, Any]] | None = None,
) -> FKGraph:
    """Build the merged FK graph for the current allowed-tables config.

    ``inferred_edges`` is the raw output of
    :py:meth:`ChatbotRepository.get_foreign_key_edges` (a list of dicts).
    It is passed in rather than fetched here so the repository (which owns
    the DB session) stays the only async I/O point and this module can
    remain pure / synchronous.
    """
    documented = list(
        _documented_edges_cached(_allowed_tables_to_cache_key(allowed_tables))
    )
    inferred: list[FKEdge] = []
    for raw in inferred_edges or []:
        src_tbl = raw.get("from_table") or raw.get("from") or ""
        src_col = raw.get("from_column") or ""
        tgt_tbl = raw.get("to_table") or raw.get("to") or ""
        tgt_col = raw.get("to_column") or ""
        if not src_tbl or not src_col or not tgt_tbl or not tgt_col:
            continue
        inferred.append(
            FKEdge(
                from_table=str(src_tbl),
                from_column=str(src_col),
                to_table=str(tgt_tbl),
                to_column=str(tgt_col),
                source="inferred",
            )
        )

    # Documented first so they win the origin-column dedupe in FKGraph.
    return FKGraph(documented + inferred)


__all__ = ["FKEdge", "FKGraph", "build_fk_graph"]
