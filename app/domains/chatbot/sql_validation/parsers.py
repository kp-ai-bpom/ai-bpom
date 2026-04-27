"""Parsers untuk output Refiner (SQL revisi) dan Semantic Judge (rubric JSON)."""

from __future__ import annotations

import json
import re

from ..sql_generator.parsers import (
    extract_json_object,
    extract_sql_from_fence,
    extract_sql_from_text,
)
from .types import (
    RUBRIC_DIMENSIONS,
    RubricDimensionVerdict,
    RubricVerdict,
    ValidationStatus,
)


_SQL_KEYWORD_PATTERN = re.compile(r"\b(select|with)\b", re.IGNORECASE)


def parse_refined_sql(content: str) -> tuple[str | None, str]:
    """Parse output refiner. Mengembalikan (sql_kandidat, explanation).

    Refiner diharapkan menjawab dalam JSON ``{"query": "...", "explanation": "..."}``;
    jika tidak, fallback ke fenced code block dan terakhir teks bebas yang mengandung
    SELECT/WITH (mengikuti pola di ``sql_generator.parsers``).
    """
    if not content:
        return None, ""

    cleaned = content.strip()

    # JSON path (preferred).
    parsed = extract_json_object(cleaned)
    if parsed is not None:
        sql_value = str(parsed.get("query") or parsed.get("sql") or "").strip()
        explanation = str(parsed.get("explanation") or "").strip()
        if sql_value and _SQL_KEYWORD_PATTERN.search(sql_value):
            return sql_value, explanation

    # Fenced ```sql ... ``` block.
    fenced = extract_sql_from_fence(cleaned)
    if fenced:
        return fenced, ""

    # Free-text starting with SELECT/WITH.
    free = extract_sql_from_text(cleaned)
    if free:
        return free, ""

    return None, ""


def _parse_label(value: object) -> ValidationStatus | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    # Sengaja batasi label rubric ke tiga nilai diskrit yang valid: PASS,
    # PARTIAL, FAIL. Status ``SKIPPED`` adalah mode internal orchestrator,
    # bukan label rubric — bila model menghasilkannya, anggap tidak dikenal
    # supaya jatuh ke jalur lenient (FAIL aman).
    allowed = {ValidationStatus.PASS, ValidationStatus.PARTIAL, ValidationStatus.FAIL}
    for member in allowed:
        if member.value == raw:
            return member
    # Toleran terhadap variasi yang sering dihasilkan model.
    if raw in {"OK", "GOOD", "CORRECT"}:
        return ValidationStatus.PASS
    if raw in {"PARTIAL_PASS", "PARTIALLY_CORRECT", "MIXED"}:
        return ValidationStatus.PARTIAL
    if raw in {"BAD", "WRONG", "INCORRECT"}:
        return ValidationStatus.FAIL
    return None


def _aggregate_overall(
    dimensions: dict[str, RubricDimensionVerdict],
    judge_overall: ValidationStatus | None,
) -> ValidationStatus:
    """Hitung label agregat dengan kebijakan max-severity dari per-dimensi.

    Aturan: ada satu FAIL → FAIL; tidak ada FAIL tetapi ada PARTIAL → PARTIAL;
    semua PASS → PASS. Bila judge sendiri memberi label lebih buruk dari
    agregat per-dimensi, hormati label judge (ambil yang paling parah). Ini
    mencegah judge memberi PASS overall padahal salah satu dimensi FAIL.
    """
    severity = {ValidationStatus.PASS: 0, ValidationStatus.PARTIAL: 1, ValidationStatus.FAIL: 2}
    worst = ValidationStatus.PASS
    for verdict in dimensions.values():
        if severity[verdict.label] > severity[worst]:
            worst = verdict.label
    if judge_overall and severity[judge_overall] > severity[worst]:
        worst = judge_overall
    return worst


def parse_judge_verdict(content: str) -> RubricVerdict | None:
    """Parse JSON output Semantic Judge menjadi ``RubricVerdict``.

    Skema yang diharapkan dari prompt::

        {
          "rubric": {
            "faithfulness": {"label": "PASS", "reason": "..."},
            "relevance":    {"label": "PASS", "reason": "..."},
            "select":       {"label": "PASS", "reason": "..."},
            "where":        {"label": "PARTIAL", "reason": "..."},
            "group_by":     {"label": "PASS", "reason": "..."}
          },
          "overall": "PARTIAL",
          "summary": "filter periode kemungkinan tidak sesuai"
        }

    Mengembalikan ``None`` bila JSON tidak valid atau tidak ada satu pun
    dimensi yang dapat diparsing.
    """
    if not content:
        return None

    parsed = extract_json_object(content)
    if parsed is None:
        return None

    raw_rubric = parsed.get("rubric")
    if not isinstance(raw_rubric, dict):
        return None

    # Spec wajibkan rubric 5 dimensi (Faithfulness, Relevance, SELECT, WHERE,
    # GROUP BY). Bila salah satu dimensi tidak hadir atau labelnya invalid,
    # tolak parse strict — biarkan ``parse_judge_verdict_lenient`` jatuh ke
    # FAIL aman. Ini menutup celah di mana judge mengeluarkan output rubric
    # parsial yang berpotensi diloloskan sebagai PASS/PARTIAL.
    dimensions: dict[str, RubricDimensionVerdict] = {}
    for dim_name in RUBRIC_DIMENSIONS:
        raw_entry = raw_rubric.get(dim_name)
        if not isinstance(raw_entry, dict):
            return None
        label = _parse_label(raw_entry.get("label"))
        if label is None:
            return None
        reason = str(raw_entry.get("reason") or "").strip()
        dimensions[dim_name] = RubricDimensionVerdict(label=label, reason=reason)

    if len(dimensions) != len(RUBRIC_DIMENSIONS):
        return None

    judge_overall = _parse_label(parsed.get("overall"))
    overall = _aggregate_overall(dimensions, judge_overall)
    summary = str(parsed.get("summary") or "").strip()

    return RubricVerdict(overall=overall, dimensions=dimensions, summary=summary)


def parse_judge_verdict_lenient(content: str) -> RubricVerdict:
    """Versi toleran dari ``parse_judge_verdict`` untuk fallback.

    Bila parser strict mengembalikan ``None`` *atau* melempar exception apa
    pun (mis. label tak terduga yang tidak ada di severity map), kita
    kembalikan FAIL agregat dengan summary berisi cuplikan output mentah
    supaya orchestrator bisa memutuskan safety-net tanpa crash.
    """
    try:
        parsed = parse_judge_verdict(content)
    except Exception:  # pragma: no cover - defensive fallback
        parsed = None
    if parsed is not None:
        return parsed
    snippet = (content or "").strip().replace("\n", " ")[:200]
    fail_dimensions = {
        name: RubricDimensionVerdict(
            label=ValidationStatus.FAIL,
            reason="Output judge tidak dapat diparsing",
        )
        for name in RUBRIC_DIMENSIONS
    }
    return RubricVerdict(
        overall=ValidationStatus.FAIL,
        dimensions=fail_dimensions,
        summary=f"Judge output unparseable: {snippet}",
    )


def is_trivial_select(sql: str) -> bool:
    """Heuristik sederhana untuk menentukan apakah query layak skip semantic loop.

    Trivial = SELECT tunggal tanpa JOIN, GROUP BY, HAVING, atau subquery.
    Kondisi WHERE diperbolehkan asal tidak memuat subquery.
    """
    if not sql:
        return False

    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    if not normalized.startswith("select"):
        return False

    forbidden_patterns = [
        r"\bjoin\b",
        r"\bgroup\s+by\b",
        r"\bhaving\b",
        r"\bunion\b",
        r"\bintersect\b",
        r"\bexcept\b",
        r"\bwith\s+\w+\s+as\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, normalized):
            return False

    # Subquery di klausa apa pun (heuristik kasar: parens yang berisi SELECT).
    if re.search(r"\(\s*select\b", normalized):
        return False

    return True
