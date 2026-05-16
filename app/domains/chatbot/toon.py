import json
from typing import Any, Mapping, Sequence

from toon_format import encode as encode_toon

from app.domains.chatbot.core.config import chatbot_settings as settings

TOON_NA = "N/A"


def is_toon_enabled() -> bool:
    """Single global switch for TOON encoding across the entire chatbot
    pipeline (ambiguity, question_rewriting, semantic_memory, sql_generator).
    Controlled by the env var ``CHATBOT_USE_TOON`` (default ``true``).
    When False, all encoders fall back to compact JSON so the impact of TOON
    vs JSON formatting can be measured end-to-end.
    """
    return bool(getattr(settings, "CHATBOT_USE_TOON", True))


def current_format_label() -> str:
    """Human-readable label of the currently active encoding for use in
    prompt headers (e.g. ``[context (format=TOON)]``).
    """
    return "TOON" if is_toon_enabled() else "JSON"


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalize_cell(value: Any, list_delimiter: str) -> str:
    if isinstance(value, Mapping):
        pairs: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            pairs.append(f"{_normalize_scalar(key)}={_normalize_scalar(item)}")
        return list_delimiter.join(pairs)

    if isinstance(value, set):
        iterable = sorted(value, key=lambda item: str(item))
        return list_delimiter.join(_normalize_scalar(item) for item in iterable)

    if isinstance(value, (list, tuple)):
        return list_delimiter.join(_normalize_scalar(item) for item in value)

    return _normalize_scalar(value)


def _normalize_row(
    row: Mapping[str, Any],
    normalized_fields: Sequence[str],
    list_delimiter: str,
) -> dict[str, str]:
    normalized_row: dict[str, str] = {}
    for field in normalized_fields:
        normalized_row[field] = _normalize_cell(
            row.get(field),
            list_delimiter=list_delimiter,
        )
    return normalized_row


def _encode_json_table(
    name: str,
    normalized_rows: Sequence[Mapping[str, Any]],
) -> str:
    """JSON fallback when TOON is disabled. Uses a compact structure that
    mirrors the TOON shape so prompts remain readable: a single object with
    the table name as key and the list of rows as value.
    """
    payload = {name: list(normalized_rows)}
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def encode_table(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    list_delimiter: str = "|",
) -> str:
    if not name.strip() or not rows:
        return TOON_NA

    normalized_fields = [str(field).strip() for field in fields if str(field).strip()]
    if not normalized_fields:
        return TOON_NA

    normalized_rows = [
        _normalize_row(
            row=row,
            normalized_fields=normalized_fields,
            list_delimiter=list_delimiter,
        )
        for row in rows
    ]

    if not is_toon_enabled():
        encoded = _encode_json_table(name=name, normalized_rows=normalized_rows)
        return encoded if encoded else TOON_NA

    encoded = encode_toon(
        {name: normalized_rows},
        {"delimiter": ","},
    ).strip()
    return encoded if encoded else TOON_NA


def encode_or_na(value: str | None) -> str:
    if value is None:
        return TOON_NA

    trimmed = value.strip()
    if not trimmed:
        return TOON_NA

    return trimmed


def encode_table_with_max_chars(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    max_chars: int,
    list_delimiter: str = "|",
    trim_from_start: bool = False,
) -> str:
    if max_chars <= 0:
        return TOON_NA

    candidate_rows = list(rows)
    while candidate_rows:
        encoded = encode_table(
            name=name,
            rows=candidate_rows,
            fields=fields,
            list_delimiter=list_delimiter,
        )
        if encoded != TOON_NA and len(encoded) <= max_chars:
            return encoded

        if trim_from_start:
            candidate_rows = candidate_rows[1:]
        else:
            candidate_rows = candidate_rows[:-1]

    return TOON_NA
