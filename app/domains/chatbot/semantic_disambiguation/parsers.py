import json
import re

from .types import AMBIGUITY_TYPES, AmbiguityDetectionResult, InterpretationOption


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(content: str) -> dict | None:
    candidate = re.sub(r"^```(?:json)?\s*", "", content)
    candidate = re.sub(r"\s*```$", "", candidate)

    match = _JSON_OBJECT_PATTERN.search(candidate)
    if not match:
        return None

    raw_object = re.sub(r",(\s*[}\]])", r"\1", match.group(0))
    try:
        parsed = json.loads(raw_object)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_options(raw_options) -> list[InterpretationOption]:
    """Parse interpretation_options from LLM JSON.

    Accepts two shapes for forward/backward compatibility:
    1. New (preferred): list of objects ``{"label": "...", "description": "..."}``
    2. Legacy: list of plain strings ``["...", "..."]`` (description defaults
       to empty)
    """
    options: list[InterpretationOption] = []
    seen_labels: set[str] = set()

    if not isinstance(raw_options, list):
        return options

    for item in raw_options:
        if isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            description = str(item.get("description") or "").strip()
        else:
            label = str(item or "").strip()
            description = ""

        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        options.append(
            InterpretationOption(label=label, description=description)
        )

    return options


def parse_detection_output(content: str) -> AmbiguityDetectionResult | None:
    parsed = _extract_json_object(content)
    if parsed is None:
        return None

    is_ambiguous = bool(parsed.get("is_ambiguous"))
    ambiguity_type_raw = str(parsed.get("ambiguity_type") or "").strip().lower()
    ambiguity_type = ambiguity_type_raw if ambiguity_type_raw in AMBIGUITY_TYPES else None

    clarification_question = str(parsed.get("clarification_question") or "").strip() or None

    options = _parse_options(parsed.get("interpretation_options"))

    if not is_ambiguous:
        return AmbiguityDetectionResult(is_ambiguous=False)

    if ambiguity_type is None or clarification_question is None or not options:
        return None

    return AmbiguityDetectionResult(
        is_ambiguous=True,
        ambiguity_type=ambiguity_type,
        clarification_question=clarification_question,
        interpretation_options=options[:4],
    )


def parse_interpretation_sample(content: str) -> str:
    """Parse output sampler interpretasi → string ``pertanyaan_mandiri``.

    Mengembalikan string kosong bila JSON invalid atau field hilang. Caller
    bertanggung jawab handle empty string sebagai sample gagal (tidak
    di-embed, dihitung sebagai ``ERROR`` fingerprint di UQ).
    """
    parsed = _extract_json_object(content)
    if parsed is None:
        return ""
    text = str(parsed.get("pertanyaan_mandiri") or "").strip()
    return text


def parse_refined_question(content: str) -> str:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned
