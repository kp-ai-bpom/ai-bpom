import json
import re


_REWRITTEN_TAG_PATTERN = re.compile(r"\[rewritten\]\s*:\s*(.+)", re.IGNORECASE)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_PATTERN = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def strip_thinking(content: str) -> str:
    content = _THINK_BLOCK_PATTERN.sub("", content)
    content = _THINK_TAIL_PATTERN.sub("", content)
    return content.strip()


def _fix_json(json_candidate: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", json_candidate)


def extract_json_object(content: str) -> dict | None:
    candidate = re.sub(r"^```(?:json)?\s*", "", content)
    candidate = re.sub(r"\s*```$", "", candidate)

    match = _JSON_OBJECT_PATTERN.search(candidate)
    if not match:
        return None

    raw_object = _fix_json(match.group(0))
    try:
        parsed = json.loads(raw_object)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_rewritten(output: str) -> str:
    cleaned = strip_thinking(output)
    if not cleaned:
        return ""

    parsed = extract_json_object(cleaned)
    if parsed:
        rewritten = str(
            parsed.get("pertanyaan_mandiri")
            or parsed.get("rewritten_query")
            or parsed.get("rewritten")
            or ""
        ).strip()
        if rewritten:
            return rewritten

    for line in cleaned.splitlines():
        line = line.strip()
        match = _REWRITTEN_TAG_PATTERN.match(line)
        if match:
            return match.group(1).strip()

    return cleaned.splitlines()[-1].strip()
