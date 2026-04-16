import json
import re


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_PATTERN = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_SQL_FENCE_PATTERN = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_TEXT_PATTERN = re.compile(r"\b(select|with)\b.*", re.IGNORECASE | re.DOTALL)


def strip_thinking(content: str) -> str:
    content = _THINK_BLOCK_PATTERN.sub("", content)
    content = _THINK_TAIL_PATTERN.sub("", content)
    return content.strip()


def _fix_json(json_candidate: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", json_candidate)


def extract_json_object(content: str) -> dict[str, str] | None:
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


def extract_sql_from_fence(content: str) -> str | None:
    match = _SQL_FENCE_PATTERN.search(content)
    if not match:
        return None

    sql = match.group(1).strip()
    return sql or None


def extract_sql_from_text(content: str) -> str | None:
    match = _SQL_TEXT_PATTERN.search(content)
    if not match:
        return None

    sql = match.group(0).replace("```", "").strip()
    if not re.match(r"^(select|with)\b", sql, flags=re.IGNORECASE):
        return None
    return sql
