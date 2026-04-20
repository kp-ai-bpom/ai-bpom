import asyncio
import json
import re
from typing import Any, Dict, List

from app.core.logger import log

from ..core.config import _executor, settings as local_settings


def _parse_box_number(posisi: str) -> int | None:
    """Parse box number from posisi_nine_box_talenta string, e.g. 'Kotak 9 (...)' → 9."""
    if not posisi:
        return None
    match = re.search(r"Kotak\s+(\d+)", posisi)
    return int(match.group(1)) if match else None


def _load_candidates() -> List[Dict[str, Any]]:
    """Load candidates.json with caching."""
    import app.domains.pemetaan_suksesor.core.config as _c

    if _c._candidates_cache is None:
        try:
            with open(local_settings.CANDIDATES_JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            _c._candidates_cache = data
            log.info(f"📋 Candidates loaded — {len(data)} kandidat")
        except FileNotFoundError:
            log.warning("⚠️ candidates.json tidak ditemukan")
            _c._candidates_cache = []
        except json.JSONDecodeError:
            log.warning("⚠️ candidates.json format tidak valid")
            _c._candidates_cache = []

    assert _c._candidates_cache is not None
    return _c._candidates_cache


def _call_agent(agent, prompt: str):
    """Call a strands Agent synchronously and return the raw AgentResult.

    Returns the AgentResult object so callers can access both text output
    (via str(result)) and token usage metrics (via result.metrics.accumulated_usage).
    Returns None on error.
    """
    try:
        result = agent(prompt)
        return result
    except Exception as exc:
        log.exception(f"❌ Agent call failed: {exc}")
        return None


def _extract_json(text: str) -> Any:
    """Extract the first valid JSON object/array from LLM text output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    text = text.strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


async def _run_agent_async(agent, prompt: str):
    """Run a blocking Agent call in a thread-pool so the event loop stays free.

    Returns AgentResult (or None on error) — same as _call_agent.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _call_agent, agent, prompt)


def _safe_get(d: Any, *keys, default=None):
    """Safely traverse nested dicts/lists."""
    current = d
    for key in keys:
        try:
            current = current[key]
        except (KeyError, TypeError, IndexError):
            return default
    return current