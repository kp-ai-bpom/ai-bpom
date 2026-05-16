"""SemanticJudge: penilai SQL kandidat berbasis rubric 5 dimensi (deep_think)."""

from __future__ import annotations

import asyncio
import logging

from app.core.llm import LLMAdapter
from app.core.logger import log as core_log

from .generator_parsers import strip_thinking
from .config import SQLValidationConfig
from .parsers import parse_judge_verdict_lenient
from .prompts import build_judge_prompt, judge_system_message
from .types import RubricVerdict


log = core_log


class SemanticJudge:
    """LLM-as-a-Judge yang mengembalikan ``RubricVerdict``.

    Memakai model ``deep_think`` agar penilaian semantik lebih konsisten
    dibanding refiner. Ini implementasi asimetri model yang dijelaskan di
    Bab 2.9 (Scale partial-reward + Huyen judge).
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        config: SQLValidationConfig,
    ):
        self._llm_adapter = llm_adapter
        self._config = config

    async def judge(
        self,
        *,
        question: str,
        schema_context: str,
        sql: str,
    ) -> RubricVerdict:
        prompt = build_judge_prompt(
            question=question,
            schema_context=schema_context,
            sql=sql,
        )

        try:
            response = await self._llm_adapter.deep_think.bind(
                max_tokens=self._config.judge_max_tokens,
                temperature=self._config.judge_llm_temperature,
            ).ainvoke(
                [
                    {"role": "system", "content": judge_system_message()},
                    {"role": "user", "content": prompt},
                ]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("SemanticJudge: panggilan deep_think gagal: %s", exc)
            # Lenient parser dengan content kosong akan menghasilkan FAIL agregat
            # yang menjadi sinyal safety-net bagi orchestrator.
            return parse_judge_verdict_lenient("")

        content = strip_thinking(str(getattr(response, "content", "") or "")).strip()
        return parse_judge_verdict_lenient(content)
