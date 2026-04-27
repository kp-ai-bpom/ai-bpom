"""SQLRefiner: revisi SQL satu kali per panggilan menggunakan model ``think``."""

from __future__ import annotations

import asyncio
import logging

from app.core.llm import LLMAdapter
from app.core.logger import log as core_log

from ..sql_generator.parsers import strip_thinking
from .config import SQLValidationConfig
from .parsers import parse_refined_sql
from .prompts import build_refiner_prompt, refiner_system_message


log = core_log


class SQLRefiner:
    """Single-LLM refiner.

    Mengikuti rancangan asimetri model: refiner memakai model ``think`` (lebih
    murah dan cepat) sedangkan semantic judge memakai ``deep_think``.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        config: SQLValidationConfig,
    ):
        self._llm_adapter = llm_adapter
        self._config = config

    async def refine(
        self,
        *,
        question: str,
        schema_context: str,
        broken_sql: str,
        feedback: str,
        trigger: str,
    ) -> tuple[str | None, str]:
        """Mengembalikan (sql_revisi, explanation_singkat).

        Bila revisi tidak berhasil dihasilkan (LLM error / output tidak dapat
        diparsing), mengembalikan ``(None, "")`` agar orchestrator memutuskan
        langkah berikutnya (lanjut iterasi atau berhenti).
        """
        prompt = build_refiner_prompt(
            question=question,
            schema_context=schema_context,
            broken_sql=broken_sql,
            feedback=feedback,
            trigger=trigger,
        )

        try:
            response = await self._llm_adapter.think.bind(
                max_tokens=self._config.refiner_max_tokens
            ).ainvoke(
                [
                    {"role": "system", "content": refiner_system_message()},
                    {"role": "user", "content": prompt},
                ]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("SQLRefiner: panggilan think gagal: %s", exc)
            return None, ""

        content = strip_thinking(str(getattr(response, "content", "") or "")).strip()
        if not content:
            log.warning("SQLRefiner: think mengembalikan respons kosong")
            return None, ""

        sql_candidate, explanation = parse_refined_sql(content)
        if not sql_candidate:
            log.warning("SQLRefiner: gagal mem-parsing SQL dari output revisi")
            return None, ""

        return sql_candidate, explanation
