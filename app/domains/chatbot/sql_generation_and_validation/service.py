"""SQLValidationService: orkestrator dua-loop (execution + semantic).

Implementasi Tahap 5 sesuai Bab 2.9:

* **Loop eksekusi** (max ``max_execution_iterations`` iterasi): jalankan dry-run
  dengan ``ExecutionValidator``. Bila gagal, panggil ``SQLRefiner`` (model
  ``think``) memakai trigger ``execution_error`` lalu coba lagi. Bila habis
  iterasi → label ``FAIL`` dan masuk safety-net.
* **Loop semantik** (max ``max_semantic_iterations`` iterasi, default 1):
  setelah execution PASS, panggil ``SemanticJudge`` (model ``deep_think``).
  Status ``PASS`` → selesai. Status ``PARTIAL`` → satu kesempatan revisi
  dengan trigger ``semantic_partial`` lalu re-execute, hasil akhir mengikuti
  judgment iterasi berikutnya. Status ``FAIL`` → safety-net.

Kebijakan label diskrit ini mengikuti Scale [23] partial-reward yang dibahas
di Bab 2.9. Heuristik ``is_trivial_select`` dapat melompati semantic loop
ketika query sangat sederhana (mengikuti ``trivial_skip``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from app.core.llm import LLMAdapter
from app.core.logger import log as core_log

from ..repositories import ChatbotRepository
from ..sql_generation_and_validation import SQLGenerator
from .config import SQLValidationConfig, get_sql_validation_config
from .execution_validator import ExecutionValidator
from .parsers import is_trivial_select
from .refiner import SQLRefiner
from .semantic_judge import SemanticJudge
from .types import (
    RubricVerdict,
    ValidationLevel,
    ValidationResult,
    ValidationRevision,
    ValidationStatus,
)


log = core_log


class SQLValidationService:
    """Fasad utama untuk pipeline validasi SQL.

    Konsumen tipikal: ``SemanticMemoryPipeline.run_from_prepared``. Service ini
    *tidak* menjalankan SQL final; ia hanya memutuskan apakah SQL aman
    dijalankan oleh pipeline. Eksekusi final tetap dilakukan caller agar lapisan
    eksekusi tetap ada di satu tempat.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: Optional[SQLValidationConfig] = None,
    ):
        self._config = config or get_sql_validation_config()
        self._execution_validator = ExecutionValidator(
            repository=repository,
            config=self._config,
        )
        self._refiner = SQLRefiner(
            llm_adapter=llm_adapter,
            config=self._config,
        )
        self._judge = SemanticJudge(
            llm_adapter=llm_adapter,
            config=self._config,
        )

    @property
    def config(self) -> SQLValidationConfig:
        return self._config

    async def validate(
        self,
        *,
        question: str,
        schema_context: str,
        candidate_sql: str,
        skip_validation: bool = False,
    ) -> ValidationResult:
        """Jalankan validasi terhadap ``candidate_sql``.

        Bila ``skip_validation`` True (mis. procedural hit) atau level NONE,
        kembalikan ``ValidationStatus.SKIPPED`` segera tanpa memanggil LLM.
        """
        if skip_validation or self._config.level is ValidationLevel.NONE:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                final_sql=candidate_sql,
                explanation="Validasi dilewati (skip_validation atau level=NONE).",
            )

        if not candidate_sql or not candidate_sql.strip():
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=candidate_sql or "",
                explanation="SQL kandidat kosong; safety-net diaktifkan.",
            )

        # Catat durasi keseluruhan pipeline validasi (tanpa data sensitif).
        # Detail per-iterasi di-log oleh modul anak (refiner, judge,
        # execution_validator).
        _t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._run_pipeline(
                    question=question,
                    schema_context=schema_context,
                    candidate_sql=candidate_sql,
                ),
                timeout=self._config.timeout_seconds,
            )
            log.info(
                "sql_validation: status=%s exec_iter=%d sem_iter=%d duration=%.3fs",
                result.status.value,
                result.execution_iterations,
                result.semantic_iterations,
                time.perf_counter() - _t0,
            )
            return result
        except asyncio.TimeoutError:
            log.warning(
                "SQLValidationService: melewati batas waktu %.1fs; safety-net diaktifkan.",
                self._config.timeout_seconds,
            )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=candidate_sql,
                explanation=(
                    f"Validasi melebihi batas waktu {self._config.timeout_seconds:.1f} detik; "
                    "SQL tidak dieksekusi sebagai langkah pengaman."
                ),
            )

    @staticmethod
    def _normalize_for_compare(sql: str) -> str:
        """Normalisasi whitespace + casing untuk deteksi revisi no-op.

        Refiner yang mengembalikan SQL identik (atau hanya berbeda spasi/case)
        tidak boleh dianggap sebagai revisi yang sah. Normalisasi ini cukup
        agresif untuk skenario tersebut tetapi tetap konservatif: kami tidak
        mengubah quoted identifier (PostgreSQL case-sensitive di dalam tanda
        kutip ganda).
        """
        return re.sub(r"\s+", " ", (sql or "").strip().lower())

    @staticmethod
    def _safety_guard(sql: str) -> str | None:
        """Read-only / multi-statement guard untuk SQL hasil refiner.

        Memakai validator statis dari ``SQLGenerator`` (dipakai juga pada SQL
        yang dihasilkan generator awal). Menolak SQL yang bukan SELECT/WITH,
        memuat kata kunci berbahaya (DML/DDL), atau memuat lebih dari satu
        statement. Mengembalikan ``None`` bila aman, atau pesan kesalahan.
        """
        return SQLGenerator.validate_sql_candidate(sql)

    async def _refine_with_guard(
        self,
        *,
        question: str,
        schema_context: str,
        broken_sql: str,
        feedback: str,
        trigger: str,
    ) -> tuple[str | None, str]:
        """Wrapper ``SQLRefiner.refine`` yang menerapkan safety guard.

        Bila SQL revisi melanggar guard read-only/multi-statement, kita
        memperlakukannya sebagai "refiner gagal" sehingga orchestrator
        memilih jalur safety-net. Ini menutup celah agar SQL berbahaya
        dari LLM tidak pernah lolos ke ``execute_sql``.
        """
        revised_sql, refine_explanation = await self._refiner.refine(
            question=question,
            schema_context=schema_context,
            broken_sql=broken_sql,
            feedback=feedback,
            trigger=trigger,
        )
        if not revised_sql:
            return None, ""

        # Apply normalisasi tabel/kolom yang sama dengan generator agar
        # halusinasi yang sering muncul saat refine (mis. `public.satker`,
        # `public.pegawai`, `public.SIAP_SATKER_TOP` tanpa quote) langsung
        # dirapikan sebelum eksekusi. Rewrite ini idempotent.
        try:
            from .generator import SQLGenerator as _SQLGen
            revised_sql = _SQLGen.normalize_known_aliases(revised_sql)
        except Exception:  # pragma: no cover - best effort
            log.debug("normalize_known_aliases gagal dipanggil pada refiner output", exc_info=True)

        # Tolak no-op revision: jika refiner mengembalikan SQL yang identik
        # (setelah normalisasi whitespace/case) dengan SQL sebelumnya, perlakukan
        # sebagai gagal merevisi agar status tidak naik ke PASS palsu.
        if self._normalize_for_compare(revised_sql) == self._normalize_for_compare(
            broken_sql
        ):
            log.info(
                "SQLValidationService: refiner mengembalikan SQL identik (no-op); "
                "diperlakukan sebagai gagal revisi (trigger=%s).",
                trigger,
            )
            return None, ""

        guard_error = self._safety_guard(revised_sql)
        if guard_error is not None:
            log.warning(
                "SQLValidationService: SQL refiner ditolak guard (%s): %s",
                guard_error,
                revised_sql[:200],
            )
            return None, ""

        return revised_sql, refine_explanation

    async def _run_pipeline(
        self,
        *,
        question: str,
        schema_context: str,
        candidate_sql: str,
    ) -> ValidationResult:
        revisions: list[ValidationRevision] = []
        current_sql = candidate_sql.strip()
        last_execution_error: str | None = None

        # --- Loop eksekusi ----------------------------------------------------
        execution_iterations = 0
        execution_passed = False

        for attempt in range(1, self._config.max_execution_iterations + 1):
            execution_iterations = attempt
            _t_exec = time.perf_counter()
            verdict = await self._execution_validator.validate(current_sql)
            _exec_dur = time.perf_counter() - _t_exec

            log.info(
                "sql_validation.exec iter=%d ok=%s duration=%.3fs sql_chars=%d",
                attempt,
                verdict.ok,
                _exec_dur,
                len(current_sql),
            )

            if verdict.ok:
                execution_passed = True
                last_execution_error = None
                break

            last_execution_error = verdict.error_message
            log.info(
                "sql_validation.exec iter=%d error=%s",
                attempt,
                (last_execution_error or "")[:200],
            )

            # Tidak ada anggaran iterasi tersisa untuk merevisi.
            if attempt >= self._config.max_execution_iterations:
                break

            revised_sql, _ = await self._refine_with_guard(
                question=question,
                schema_context=schema_context,
                broken_sql=current_sql,
                feedback=last_execution_error or "(tidak ada pesan error)",
                trigger="execution_error",
            )
            if not revised_sql:
                # Refiner tidak menghasilkan revisi yang valid (atau ditolak
                # safety guard) — hentikan loop.
                break

            revisions.append(
                ValidationRevision(
                    iteration=attempt,
                    trigger="execution_error",
                    feedback=last_execution_error or "",
                    sql_before=current_sql,
                    sql_after=revised_sql,
                )
            )
            current_sql = revised_sql

        if not execution_passed:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=current_sql,
                explanation=(
                    "SQL gagal dieksekusi setelah "
                    f"{execution_iterations} iterasi. "
                    f"Pesan terakhir: {last_execution_error or '(tidak diketahui)'}"
                ),
                execution_iterations=execution_iterations,
                revisions=revisions,
                last_execution_error=last_execution_error,
            )

        # --- Mode EXECUTION_ONLY: lewati semantic loop ------------------------
        if self._config.level is ValidationLevel.EXECUTION_ONLY:
            return ValidationResult(
                status=ValidationStatus.PASS,
                final_sql=current_sql,
                explanation="Execution-only mode: SQL berhasil dieksekusi pada dry-run.",
                execution_iterations=execution_iterations,
                revisions=revisions,
            )

        # --- Trivial-skip: SELECT sederhana lewat tanpa judge -----------------
        if self._config.trivial_skip and is_trivial_select(current_sql):
            return ValidationResult(
                status=ValidationStatus.PASS,
                final_sql=current_sql,
                explanation=(
                    "SQL berupa SELECT sederhana; semantic loop dilewati sesuai "
                    "kebijakan trivial_skip."
                ),
                execution_iterations=execution_iterations,
                revisions=revisions,
            )

        # --- Loop semantik (kebijakan PARTIAL fix sesuai Bab 2.9) -------------
        # Alur:
        #   1) Judge#1.
        #   2) PASS → terima.
        #   3) FAIL → safety-net (tanpa revisi).
        #   4) PARTIAL → revise (dengan guard) → re-execute → judge#2 wajib;
        #      hasil judge#2 = PASS → terima; PARTIAL/FAIL → safety-net.
        #
        # ``max_semantic_iterations`` kini berfungsi sebagai sakelar: nilai 0
        # menonaktifkan semantic loop sepenuhnya; nilai >= 1 mengaktifkan
        # kebijakan satu putaran revisi+re-judge di atas.
        if self._config.max_semantic_iterations <= 0:
            return ValidationResult(
                status=ValidationStatus.PASS,
                final_sql=current_sql,
                explanation="Semantic loop dinonaktifkan; menerima hasil execution PASS.",
                execution_iterations=execution_iterations,
                revisions=revisions,
            )

        # ----- Judge #1 -------------------------------------------------------
        semantic_iterations = 1
        _t_judge1 = time.perf_counter()
        rubric = await self._judge.judge(
            question=question,
            schema_context=schema_context,
            sql=current_sql,
        )
        log.info(
            "sql_validation.judge stage=judge1 overall=%s duration=%.3fs labels=%s",
            rubric.overall.value,
            time.perf_counter() - _t_judge1,
            self._format_rubric_labels(rubric),
        )

        if rubric.overall is ValidationStatus.PASS:
            return ValidationResult(
                status=ValidationStatus.PASS,
                final_sql=current_sql,
                explanation=(
                    rubric.summary or "Semua dimensi rubric PASS; SQL siap dieksekusi."
                ),
                rubric=rubric,
                execution_iterations=execution_iterations,
                semantic_iterations=semantic_iterations,
                revisions=revisions,
            )

        if rubric.overall is ValidationStatus.FAIL:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=current_sql,
                explanation=(
                    rubric.summary
                    or "Judge memberi label FAIL pada rubric agregat; safety-net diaktifkan."
                ),
                rubric=rubric,
                execution_iterations=execution_iterations,
                semantic_iterations=semantic_iterations,
                revisions=revisions,
            )

        # ----- PARTIAL: revise → re-execute → judge#2 -------------------------
        judge_feedback = self._compose_partial_feedback(rubric)
        revised_sql, _ = await self._refine_with_guard(
            question=question,
            schema_context=schema_context,
            broken_sql=current_sql,
            feedback=judge_feedback,
            trigger="semantic_partial",
        )
        if not revised_sql:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=current_sql,
                explanation=(
                    "Refiner tidak menghasilkan revisi setelah judge PARTIAL; "
                    "safety-net diaktifkan."
                ),
                rubric=rubric,
                execution_iterations=execution_iterations,
                semantic_iterations=semantic_iterations,
                revisions=revisions,
            )

        revisions.append(
            ValidationRevision(
                iteration=1,
                trigger="semantic_partial",
                feedback=judge_feedback,
                sql_before=current_sql,
                sql_after=revised_sql,
            )
        )

        _t_re_exec = time.perf_counter()
        re_verdict = await self._execution_validator.validate(revised_sql)
        execution_iterations += 1
        log.info(
            "sql_validation.exec iter=%d ok=%s duration=%.3fs sql_chars=%d stage=re_exec",
            execution_iterations,
            re_verdict.ok,
            time.perf_counter() - _t_re_exec,
            len(revised_sql),
        )
        if not re_verdict.ok:
            last_execution_error = re_verdict.error_message
            return ValidationResult(
                status=ValidationStatus.FAIL,
                final_sql=revised_sql,
                explanation=(
                    "Revisi semantik gagal saat re-execution: "
                    f"{last_execution_error or '(tidak diketahui)'}"
                ),
                rubric=rubric,
                execution_iterations=execution_iterations,
                semantic_iterations=semantic_iterations,
                revisions=revisions,
                last_execution_error=last_execution_error,
            )

        current_sql = revised_sql

        # Judge#2 wajib: konfirmasi apakah revisi benar-benar memperbaiki
        # masalah semantik atau hanya secara dangkal melewati execution.
        semantic_iterations = 2
        _t_judge2 = time.perf_counter()
        second_rubric = await self._judge.judge(
            question=question,
            schema_context=schema_context,
            sql=current_sql,
        )
        log.info(
            "sql_validation.judge stage=judge2 overall=%s duration=%.3fs labels=%s",
            second_rubric.overall.value,
            time.perf_counter() - _t_judge2,
            self._format_rubric_labels(second_rubric),
        )

        if second_rubric.overall is ValidationStatus.PASS:
            return ValidationResult(
                status=ValidationStatus.PASS,
                final_sql=current_sql,
                explanation=(
                    second_rubric.summary
                    or "Revisi atas label PARTIAL berhasil; semua dimensi rubric PASS."
                ),
                rubric=second_rubric,
                execution_iterations=execution_iterations,
                semantic_iterations=semantic_iterations,
                revisions=revisions,
            )

        # PARTIAL atau FAIL pada judge#2 → safety-net (kebijakan konservatif:
        # tidak dieksekusi). Status final dipertahankan apa adanya agar
        # kontrak API PASS/PARTIAL/FAIL tetap utuh; pipeline yang menentukan
        # bahwa baik PARTIAL maupun FAIL sama-sama tidak dieksekusi.
        return ValidationResult(
            status=second_rubric.overall,
            final_sql=current_sql,
            explanation=(
                second_rubric.summary
                or "Setelah satu revisi, judge masih menilai SQL belum tepat; "
                "safety-net diaktifkan."
            ),
            rubric=second_rubric,
            execution_iterations=execution_iterations,
            semantic_iterations=semantic_iterations,
            revisions=revisions,
        )

    @staticmethod
    def _format_rubric_labels(rubric: RubricVerdict) -> str:
        """Bentuk string ringkas label per-dimensi untuk log terstruktur.

        Tidak menyertakan alasan (yang bisa berisi snippet data); hanya label
        diskrit per dimensi agar aman dimasukkan ke log produksi.
        """
        return ",".join(
            f"{name}:{verdict.label.value}"
            for name, verdict in rubric.dimensions.items()
        )

    @staticmethod
    def _compose_partial_feedback(rubric: RubricVerdict) -> str:
        """Rangkum dimensi yang bermasalah agar refiner punya konteks revisi."""
        problem_lines: list[str] = []
        for name, verdict in rubric.dimensions.items():
            if verdict.label is ValidationStatus.PASS:
                continue
            label = verdict.label.value
            reason = verdict.reason or "(tanpa alasan)"
            problem_lines.append(f"- {name} [{label}]: {reason}")

        if not problem_lines:
            return rubric.summary or "Judge menilai SQL belum sepenuhnya tepat."

        header = "Judge memberi label PARTIAL pada rubric. Dimensi bermasalah:"
        summary_line = f"\nRingkasan judge: {rubric.summary}" if rubric.summary else ""
        return header + "\n" + "\n".join(problem_lines) + summary_line
