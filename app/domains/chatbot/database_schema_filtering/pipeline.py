import csv
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.llm import LLMAdapter

from ..repositories import ChatbotRepository
from ..sql_generation_and_validation import SQLGenerator, get_sql_generator_config
from ..sql_generation_and_validation import (
    SQLValidationService,
    ValidationResult,
    ValidationStatus,
    get_sql_validation_config,
)
from ..sql_generation_and_validation.types import RubricDimensionVerdict, ValidationRevision
from .config import SemanticMemoryConfig
from .context_builder import ContextBuilder
from .keyword_extractor import KeywordExtractor
from .table_retriever import TableRetriever
from .types import PipelineResult, PreparedSchemaContext, RetrievedTable


_COLUMN_ALIAS_STOPWORDS = {
    "atau",
    "dan",
    "yang",
    "dengan",
    "untuk",
    "pada",
    "di",
    "ke",
    "dari",
}

_BASE_KNOWLEDGE_CSV_PATH = Path(__file__).resolve().parents[4] / "data" / "base_knowledge.csv"


def _normalize_alias_key(identifier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", identifier.strip().lower())


def _register_column_alias(
    aliases_by_table: dict[str, dict[str, set[str]]],
    table_key: str,
    alias: str,
    column_name: str,
) -> None:
    normalized_alias = _normalize_alias_key(alias)
    if not normalized_alias:
        return

    aliases_by_table.setdefault(table_key, {}).setdefault(normalized_alias, set()).add(
        column_name
    )


def _register_domain_aliases(
    aliases_by_table: dict[str, dict[str, set[str]]],
) -> None:
    # Known education aliases frequently used in user prompts.
    education_table = "siap.v_pendidikan_terakhir"
    for alias in {
        "nama_pt",
        "namapt",
        "perguruan_tinggi",
        "kampus",
        "universitas",
        "universitas_atau_sekolah",
        "almamater",
    }:
        _register_column_alias(aliases_by_table, education_table, alias, "namasekolah")

    for alias in {
        "nama_prodi",
        "namaprodi",
        "prodi",
        "jurusan",
        "program_studi",
    }:
        _register_column_alias(aliases_by_table, education_table, alias, "programstudi")

    # Common hallucinated person-name fields for pegawai data.
    pegawai_table = "public.pegawai_tm"
    for alias in {
        "first_name",
        "firstname",
        "last_name",
        "lastname",
        "full_name",
        "nama_lengkap",
    }:
        _register_column_alias(aliases_by_table, pegawai_table, alias, "nama")


@lru_cache(maxsize=1)
def _load_base_knowledge_column_aliases() -> dict[str, dict[str, set[str]]]:
    aliases_by_table: dict[str, dict[str, set[str]]] = {}

    if not _BASE_KNOWLEDGE_CSV_PATH.exists():
        _register_domain_aliases(aliases_by_table)
        return aliases_by_table

    try:
        with _BASE_KNOWLEDGE_CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entity_type = str(row.get("entity_type") or "").strip().lower()
                if entity_type != "column":
                    continue

                schema_name = str(row.get("schema_name") or "").strip()
                table_name = str(row.get("table_name") or "").strip()
                column_name = str(row.get("column_name") or "").strip()
                if not schema_name or not table_name or not column_name:
                    continue

                table_key = f"{schema_name.lower()}.{table_name.lower()}"

                # Always register the physical column name itself.
                _register_column_alias(
                    aliases_by_table,
                    table_key,
                    alias=column_name,
                    column_name=column_name,
                )

                alias_field = str(row.get("column_alias") or "")
                if alias_field:
                    for alias in re.split(r"[,;/|]", alias_field):
                        cleaned_alias = alias.strip()
                        if not cleaned_alias:
                            continue
                        _register_column_alias(
                            aliases_by_table,
                            table_key,
                            alias=cleaned_alias,
                            column_name=column_name,
                        )
    except Exception:
        # Keep pipeline resilient if CSV cannot be parsed.
        aliases_by_table = {}

    _register_domain_aliases(aliases_by_table)
    return aliases_by_table


_COLUMN_ALIAS_CANONICAL_MAP = {
    # Frequently produced aliases by LLM for education attributes.
    "namaprodi": "programstudi",
    "prodi": "programstudi",
    "programstudi": "programstudi",
    "namapt": "namasekolah",
    "perguruantinggi": "namasekolah",
    "universitasatausekolah": "namasekolah",
    "namasekolah": "namasekolah",
    # Common hallucinated person-name columns for pegawai table.
    "firstname": "nama",
    "lastname": "nama",
}

_COLUMN_TOKEN_EXPANSIONS = {
    "prodi": {"programstudi"},
    "namaprodi": {"programstudi"},
    "program": {"programstudi"},
    "studi": {"programstudi"},
    "universitas": {"namasekolah"},
    "kampus": {"namasekolah"},
    "sekolah": {"namasekolah"},
    "perguruan": {"namasekolah"},
    "tinggi": {"namasekolah"},
    "almamater": {"namasekolah"},
    "jurusan": {"programstudi"},
}


class SemanticMemoryPipeline:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: SemanticMemoryConfig,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._config = config
        self._keyword_extractor = KeywordExtractor(
            llm_adapter=llm_adapter,
            allowed_tables=config.allowed_tables,
            retries=config.keyword_retries,
            llm_temperature=config.llm_temperature,
        )
        self._table_retriever = TableRetriever(
            llm_adapter=llm_adapter,
            repository=repository,
            config=config,
        )
        self._context_builder = ContextBuilder(
            column_similarity_threshold=config.column_similarity_threshold,
            max_context_chars=config.max_context_chars,
        )
        sql_generator_config = get_sql_generator_config()
        self._sql_generator = SQLGenerator(
            llm_adapter=llm_adapter,
            allowed_tables=config.allowed_tables,
            config=sql_generator_config,
        )
        self._sql_validation_config = get_sql_validation_config()
        self._sql_validation_service = SQLValidationService(
            llm_adapter=llm_adapter,
            repository=repository,
            config=self._sql_validation_config,
        )

    async def prepare_context(self, query: str) -> PreparedSchemaContext:
        schema_tables = await self._repository.load_schema(self._config.allowed_tables)
        if not schema_tables:
            raise RuntimeError("No schema loaded from database")

        keywords = await self._keyword_extractor.extract(query)
        predicted_tables = await self._table_retriever.retrieve(
            keywords=keywords,
            raw_query=query,
        )
        if not predicted_tables:
            predicted_tables = self._fallback_tables(
                keywords=keywords,
                schema_tables=schema_tables,
            )

        relevant_schema_tables = [
            table
            for table in schema_tables
            if f"{table['schema']}.{table['name']}" in predicted_tables
        ]

        table_descriptions: dict[str, str] = {}
        try:
            if await self._repository.is_vector_table_available(self._config.vector_table):
                table_descriptions = await self._repository.load_table_descriptions(
                    self._config.vector_table
                )
        except Exception:
            table_descriptions = {}

        samples = await self._repository.load_column_samples(
            schema_tables=relevant_schema_tables,
            n_samples=self._config.sample_rows_per_table,
        )

        context = self._context_builder.build(
            predicted_tables=predicted_tables,
            schema_tables=schema_tables,
            samples=samples,
            table_descriptions=table_descriptions,
        )

        return PreparedSchemaContext(
            keywords=keywords,
            predicted_tables=predicted_tables,
            context=context,
            schema_tables=schema_tables,
        )

    async def run_from_prepared(
        self,
        query: str,
        prepared: PreparedSchemaContext,
        skip_validation: bool = False,
        ablate_active_filter: bool = False,
    ) -> PipelineResult:
        """Hasilkan SQL dari konteks yang sudah disiapkan.

        ``ablate_active_filter`` (eval-only): bila True, semua jaring
        deterministik untuk filter "pegawai aktif" (Stage 5b + pre-inject +
        safety-inject di akhir) dilewati. Hanya untuk benchmark ablation.
        """
        context = prepared.context
        schema_tables = prepared.schema_tables

        sql, explanation = await self._sql_generator.generate(query, context)
        if not sql:
            raise RuntimeError(explanation or "Failed to generate SQL")

        corrected_sql = self._autocorrect_sql_identifiers(sql, schema_tables)
        if corrected_sql != sql:
            sql = corrected_sql

        validation_error = self._sql_generator.validate_sql_candidate(sql)
        if validation_error:
            raise RuntimeError(f"Generated SQL is invalid: {validation_error}")

        # PRE-INJECT: pasang filter default pegawai aktif sebelum SQL
        # dikirim ke validation pipeline. Tanpa ini, judge menilai SQL
        # versi pra-inject dan memberi label ``where=FAIL`` walaupun
        # boundary inject di akhir pipeline akan menambahkan filter.
        # Akibatnya ``validation_status`` ter-surface ke client sebagai
        # ``FAIL`` padahal SQL final yang dieksekusi sudah benar. Pre-inject
        # di sini menjamin judge dan client melihat SQL yang konsisten.
        if not ablate_active_filter:
            sql = self._apply_pegawai_filter_safety_inject(
                sql=sql,
                user_query=query,
            )

        # Tahap 5: SQL Validation Pipeline (refiner + judge). Service ini
        # mengembalikan ``ValidationResult`` dengan label diskrit dan, bila
        # perlu, SQL revisi. Eksekusi final tetap di pipeline ini.
        validation_result = await self._sql_validation_service.validate(
            question=query,
            schema_context=context,
            candidate_sql=sql,
            skip_validation=skip_validation,
        )

        # Tahap 5b: post-processing deterministik untuk FAIL pada dimensi
        # ``where`` yang murni soal "filter default pegawai aktif tidak
        # dipasang". Lihat ``_apply_default_active_filter_repair`` untuk
        # rasional lengkap (termasuk false-positive judge pada tabel
        # ``mantel.period_employees`` yang tidak punya kolom status_pegawai/
        # kedudukan_pegawai). Bila kondisi tidak terpenuhi, hasil validasi
        # dikembalikan apa adanya tanpa modifikasi.
        if not ablate_active_filter:
            validation_result = self._apply_default_active_filter_repair(
                validation_result=validation_result,
                user_query=query,
            )

        validation_payload = self._build_validation_payload(validation_result)

        # Belt-and-suspenders fail-safe di boundary pipeline.
        #
        # Walau ``SQLGenerator.generate`` sudah memiliki fail-safe
        # ``_inject_default_pegawai_filters`` di akhir loop retry, kita
        # ulangi mekanisme yang sama di sini sebagai jaring pengaman
        # terakhir. Alasannya:
        #
        # 1. Refiner (SQL Validation Pipeline / Tahap 5) bisa menghasilkan
        #    SQL revisi yang justru menghapus filter default pegawai aktif
        #    karena fokus refiner adalah memperbaiki dimensi rubric yang
        #    bermasalah dan tidak selalu mempertahankan klausa WHERE.
        # 2. Bila pipeline mengembalikan SQL pada jalur safety-net (FAIL/
        #    PARTIAL), kita tetap ingin SQL yang ditampilkan ke
        #    pengguna/operator memenuhi kontrak filter default sehingga
        #    audit trail dan transparansi konsisten dengan kebijakan
        #    bahasa-Indonesia "pegawai aktif by default".
        # 3. Kontrak inject bersifat idempotent: kalau filter sudah ada
        #    SQL tidak diubah; kalau user eksplisit minta non-aktif kita
        #    skip inject.
        candidate_final_sql = validation_result.final_sql or sql
        if not ablate_active_filter:
            candidate_final_sql = self._apply_pegawai_filter_safety_inject(
                sql=candidate_final_sql,
                user_query=query,
            )

        # Bila validation FAIL atau PARTIAL → safety-net: jangan eksekusi SQL.
        # `is_safe_to_execute` hanya True untuk PASS/SKIPPED, sehingga PARTIAL
        # juga otomatis ikut safety-net. Status final tetap dipertahankan apa
        # adanya pada metadata API agar pengguna/operator tahu alasan persis
        # SQL tidak dijalankan.
        if not validation_result.is_safe_to_execute:
            return PipelineResult(
                keywords=prepared.keywords,
                predicted_tables=prepared.predicted_tables,
                context=context,
                sql=candidate_final_sql,
                explanation=(
                    f"{explanation} | Validasi SQL "
                    f"{validation_result.status.value.lower()}: "
                    f"{validation_result.explanation}"
                ),
                executed=False,
                execution_error=validation_result.last_execution_error,
                rows=None,
                **validation_payload,
            )

        # PASS atau SKIPPED: eksekusi SQL final (yang mungkin sudah direvisi).
        final_sql = candidate_final_sql

        # Belt-and-suspenders: guard read-only/multi-statement diterapkan ulang
        # sebelum final execute. Validation service sudah memanggil guard pada
        # setiap revisi; pengulangan di sini memastikan SQL skenario SKIPPED
        # (procedural hit) maupun PASS dari path manapun tetap melewati guard
        # sebelum mencapai database.
        final_guard_error = self._sql_generator.validate_sql_candidate(final_sql)
        if final_guard_error:
            return PipelineResult(
                keywords=prepared.keywords,
                predicted_tables=prepared.predicted_tables,
                context=context,
                sql=final_sql,
                explanation=(
                    f"{explanation} | Validasi SQL gagal: SQL final ditolak "
                    f"safety guard ({final_guard_error})."
                ),
                executed=False,
                execution_error=final_guard_error,
                rows=None,
                **validation_payload,
            )

        rows, execution_error = await self._repository.execute_sql(
            sql=final_sql,
            timeout_ms=self._config.sql_timeout_ms,
        )
        executed = execution_error is None
        if execution_error:
            explanation = (
                f"{explanation} | SQL execution failed: {execution_error[:180]}"
            )

        return PipelineResult(
            keywords=prepared.keywords,
            predicted_tables=prepared.predicted_tables,
            context=context,
            sql=final_sql,
            explanation=explanation,
            executed=executed,
            execution_error=execution_error,
            rows=rows,
            **validation_payload,
        )

    async def run(self, query: str) -> PipelineResult:
        prepared = await self.prepare_context(query)
        return await self.run_from_prepared(query=query, prepared=prepared)

    def _apply_pegawai_filter_safety_inject(
        self,
        *,
        sql: str,
        user_query: str,
    ) -> str:
        """Pasang filter default pegawai aktif pada SQL akhir bila belum ada.

        Idempotent: kalau SQL tidak menyentuh ``public.pegawai_tm`` atau
        pengguna eksplisit minta data non-aktif, SQL dikembalikan apa adanya.
        Kalau filter sudah lengkap, juga dikembalikan apa adanya. Kalau salah
        satu/keduanya hilang, kita tambahkan secara mekanis menggunakan
        helper di ``SQLGenerator`` yang menjadi sumber kebenaran.
        """
        if not sql:
            return sql

        try:
            generator = self._sql_generator
            if not generator._references_pegawai_table(sql):
                return sql
            if generator._user_requests_non_active(user_query):
                return sql
            return generator._inject_default_pegawai_filters(sql)
        except Exception:
            # Fail-safe: jangan pernah memutus pipeline karena bug di inject;
            # kembalikan SQL apa adanya supaya alur utama tidak terganggu.
            return sql

    @staticmethod
    def _references_period_employees(sql: str) -> bool:
        """Cek apakah SQL menyentuh tabel ``mantel.period_employees``.

        Tabel ini adalah snapshot per-periode pegawai aktif (sudah pre-filter
        oleh ETL upstream) dan TIDAK memiliki kolom ``status_pegawai`` /
        ``kedudukan_pegawai``. Aturan default active filter pada rubric judge
        tidak applicable di sini, sehingga FAIL pada dimensi ``where`` dengan
        alasan "filter default pegawai aktif" adalah false-positive yang
        dapat di-override secara deterministik.
        """
        if not sql:
            return False
        return bool(
            re.search(
                r"\b(?:mantel\.)?\"?period_employees\"?\b",
                sql,
                re.IGNORECASE,
            )
        )

    def _apply_default_active_filter_repair(
        self,
        *,
        validation_result: ValidationResult,
        user_query: str,
    ) -> ValidationResult:
        """Post-processing deterministik untuk FAIL pada dimensi ``where``.

        Judge LLM kadang memberi label ``where=FAIL`` dengan alasan "filter
        default pegawai aktif tidak dipasang", padahal:

        (a) Untuk SQL yang menyentuh ``public.pegawai_tm``, filter dapat
            di-injeksi secara mekanis (idempotent) tanpa memanggil LLM lagi.
            Bila injeksi sudah no-op (mis. filter sebenarnya sudah ada di
            subquery/CTE tetapi judge tidak melihatnya), verdict FAIL juga
            dapat di-override karena kontrak fisik sudah dipenuhi.

        (b) Untuk SQL yang HANYA menyentuh ``mantel.period_employees`` (tidak
            menyentuh ``public.pegawai_tm`` sama sekali), aturan default
            active filter tidak relevan: tabel ini adalah snapshot
            per-periode pegawai aktif dan tidak memiliki kolom
            ``status_pegawai`` / ``kedudukan_pegawai``. Verdict FAIL adalah
            false-positive judge yang dapat di-override deterministik.

        Pada kedua kasus, kita upgrade rubric (where → PASS, overall → PASS)
        selama dimensi rubric lain semuanya PASS, lalu tambahkan satu entri
        ``ValidationRevision`` sebagai jejak audit (trigger dimulai dengan
        ``deterministic_active_filter_*``). Bila kondisi tidak terpenuhi,
        ``ValidationResult`` dikembalikan apa adanya.

        Catatan: helper ini sengaja tidak menggunakan LLM apa pun supaya
        latensi pipeline tidak terdampak (Bab 3.x: deterministic safety net
        untuk false-positive judge).
        """
        # 1) Hanya trigger pada FAIL — PARTIAL/PASS/SKIPPED tidak relevan.
        if validation_result.status is not ValidationStatus.FAIL:
            return validation_result

        rubric = validation_result.rubric
        if rubric is None:
            return validation_result

        where_verdict = rubric.dimensions.get("where")
        if where_verdict is None or where_verdict.label is not ValidationStatus.FAIL:
            return validation_result

        # 2) Semua dimensi non-where harus PASS supaya repair aman dilakukan.
        #    Bila ada dimensi lain yang FAIL/PARTIAL, masalah lebih luas dari
        #    sekadar default active filter dan harus tetap masuk safety-net.
        for name, dim in rubric.dimensions.items():
            if name == "where":
                continue
            if dim.label is not ValidationStatus.PASS:
                return validation_result

        # 3) Reason judge harus menyebut "filter default pegawai aktif" agar
        #    repair ini relevan. Bila FAIL where karena alasan lain (mis.
        #    filter periode/jabatan keliru), jangan ditimpa.
        reason_normalized = (where_verdict.reason or "").lower()
        keywords = (
            "filter default",
            "pegawai aktif",
            "status_pegawai",
            "kedudukan_pegawai",
        )
        if not any(k in reason_normalized for k in keywords):
            return validation_result

        # 4) User eksplisit minta non-aktif → tidak ada repair (filter
        #    memang sengaja tidak dipasang dan judge salah menilai FAIL,
        #    tapi membiarkan FAIL lebih aman daripada menimpa silently).
        try:
            if self._sql_generator._user_requests_non_active(user_query):
                return validation_result
        except Exception:
            return validation_result

        candidate_sql = validation_result.final_sql
        if not candidate_sql:
            return validation_result

        # 5) Tentukan strategi repair berdasarkan tabel yang disentuh SQL.
        try:
            references_pegawai = self._sql_generator._references_pegawai_table(
                candidate_sql
            )
            references_period_employees = self._references_period_employees(
                candidate_sql
            )
        except Exception:
            return validation_result

        repaired_sql = candidate_sql
        repair_trigger: str | None = None
        repair_feedback: str | None = None
        new_where_reason: str | None = None

        if references_pegawai:
            # Sub-case A: SQL menyentuh pegawai_tm. Coba injeksi mekanis;
            # bila sudah no-op berarti filter sebenarnya sudah ada (judge
            # mungkin tidak membaca subquery), kita override saja.
            try:
                repaired_sql = self._sql_generator._inject_default_pegawai_filters(
                    candidate_sql
                )
            except Exception:
                return validation_result

            if repaired_sql == candidate_sql:
                repair_trigger = "deterministic_active_filter_override_pegawai"
                repair_feedback = (
                    "Filter default pegawai aktif sudah hadir di SQL "
                    "(termasuk di subquery/CTE bila ada). Verdict where=FAIL "
                    "ditimpa menjadi PASS oleh post-processing deterministik."
                )
                new_where_reason = (
                    "Override deterministik: filter default pegawai aktif "
                    "(status_pegawai + kedudukan_pegawai) sudah ada di SQL."
                )
            else:
                repair_trigger = "deterministic_active_filter_inject"
                repair_feedback = (
                    "Post-processing deterministik menambahkan filter default "
                    "pegawai aktif ke klausa WHERE tanpa memanggil LLM."
                )
                new_where_reason = (
                    "Filter default pegawai aktif (status_pegawai + "
                    "kedudukan_pegawai) di-inject otomatis oleh "
                    "post-processing deterministik."
                )
        elif references_period_employees:
            # Sub-case B: SQL hanya menyentuh mantel.period_employees yang
            # tidak punya kolom status_pegawai/kedudukan_pegawai. Override
            # FAIL menjadi PASS karena aturan tidak applicable.
            repair_trigger = "deterministic_active_filter_override_period_employees"
            repair_feedback = (
                "SQL hanya menyentuh mantel.period_employees (snapshot "
                "per-periode pegawai aktif). Aturan default active filter "
                "tidak berlaku karena tabel ini tidak punya kolom "
                "status_pegawai/kedudukan_pegawai."
            )
            new_where_reason = (
                "Tidak berlaku: mantel.period_employees adalah snapshot "
                "per-periode pegawai aktif (tidak ada kolom "
                "status_pegawai/kedudukan_pegawai)."
            )
        else:
            # Tidak ada tabel yang bisa di-repair → biarkan FAIL apa adanya.
            return validation_result

        # 6) Bangun ulang ValidationResult (frozen dataclass) dengan rubric
        #    yang sudah di-upgrade dan revisi tambahan untuk audit trail.
        new_dimensions = dict(rubric.dimensions)
        new_dimensions["where"] = RubricDimensionVerdict(
            label=ValidationStatus.PASS,
            reason=new_where_reason or "Override deterministik.",
        )
        new_rubric = replace(
            rubric,
            overall=ValidationStatus.PASS,
            dimensions=new_dimensions,
        )

        new_revision = ValidationRevision(
            iteration=len(validation_result.revisions) + 1,
            trigger=repair_trigger,
            feedback=repair_feedback or "",
            sql_before=candidate_sql,
            sql_after=repaired_sql,
        )
        new_revisions = list(validation_result.revisions) + [new_revision]

        existing_explanation = (validation_result.explanation or "").strip()
        appended_note = (
            "Repair deterministik diterapkan: dimensi where di-upgrade ke "
            "PASS karena aturan default active filter sudah dipenuhi atau "
            "tidak applicable pada tabel target."
        )
        if existing_explanation:
            new_explanation = f"{existing_explanation} | {appended_note}"
        else:
            new_explanation = appended_note

        return replace(
            validation_result,
            status=ValidationStatus.PASS,
            final_sql=repaired_sql,
            explanation=new_explanation,
            rubric=new_rubric,
            revisions=new_revisions,
        )

    def _build_validation_payload(
        self,
        validation_result: ValidationResult,
    ) -> dict[str, Any]:
        """Bentuk dict yang siap dipakai sebagai field tambahan ``PipelineResult``.

        Bila status SKIPPED (mis. ``CHATBOT_VALIDATION_LEVEL=none`` atau
        procedural-memory hit yang melewati validasi), kembalikan dict kosong
        sehingga ``PipelineResult.validation_*`` tetap ``None`` dan respons API
        tidak menyertakan metadata validasi sama sekali. Ini menjaga "perilaku
        lama persis sama" yang disyaratkan untuk mode ``none``.
        """
        if validation_result.status is ValidationStatus.SKIPPED:
            return {}

        rubric_payload: dict[str, dict[str, str]] | None = None
        if validation_result.rubric is not None:
            rubric_payload = validation_result.rubric.as_payload()

        revisions_payload: list[dict[str, Any]] | None = None
        if validation_result.revisions:
            revisions_payload = [
                {
                    "iteration": rev.iteration,
                    "trigger": rev.trigger,
                    "feedback": rev.feedback,
                    "sql_before": rev.sql_before,
                    "sql_after": rev.sql_after,
                }
                for rev in validation_result.revisions
            ]

        return {
            "validation_status": validation_result.status.value,
            "validation_iterations": {
                "execution": validation_result.execution_iterations,
                "semantic": validation_result.semantic_iterations,
            },
            "validation_rubric": rubric_payload,
            "validation_revisions": revisions_payload,
        }

    def _fallback_tables(
        self,
        keywords: list[str],
        schema_tables: list[dict[str, Any]],
    ) -> dict[str, RetrievedTable]:
        scored: list[tuple[str, RetrievedTable]] = []

        lowered_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]
        for table in schema_tables:
            table_schema = str(table["schema"])
            table_name = str(table["name"])
            table_key = f"{table_schema}.{table_name}"
            table_name_lower = table_name.lower()

            table_score = 0.0
            matched_columns: dict[str, float] = {}

            for keyword in lowered_keywords:
                if keyword in table_name_lower:
                    table_score += 1.0

                for column in table["columns"]:
                    column_name = str(column["name"])
                    if keyword in column_name.lower():
                        table_score += 0.8
                        matched_columns[column_name] = max(
                            matched_columns.get(column_name, 0.0),
                            0.8,
                        )

            if table_score > 0:
                scored.append(
                    (
                        table_key,
                        RetrievedTable(
                            schema=table_schema,
                            table=table_name,
                            score=table_score,
                            column_scores=matched_columns,
                        ),
                    )
                )

        if not scored:
            for table in schema_tables[: self._config.max_retrieved_tables]:
                table_schema = str(table["schema"])
                table_name = str(table["name"])
                table_key = f"{table_schema}.{table_name}"
                scored.append(
                    (
                        table_key,
                        RetrievedTable(
                            schema=table_schema,
                            table=table_name,
                            score=0.1,
                            column_scores={},
                        ),
                    )
                )

        scored.sort(key=lambda item: item[1].score, reverse=True)
        limited = scored[: self._config.max_retrieved_tables]
        return {table_key: table for table_key, table in limited}

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _normalize_identifier(identifier: str) -> str:
        return re.sub(r"_+", "", identifier.strip().lower())

    def _guess_column_by_tokens(
        self,
        raw_column_name: str,
        candidate_columns: list[str],
    ) -> str | None:
        # Heuristic fallback for semantic aliases generated by the model,
        # e.g. universitas_atau_sekolah -> namasekolah.
        normalized_raw = self._normalize_identifier(raw_column_name)

        mapped_target = _COLUMN_ALIAS_CANONICAL_MAP.get(normalized_raw)
        if mapped_target:
            mapped_candidates = [
                candidate
                for candidate in candidate_columns
                if self._normalize_identifier(candidate) == mapped_target
            ]
            if len(mapped_candidates) == 1:
                return mapped_candidates[0]

        raw_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", raw_column_name.lower())
            if token and token not in _COLUMN_ALIAS_STOPWORDS and len(token) >= 4
        ]
        if not raw_tokens:
            return None

        expanded_tokens = set(raw_tokens)
        for token in list(raw_tokens):
            for probe, expansions in _COLUMN_TOKEN_EXPANSIONS.items():
                if probe == token or probe in token:
                    expanded_tokens.update(expansions)

        scores: dict[str, int] = {}
        for candidate_column in candidate_columns:
            candidate_key = self._normalize_identifier(candidate_column)
            if not candidate_key:
                continue

            score = 0
            for token in expanded_tokens:
                if token in candidate_key:
                    score += len(token)

            if score > 0:
                scores[candidate_column] = score

        if not scores:
            return None

        best_score = max(scores.values())
        best_candidates = [
            candidate for candidate, score in scores.items() if score == best_score
        ]
        if len(best_candidates) != 1:
            return None

        # Conservative thresholds to avoid accidental over-corrections.
        if best_score >= 8:
            return best_candidates[0]
        if len(scores) == 1 and best_score >= 6:
            return best_candidates[0]
        return None

    def _build_table_metadata(
        self,
        schema_tables: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        table_metadata: dict[str, dict[str, Any]] = {}
        base_knowledge_aliases = _load_base_knowledge_column_aliases()

        for table in schema_tables:
            schema_name = str(table.get("schema") or "").strip()
            table_name = str(table.get("name") or "").strip()
            if not schema_name or not table_name:
                continue

            columns_raw = [
                str(column.get("name") or "").strip()
                for column in table.get("columns", [])
                if str(column.get("name") or "").strip()
            ]
            columns_exact = set(columns_raw)
            columns_lower_map: dict[str, str] = {
                column_name.lower(): column_name for column_name in columns_raw
            }

            columns_normalized: dict[str, set[str]] = {}
            for column_name in columns_raw:
                normalized_key = self._normalize_identifier(column_name)
                if not normalized_key:
                    continue
                columns_normalized.setdefault(normalized_key, set()).add(column_name)

            table_key = f"{schema_name.lower()}.{table_name.lower()}"
            table_aliases_raw = base_knowledge_aliases.get(table_key, {})
            table_aliases: dict[str, set[str]] = {}
            for alias_key, candidate_columns in table_aliases_raw.items():
                valid_columns = {
                    candidate for candidate in candidate_columns if candidate in columns_exact
                }
                if valid_columns:
                    table_aliases[alias_key] = valid_columns

            table_name_canonical = (
                table_name
                if re.fullmatch(r"[a-z_][a-z0-9_]*", table_name)
                else self._quote_identifier(table_name)
            )
            canonical_ref = f"{schema_name}.{table_name_canonical}"

            table_metadata[table_key] = {
                "schema": schema_name,
                "table": table_name,
                "canonical_ref": canonical_ref,
                "columns_raw": columns_raw,
                "columns_exact": columns_exact,
                "columns_lower_map": columns_lower_map,
                "columns_normalized": columns_normalized,
                "column_aliases": table_aliases,
            }

        return table_metadata

    def _autocorrect_sql_identifiers(
        self,
        sql: str,
        schema_tables: list[dict[str, Any]],
    ) -> str:
        table_metadata = self._build_table_metadata(schema_tables)
        if not table_metadata:
            return sql

        corrected_sql = sql

        # Normalize table references so case-sensitive table names are properly quoted.
        for table_key in sorted(table_metadata.keys(), key=len, reverse=True):
            metadata = table_metadata[table_key]
            pattern = re.compile(
                rf"\b{re.escape(table_key)}\b",
                re.IGNORECASE,
            )
            corrected_sql = pattern.sub(str(metadata["canonical_ref"]), corrected_sql)

        alias_map: dict[str, dict[str, Any]] = {}
        table_alias_pattern = re.compile(
            r"\b(?:from|join)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*\.(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*))"
            r"(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            re.IGNORECASE,
        )

        for match in table_alias_pattern.finditer(corrected_sql):
            table_ref = str(match.group(1) or "").strip()
            alias = str(match.group(2) or "").strip()
            if "." not in table_ref:
                continue

            schema_name, table_token = table_ref.split(".", 1)
            table_name = table_token.strip().strip('"')
            metadata = table_metadata.get(f"{schema_name.lower()}.{table_name.lower()}")
            if metadata is None:
                continue

            resolved_alias = alias or table_name
            alias_map[resolved_alias] = metadata
            alias_map[resolved_alias.lower()] = metadata

        if not alias_map:
            return corrected_sql

        column_ref_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

        def _replace_column_ref(match: re.Match[str]) -> str:
            alias = str(match.group(1) or "")
            column_name = str(match.group(2) or "")

            metadata = alias_map.get(alias) or alias_map.get(alias.lower())
            if metadata is None:
                return match.group(0)

            columns_exact: set[str] = metadata["columns_exact"]
            if column_name in columns_exact:
                return match.group(0)

            columns_lower_map: dict[str, str] = metadata["columns_lower_map"]
            case_match = columns_lower_map.get(column_name.lower())
            if case_match and case_match != column_name:
                return f"{alias}.{case_match}"

            normalized_key = self._normalize_identifier(column_name)
            columns_normalized: dict[str, set[str]] = metadata["columns_normalized"]
            candidates = columns_normalized.get(normalized_key, set())
            if len(candidates) == 1:
                resolved_column = next(iter(candidates))
                if resolved_column != column_name:
                    return f"{alias}.{resolved_column}"

            column_aliases: dict[str, set[str]] = metadata.get("column_aliases", {})
            alias_candidates = column_aliases.get(normalized_key, set())
            if len(alias_candidates) == 1:
                resolved_column = next(iter(alias_candidates))
                if resolved_column != column_name:
                    return f"{alias}.{resolved_column}"

            token_match = self._guess_column_by_tokens(
                raw_column_name=column_name,
                candidate_columns=list(metadata.get("columns_raw", [])),
            )
            if token_match and token_match != column_name:
                return f"{alias}.{token_match}"

            return match.group(0)

        corrected_sql = column_ref_pattern.sub(_replace_column_ref, corrected_sql)
        return corrected_sql
