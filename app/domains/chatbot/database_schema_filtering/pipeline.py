import asyncio
import csv
import logging
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.llm import LLMAdapter

logger = logging.getLogger(__name__)

from ..repositories import ChatbotRepository
from ..semantic_disambiguation.parsers import parse_detection_output
from ..semantic_disambiguation.types import (
    AmbiguityDetectionResult,
    InterpretationOption,
)
from app.core.config import settings as app_settings
from .schema_uq import SchemaUQResult, compute_schema_uq
from .ambiguous_lexicon import build_clarification, match_ambiguous_term
from .schema_interpretation import (
    build_schema_interpretation_prompt,
    format_schema_catalog,
    parse_schema_interpretation,
)
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

# Budget token untuk satu sample interpretasi-skema Stage 2 UQ. Interpretasi
# adalah satu baris pemetaan ringkas, jadi tidak butuh banyak token.
_SCHEMA_UQ_SAMPLE_MAX_TOKENS = 400


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
        # Dedicated embedding client (lazy) untuk Stage 2 UQ. Sengaja TIDAK pakai
        # ``llm_adapter.embeddings`` (model vector-store knowledge entities)
        # karena interpretasi skema di-embed dengan model NL keluarga kalibrasi
        # UQ (``schema_uq_embedding_model``). Lihat ``_get_uq_embeddings``.
        self._uq_embeddings: Any = None
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

        # Schema-level keyword gate: drop tables in gated schemas unless
        # the raw query OR any extracted keyword contains one of the
        # schema's required terms. Suppresses noise from special-purpose
        # schemas whose embeddings often nyangkut di Top-N untuk query
        # generik (default: ``mantel`` cuma masuk kalau user nyebut
        # "pool"). Applied BOTH before AND after FK closure: the BEFORE
        # pass prevents a gated table from pulling its FK neighbors into
        # the predicted set via closure; the AFTER pass prevents closure
        # from re-introducing a gated table as a missing endpoint of
        # some ungated table's FK edge. Architect review (May 2026)
        # caught the bypass when only the BEFORE pass existed.
        self._apply_schema_keyword_gate(predicted_tables, query, keywords)

        # FK closure: for every documented FK edge that has exactly one
        # endpoint in the predicted set, pull in the missing endpoint as a
        # sentinel entry (score=0.0) carrying just its FK column at 1.0.
        # This rescues join paths the retriever missed — e.g. when a query
        # like "pegawai berpendidikan S2 + gelar + jabatan" surfaces
        # ``jabatan_tm`` and ``V_PENDIDIKAN_TERAKHIR`` but not ``pegawai_tm``
        # itself, the closure adds ``pegawai_tm`` back so its FK columns
        # (``jabatan_id``, ``pendidikan_top_id``, …) become visible in the
        # Stage 2 trace with proper ``FK→…`` badges.
        if getattr(self._config, "auto_include_keys", False):
            self._expand_predicted_via_fk_closure(predicted_tables)
            # Re-apply gate: closure can introduce a gated-schema table
            # as the missing endpoint of some ungated table's FK edge.
            self._apply_schema_keyword_gate(predicted_tables, query, keywords)

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

        join_graph, column_roles = await self._build_join_graph_and_roles(
            predicted_tables=predicted_tables,
        )

        return PreparedSchemaContext(
            keywords=keywords,
            predicted_tables=predicted_tables,
            context=context,
            schema_tables=schema_tables,
            join_graph=join_graph,
            column_roles=column_roles,
        )

    def _get_uq_embeddings(self) -> Any:
        """Lazy-init dedicated embedding client untuk interpretasi skema.

        Sengaja TIDAK pakai ``llm_adapter.embeddings`` (model vector-store
        knowledge entities) supaya parity dengan kalibrasi UQ
        (``schema_uq_embedding_model``, default ``text-embedding-3-small``).
        """
        if self._uq_embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr

            self._uq_embeddings = OpenAIEmbeddings(
                model=self._config.schema_uq_embedding_model,
                api_key=SecretStr(app_settings.OPENAI_API_KEY),
                base_url=app_settings.AI_BASE_URL,
            )
            logger.info(
                "schema_uq_embeddings_init",
                extra={"model": self._config.schema_uq_embedding_model},
            )
        return self._uq_embeddings

    async def _embed_interpretations(self, samples: list[str]) -> list[list[float]]:
        """Embed interpretasi skema di worker thread (LangChain sync only)."""
        loop = asyncio.get_running_loop()
        client = self._get_uq_embeddings()

        def _embed_all() -> list[list[float]]:
            return [list(client.embed_query(s)) for s in samples]

        try:
            return await loop.run_in_executor(None, _embed_all)
        except Exception:
            logger.warning("schema_uq_embedding_failed", exc_info=True)
            return []

    async def _sample_schema_interpretation(
        self, query: str, schema_catalog: str
    ) -> str:
        """Satu panggilan sampler interpretasi-skema @ T_SAMPLING.

        Mengembalikan string ``interpretasi`` (pemetaan konsep→tabel/kolom)
        atau string kosong bila gagal. String kosong diperlakukan sebagai
        ``ERROR`` fingerprint di UQ — DISENGAJA mirror Stage 1.
        """
        prompt = build_schema_interpretation_prompt(query, schema_catalog)
        try:
            response = await asyncio.wait_for(
                self._llm_adapter.instruct.bind(
                    max_tokens=_SCHEMA_UQ_SAMPLE_MAX_TOKENS,
                    temperature=float(self._config.schema_uq_t_sampling),
                ).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Anda penginterpretasi skema. "
                                "Keluarkan JSON valid saja sesuai format."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                ),
                timeout=30,
            )
        except Exception:
            logger.warning("schema_uq_sample_failed", exc_info=True)
            return ""
        content = str(getattr(response, "content", "") or "")
        return parse_schema_interpretation(content)

    async def assess_schema_uncertainty(
        self, query: str, *, prepared: PreparedSchemaContext | None = None
    ) -> SchemaUQResult:
        """Stage 2 UQ: ukur ketidakpastian INTERPRETASI skema (bukan retrieval).

        Skema candidate diretrieval SEKALI (oleh ``prepare_context``); hasilnya
        (``prepared``) dipakai untuk membangun *katalog skema* level-kolom yang
        diberikan ke sampler. Katalog memuat kolom konkret tabel kandidat (bukan
        sekadar nama tabel, dan tanpa filter-similarity yang membuang kolom
        alternatif) supaya istilah ambigu seperti "senior" punya alternatif
        pemetaan yang benar-benar terlihat LLM. Tanpa grounding kolom ini, setiap
        sample kolaps ke pemetaan tabel yang sama (H_norm=0, ambiguitas lolos).

        Di sini LLM meng-generate ``M`` interpretasi skema (pemetaan konsep query
        → kolom konkret) @ ``schema_uq_t_sampling`` (paralel), lalu interpretasi
        tersebut di-embed dan di-cluster secara semantik (reuse
        ``semantic_disambiguation/uq.py``). Bila interpretasi bercabang
        (≥2 cluster interpretasi BERBEDA, ERROR dikecualikan) → minta
        klarifikasi sebelum SQL. H_norm/τ_U tetap dilaporkan sebagai diagnostik
        kontinu, tetapi keputusan klarifikasi memakai jumlah interpretasi —
        bukan ambang entropi yang ter-normalisasi log(M) (lihat schema_uq.py).

        Yang di-cluster adalah OUTPUT REASONING LLM (teks interpretasi), BUKAN
        schema candidate maupun embedding tabel/kolom secara langsung.

        Bila fitur dimatikan / query kosong → hasil netral (is_uncertain=False)
        sehingga pemanggil bisa lanjut tanpa perubahan perilaku.
        """
        cfg = self._config
        tau_u = float(cfg.schema_uq_tau_u)
        query = (query or "").strip()
        if not cfg.schema_uq_enabled or not query:
            return SchemaUQResult(
                enabled=bool(cfg.schema_uq_enabled),
                h_norm=0.0,
                tau_u=tau_u,
                m_samples=0,
                valid_samples=0,
                unique_clusters=0,
                majority_ratio=0.0,
                is_uncertain=False,
                representative_text="",
                interpretations=[],
                n_error=0,
            )

        m_total = int(cfg.schema_uq_m_sampling)

        # Katalog skema level-kolom dari hasil retrieval. Tanpa ini sampler
        # hanya melihat nama tabel (atau kolom ter-filter similarity) sehingga
        # istilah ambigu kolaps ke pemetaan tabel yang sama di semua sample.
        schema_catalog = ""
        if prepared is not None:
            schema_catalog = format_schema_catalog(
                prepared.predicted_tables,
                prepared.schema_tables,
                prepared.column_roles,
            )

        # Step 1: M interpretasi paralel @ T_SAMPLING.
        samples = await asyncio.gather(
            *[
                self._sample_schema_interpretation(query, schema_catalog)
                for _ in range(m_total)
            ]
        )
        valid_samples = [s for s in samples if s and s.strip()]

        # Semua sample gagal → tidak ada apa pun untuk di-cluster. Short-circuit
        # menghindari call embedding kosong; tetap konsisten (H_norm=0).
        if not valid_samples:
            logger.warning(
                "schema_uq_degraded_no_samples", extra={"m_total": m_total}
            )
            return SchemaUQResult(
                enabled=True,
                h_norm=0.0,
                tau_u=tau_u,
                m_samples=m_total,
                valid_samples=0,
                unique_clusters=0,
                majority_ratio=0.0,
                is_uncertain=False,
                representative_text="",
                interpretations=[],
                n_error=m_total,
                degraded=True,
                degraded_reason="sampling_failed",
            )

        # Step 2: embed interpretasi valid.
        embeddings = await self._embed_interpretations(valid_samples)
        if len(embeddings) != len(valid_samples) or not embeddings:
            # Embedding gagal: jangan diam-diam jadi uncertain — fallback netral
            # (confident) supaya pipeline lanjut tanpa klarifikasi palsu.
            logger.warning("schema_uq_degraded_embedding_failed")
            return SchemaUQResult(
                enabled=True,
                h_norm=0.0,
                tau_u=tau_u,
                m_samples=m_total,
                valid_samples=len(valid_samples),
                unique_clusters=1,
                majority_ratio=1.0,
                is_uncertain=False,
                representative_text=valid_samples[0],
                interpretations=[],
                n_error=max(0, m_total - len(valid_samples)),
                degraded=True,
                degraded_reason="embedding_failed",
            )

        # Step 3: cluster + entropy via matematika Stage 1.
        result = compute_schema_uq(
            valid_samples,
            embeddings,
            m_total=m_total,
            tau_cluster=float(cfg.schema_uq_tau_cluster),
            tau_u=tau_u,
            enabled=True,
        )
        logger.info(
            "schema_uq_assessed",
            extra={
                "h_norm": round(result.h_norm, 4),
                "tau_u": round(result.tau_u, 4),
                "is_uncertain": result.is_uncertain,
                "unique_clusters": result.unique_clusters,
                "valid_samples": result.valid_samples,
                "n_error": result.n_error,
            },
        )
        return result

    def _build_schema_clarification_prompt(
        self,
        query: str,
        interpretations: list,
        schema_context: str,
    ) -> str:
        lines: list[str] = []
        for idx, it in enumerate(interpretations, start=1):
            lines.append(f"- I{idx} (didukung {it.support} sampel): {it.text}")
        interp_block = "\n".join(lines)
        ctx = (schema_context or "").strip()
        ctx_block = f"\n\nRingkasan skema relevan:\n{ctx[:1500]}" if ctx else ""
        return (
            "Pertanyaan pengguna terhadap basis data BPOM memiliki lebih dari "
            "satu interpretasi skema yang sama-sama masuk akal (reasoning LLM "
            "atas kandidat skema bercabang — mis. sebuah istilah bisa dipetakan "
            "ke kolom yang berbeda). Tugas Anda: susun satu pertanyaan "
            "klarifikasi singkat dalam Bahasa Indonesia yang membantu pengguna "
            "memilih maksudnya, beserta opsi-opsi interpretasi.\n\n"
            f"Pertanyaan pengguna:\n{query}\n\n"
            f"Kandidat interpretasi (dari sampling reasoning skema):\n"
            f"{interp_block}{ctx_block}\n\n"
            "Keluarkan HANYA JSON valid dengan bentuk:\n"
            "{\n"
            '  "is_ambiguous": true,\n'
            '  "ambiguity_type": "column",\n'
            '  "clarification_question": "<pertanyaan klarifikasi ramah, tanpa istilah teknis>",\n'
            '  "interpretation_options": [\n'
            '    {"label": "<label singkat dalam bahasa bisnis>", "description": "<penjelasan data yang akan diambil>"}\n'
            "  ]\n"
            "}\n"
            "Gunakan bahasa bisnis yang dapat dipahami pengguna non-teknis; "
            "JANGAN menampilkan nama tabel/kolom mentah. Sertakan satu opsi per "
            "interpretasi di atas."
        )

    def _fallback_schema_clarification(
        self, interpretations: list
    ) -> AmbiguityDetectionResult:
        """Klarifikasi deterministik bila LLM gagal/JSON invalid.

        UQ sudah memutuskan ambigu; JANGAN diam-diam balik jadi non-ambiguous.
        Tawarkan teks interpretasi apa adanya sebagai opsi.
        """
        options: list[InterpretationOption] = []
        seen: set[str] = set()
        for idx, it in enumerate(interpretations[:4], start=1):
            label = (it.text or f"Interpretasi {idx}").strip()[:120]
            if not label or label in seen:
                continue
            seen.add(label)
            options.append(
                InterpretationOption(label=label, description="")
            )
        return AmbiguityDetectionResult(
            is_ambiguous=bool(options),
            ambiguity_type="column",
            clarification_question=(
                "Pertanyaan Anda dapat diartikan dengan beberapa cara berbeda. "
                "Mana yang Anda maksud?"
            ),
            interpretation_options=options,
        )

    async def generate_schema_clarification(
        self,
        query: str,
        schema_uq: SchemaUQResult,
        *,
        schema_context: str = "",
    ) -> AmbiguityDetectionResult:
        """Bangun ``AmbiguityDetectionResult`` klarifikasi dari hasil UQ skema.

        Telemetri UQ (h_norm, τ_U, m_samples, unique_clusters) di-inject ke
        hasil sehingga jejak/UI ambiguity yang sudah ada otomatis menampilkan
        sinyal Stage 2 tanpa perubahan downstream.
        """
        interpretations = schema_uq.interpretations[:4]
        detection: AmbiguityDetectionResult | None = None
        try:
            prompt = self._build_schema_clarification_prompt(
                query, interpretations, schema_context
            )
            response = await asyncio.wait_for(
                self._llm_adapter.instruct.bind(
                    max_tokens=768,
                    temperature=0.0,
                ).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Anda asisten data BPOM. Keluarkan JSON valid saja."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                ),
                timeout=20,
            )
            content = str(getattr(response, "content", "") or "").strip()
            detection = parse_detection_output(content)
        except Exception:
            logger.warning("schema_clarification_llm_failed", exc_info=True)
            detection = None

        if (
            detection is None
            or not detection.is_ambiguous
            or not detection.interpretation_options
        ):
            detection = self._fallback_schema_clarification(interpretations)

        return AmbiguityDetectionResult(
            is_ambiguous=True,
            ambiguity_type=detection.ambiguity_type or "column",
            clarification_question=detection.clarification_question
            or (
                "Pertanyaan Anda dapat diartikan dengan beberapa cara berbeda. "
                "Mana yang Anda maksud?"
            ),
            interpretation_options=detection.interpretation_options,
            h_norm=schema_uq.h_norm,
            tau_u=schema_uq.tau_u,
            m_samples=schema_uq.m_samples,
            unique_clusters=schema_uq.unique_clusters,
        )

    def build_lexicon_clarification(
        self, query: str
    ) -> tuple[str, AmbiguityDetectionResult] | None:
        """Safeguard DETERMINISTIK (terpisah dari UQ): cek kamus istilah-ambigu.

        Bila ``query`` memuat istilah domain yang memang ambigu (mis. "senior"),
        kembalikan ``(istilah, klarifikasi)`` siap pakai — TANPA LLM. Dipakai oleh
        pemanggil HANYA ketika UQ tidak memicu klarifikasi, untuk MENJAMIN kueri
        ambigu tetap diklarifikasi. Mengembalikan ``None`` bila fitur dimatikan
        atau tidak ada istilah yang cocok. Verdict UQ tidak diubah olehnya.
        """
        if not getattr(self._config, "ambiguous_lexicon_enabled", True):
            return None
        entry = match_ambiguous_term(query)
        if entry is None:
            return None
        return str(entry["term"]), build_clarification(entry)

    def _apply_schema_keyword_gate(
        self,
        predicted_tables: dict[str, "RetrievedTable"],
        query: str,
        keywords: list[str],
    ) -> None:
        """Drop tables whose schema is gated and whose required keyword
        is not present in the query or any extracted keyword. In-place.
        Idempotent — safe to call multiple times.
        """
        gates = getattr(self._config, "schema_keyword_gates", None) or {}
        if not gates or not predicted_tables:
            return
        haystack = " ".join([query, *keywords]).lower()
        dropped: list[str] = []
        for table_key in list(predicted_tables.keys()):
            if "." not in table_key:
                continue
            schema_name = table_key.split(".", 1)[0]
            required = gates.get(schema_name)
            if not required:
                continue
            if any(term in haystack for term in required):
                continue
            del predicted_tables[table_key]
            dropped.append(table_key)
        if dropped:
            logger.info(
                "schema_keyword_gate_dropped",
                extra={"dropped_tables": dropped, "gates": gates},
            )

    def _expand_predicted_via_fk_closure(
        self,
        predicted_tables: dict[str, "RetrievedTable"],
    ) -> None:
        """Mutate ``predicted_tables`` in place, adding missing endpoints
        of any documented FK edge with exactly one endpoint already in the
        set.

        Newly-added tables get ``score=0.0`` as a sentinel (clearly below
        any retrieval threshold) so the Stage 2 trace can distinguish them
        from semantically-retrieved tables. Their ``column_scores`` start
        with just the FK column at 1.0 so the downstream role mapper
        labels it ``FK→<target>``.

        Only documented edges (parsed from ``base_knowledge_rasl.csv``) are
        used for closure — they reflect the curated relations the user
        explicitly described, which is the right semantic level for "pull
        in the obvious neighbor". This avoids an extra DB round-trip and
        keeps the expansion deterministic.
        """
        from .fk_graph import build_fk_graph  # local import to avoid cycles

        if not predicted_tables:
            return

        fk_graph = build_fk_graph(
            allowed_tables=self._config.allowed_tables,
            inferred_edges=[],
        )

        predicted_keys = set(predicted_tables.keys())

        for edge in fk_graph.iter_edges():
            from_in = edge.from_table in predicted_keys
            to_in = edge.to_table in predicted_keys
            if from_in == to_in:
                # Both endpoints already in (edge contributes to join_graph
                # via the existing path) or both out (no anchor in the
                # retrieved set, would be a wild guess to pull in).
                continue

            missing_table = edge.to_table if from_in else edge.from_table
            missing_col = edge.to_column if from_in else edge.from_column

            if missing_table in predicted_tables:
                # Already pulled in by an earlier edge in this loop — just
                # ensure this FK column is also surfaced at score 1.0.
                existing = predicted_tables[missing_table]
                col_scores = dict(existing.column_scores or {})
                if col_scores.get(missing_col, 0.0) < 1.0:
                    col_scores[missing_col] = 1.0
                    predicted_tables[missing_table] = replace(
                        existing, column_scores=col_scores
                    )
                continue

            if "." not in missing_table:
                continue
            schema_name, table_name = missing_table.split(".", 1)
            predicted_tables[missing_table] = RetrievedTable(
                schema=schema_name,
                table=table_name,
                score=0.0,
                column_scores={missing_col: 1.0},
            )

    async def _build_join_graph_and_roles(
        self,
        *,
        predicted_tables: dict[str, "RetrievedTable"],
    ) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
        """Build ``(join_graph, column_roles)`` for the Stage 2 trace output.

        - ``join_graph``: edges (documented + inferred) whose both endpoints
          are in the predicted-table set.
        - ``column_roles``: ``{table_key: {col: role}}`` where role is one of
          ``"PK"`` / ``"FK→<target>"`` / ``"display"``.

        Both pieces are computed here so that all four callers of
        :py:meth:`prepare_context` get them for free without re-fetching the
        repository — and downstream stage-trace builders stay pure.
        """
        from .fk_graph import build_fk_graph  # local import to avoid cycles

        predicted_keys: set[str] = set(predicted_tables.keys())
        if not predicted_keys:
            return [], {}

        allowed_for_predicted: dict[str, list[str]] = {}
        for key in predicted_keys:
            if "." not in key:
                continue
            schema_name, table_name = key.split(".", 1)
            allowed_for_predicted.setdefault(schema_name, []).append(table_name)

        try:
            inferred_edges = await self._repository.get_foreign_key_edges(
                allowed_for_predicted
            )
        except Exception:
            inferred_edges = []

        try:
            key_columns = await self._repository.get_table_key_columns(
                allowed_for_predicted
            )
        except Exception:
            key_columns = {}

        fk_graph = build_fk_graph(
            allowed_tables=self._config.allowed_tables,
            inferred_edges=inferred_edges,
        )

        join_graph = [edge.to_dict() for edge in fk_graph.edges_within(predicted_keys)]

        # Inject every FK origin column known to fk_graph (documented +
        # inferred) into the predicted table's ``column_scores`` at 1.0,
        # so that logical-only FKs (declared in base_knowledge entity_text
        # but without a formal information_schema constraint — e.g.
        # ``pegawai_tm.jabatan_id``, ``satker_top_id``, ``eselon_id``,
        # ``pendidikan_top_id``) become visible in the Stage 2 trace with
        # the same ``FK→…`` badge as formal FKs. Without this step the
        # column never lands in ``column_scores`` (semantic retrieval
        # rarely surfaces ID columns), so the join-graph edge appears in
        # the Join Path panel while the source-side column silently goes
        # un-labelled — confusing for users inspecting the trace.
        #
        # Gated behind ``auto_include_keys`` to match the existing
        # information_schema-based injection in ``TableRetriever``.
        if getattr(self._config, "auto_include_keys", False):
            for edge in fk_graph.edges_within(predicted_keys):
                origin = predicted_tables.get(edge.from_table)
                if origin is None:
                    continue
                col_scores = dict(origin.column_scores or {})
                if col_scores.get(edge.from_column, 0.0) >= 1.0:
                    continue
                col_scores[edge.from_column] = 1.0
                predicted_tables[edge.from_table] = replace(
                    origin, column_scores=col_scores
                )

        # Build per-column role map. Priority within one column:
        #   FK > PK > display. (A column can technically be both PK and FK
        # in a junction table — surface FK so the join is visible.)
        column_roles: dict[str, dict[str, str]] = {}
        for table_key, retrieved in predicted_tables.items():
            roles: dict[str, str] = {}
            key_col_set = {c.lower() for c in key_columns.get(table_key, set())}
            for col_name, score in (retrieved.column_scores or {}).items():
                fk_role = fk_graph.role_for(table_key, col_name)
                if fk_role:
                    roles[col_name] = fk_role
                    continue
                if col_name.lower() in key_col_set:
                    roles[col_name] = "PK"
                    continue
                # Display columns are injected by the retriever at score=0.99
                # exactly (see TableRetriever._inject_display_columns). Use a
                # tolerant compare to avoid floating-point surprises.
                if abs(float(score) - 0.99) < 1e-6:
                    roles[col_name] = "display"
            if roles:
                column_roles[table_key] = roles

        return join_graph, column_roles

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
