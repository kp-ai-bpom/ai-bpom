import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.domains.chatbot.core.config import (
    _resolve_stage_temperature,
    chatbot_settings as settings,
)


DEFAULT_ALLOWED_TABLES: dict[str, list[str]] = {
    "public": [
        "propinsi_tm",
        "kabupaten_tm",
        "kecamatan_tm",
        "pangkat_tm",
        "tipepegawai_tm",
        "eselon_tm",
        "pegawai_tm",
        "disabilitas_tm",
        "riwayatjabatan_th",
        "SIAP_SATKER_TOP",
        "jabatan_tm",
        "sk_pegawai_v",
    ],
    "siap": [
        "R_FUNGSI",
        "T_RIWAYAT_MUTASI",
        "V_PENDIDIKAN_TERAKHIR",
    ],
    "mantel": [
        "period_employees",
        "periods",
    ],
}


@dataclass(frozen=True)
class SemanticMemoryConfig:
    allowed_tables: dict[str, list[str]]
    vector_table: str
    sql_timeout_ms: int
    top_n_per_keyword: int
    max_retrieved_tables: int
    table_weight: float
    column_weight: float
    retrieval_threshold: float
    column_similarity_threshold: float
    max_context_chars: int
    keyword_retries: int
    sample_rows_per_table: int
    llm_temperature: float
    # ── Stage 2 schema UQ (multiple-interpretation, semantic clustering) ────
    # Diisi dari ``ChatbotSettings.CHATBOT_SCHEMA_UQ_*``. Bila
    # ``schema_uq_enabled`` False, pipeline melewati assessment dan perilaku
    # identik dengan sebelum fitur ini ada. ``schema_uq_t_sampling`` dipakai
    # sebagai temperature SAMPLER INTERPRETASI skema (T>0 supaya pemetaan
    # konsep→tabel/kolom bervariasi saat ambigu). ``schema_uq_embedding_model``
    # mengembed teks interpretasi untuk clustering. τ_cluster/τ_U STALE —
    # lihat core/config.py.
    schema_uq_enabled: bool = False
    schema_uq_m_sampling: int = 15
    schema_uq_t_sampling: float = 1.3
    schema_uq_tau_cluster: float = 0.72
    schema_uq_tau_u: float = 0.12
    schema_uq_embedding_model: str = "text-embedding-3-small"
    # Safeguard deterministik (TERPISAH dari UQ): kamus istilah-ambigu yang
    # menjamin klarifikasi walau UQ konvergen. Lihat ``ambiguous_lexicon.py``.
    ambiguous_lexicon_enabled: bool = True
    auto_include_keys: bool = False
    # Whitelist kolom display ("nama", "nip", "deskripsi", "satker_nama", ...)
    # untuk tabel master (_tm / _top / V_*). Di-inject oleh TableRetriever
    # bersama PK/FK ketika ``auto_include_keys`` aktif, tapi pakai score
    # marker khusus (0.99) supaya bisa dibedakan dari PK/FK (1.0) di log
    # debug Stage 4.
    display_cols: dict[str, list[str]] = None  # type: ignore[assignment]
    # Override KEY columns (PK / JOIN keys) untuk tabel tanpa PRIMARY KEY /
    # FOREIGN KEY constraint formal di ``information_schema``. AUGMENT hasil
    # ``ChatbotRepository.get_table_key_columns()`` — bukan replace. Bug
    # nyata Task #58: ``public.SIAP_SATKER_TOP`` (impor tanpa PK constraint)
    # gagal inject ``satker_id`` di Stage 2, recall column drop ~9 pp.
    key_cols_override: dict[str, list[str]] = None  # type: ignore[assignment]
    # Schema-level keyword gate: tabel di schema ini hanya boleh masuk
    # ``predicted_tables`` ketika query/keywords mengandung salah satu
    # kata kunci di list-nya (case-insensitive). Tujuannya menekan noise
    # dari schema "khusus" yang embeddings-nya bisa nyangkut di banyak
    # query generik (mis. schema ``mantel`` cuma relevan saat user
    # eksplisit menyebut "pool"). Schema yang tidak ada di dict ini lewat
    # tanpa filter. Diterapkan di ``pipeline.prepare_context`` setelah
    # retrieve, sebelum FK closure — supaya schema yang ter-gate tidak
    # ikut menarik tabel via closure.
    schema_keyword_gates: dict[str, list[str]] = None  # type: ignore[assignment]



def _normalize_display_cols(payload: Any) -> dict[str, list[str]]:
    """Normalize display-column whitelist payload.

    Keys must be ``"schema.table"`` strings; values must be non-empty lists
    of column names. Quoted identifiers (mis. ``SIAP_SATKER_TOP``) di-keep
    apa adanya — caller (TableRetriever) yang melakukan match case-sensitif
    terhadap ``table_key`` hasil retrieval.
    """
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for table_key, columns in payload.items():
        if not isinstance(table_key, str) or "." not in table_key:
            continue
        if not isinstance(columns, list):
            continue
        valid = [c for c in columns if isinstance(c, str) and c]
        if valid:
            normalized[table_key] = valid
    return normalized


def _normalize_allowed_tables(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return DEFAULT_ALLOWED_TABLES

    normalized: dict[str, list[str]] = {}
    for schema_name, table_names in payload.items():
        if not isinstance(schema_name, str):
            continue
        if not isinstance(table_names, list):
            continue

        valid_tables = [table for table in table_names if isinstance(table, str) and table]
        if valid_tables:
            normalized[schema_name] = valid_tables

    return normalized or DEFAULT_ALLOWED_TABLES


@lru_cache(maxsize=1)
def get_semantic_memory_config() -> SemanticMemoryConfig:
    try:
        parsed_allowed_tables = json.loads(settings.CHATBOT_ALLOWED_TABLES_JSON)
    except json.JSONDecodeError:
        parsed_allowed_tables = DEFAULT_ALLOWED_TABLES

    allowed_tables = _normalize_allowed_tables(parsed_allowed_tables)

    try:
        parsed_display_cols = json.loads(
            getattr(settings, "CHATBOT_DISPLAY_COLS_JSON", "{}")
        )
    except json.JSONDecodeError:
        parsed_display_cols = {}
    display_cols = _normalize_display_cols(parsed_display_cols)

    # ``CHATBOT_KEY_COLS_JSON`` reuses the same normalizer (key/value shape
    # identik). Default di ``core.config`` sudah memuat
    # {"public.SIAP_SATKER_TOP": ["satker_id"]}; getattr fallback '{}'
    # hanya jaga-jaga kalau settings di-monkeypatch tanpa atribut tsb.
    try:
        parsed_key_cols_override = json.loads(
            getattr(settings, "CHATBOT_KEY_COLS_JSON", "{}")
        )
    except json.JSONDecodeError:
        parsed_key_cols_override = {}
    key_cols_override = _normalize_display_cols(parsed_key_cols_override)

    # Schema-level keyword gates. Default: ``{"mantel": ["pool"]}`` —
    # schema ``mantel`` (snapshot pool data) cuma relevan saat user
    # eksplisit menyebut "pool", jika tidak embeddings-nya sering
    # nyangkut di Top-N untuk query generik dan menambah noise di
    # ``predicted_tables`` (mis. query "pegawai S2 + jabatan" ikut
    # ngambil ``period_employees`` padahal tidak diminta).
    try:
        parsed_schema_gates = json.loads(
            getattr(settings, "CHATBOT_SCHEMA_KEYWORD_GATES_JSON", "")
            or '{"mantel": ["pool"]}'
        )
    except json.JSONDecodeError:
        parsed_schema_gates = {"mantel": ["pool"]}
    schema_keyword_gates: dict[str, list[str]] = {}
    if isinstance(parsed_schema_gates, dict):
        for schema_name, keywords in parsed_schema_gates.items():
            if not isinstance(schema_name, str) or not isinstance(keywords, list):
                continue
            valid = [
                k.lower().strip() for k in keywords if isinstance(k, str) and k.strip()
            ]
            if valid:
                schema_keyword_gates[schema_name] = valid

    return SemanticMemoryConfig(
        allowed_tables=allowed_tables,
        vector_table=settings.CHATBOT_VECTOR_TABLE,
        sql_timeout_ms=max(1000, settings.CHATBOT_SQL_TIMEOUT_MS),
        top_n_per_keyword=max(1, settings.CHATBOT_TOP_N_PER_KEYWORD),
        max_retrieved_tables=max(1, settings.CHATBOT_MAX_RETRIEVED_TABLES),
        table_weight=max(0.0, settings.CHATBOT_TABLE_WEIGHT),
        column_weight=max(0.0, settings.CHATBOT_COLUMN_WEIGHT),
        retrieval_threshold=max(0.0, settings.CHATBOT_RETRIEVAL_THRESHOLD),
        column_similarity_threshold=max(
            0.0, settings.CHATBOT_COLUMN_SIMILARITY_THRESHOLD
        ),
        max_context_chars=max(2000, settings.CHATBOT_MAX_CONTEXT_CHARS),
        keyword_retries=max(0, settings.CHATBOT_KEYWORD_RETRIES),
        sample_rows_per_table=max(1, settings.CHATBOT_SAMPLE_ROWS_PER_TABLE),
        llm_temperature=_resolve_stage_temperature(
            "CHATBOT_SCHEMA_KEYWORD_LLM_TEMPERATURE"
        ),
        schema_uq_enabled=bool(
            getattr(settings, "CHATBOT_SCHEMA_UQ_ENABLED", False)
        ),
        schema_uq_m_sampling=max(
            2, int(getattr(settings, "CHATBOT_SCHEMA_UQ_M_SAMPLING", 10))
        ),
        schema_uq_t_sampling=max(
            0.0, float(getattr(settings, "CHATBOT_SCHEMA_UQ_T_SAMPLING", 1.3))
        ),
        schema_uq_tau_cluster=max(
            0.0, float(getattr(settings, "CHATBOT_SCHEMA_UQ_TAU_CLUSTER", 0.80))
        ),
        schema_uq_tau_u=max(
            0.0, float(getattr(settings, "CHATBOT_SCHEMA_UQ_TAU_U", 0.12))
        ),
        schema_uq_embedding_model=(
            (
                getattr(settings, "CHATBOT_SCHEMA_UQ_EMBEDDING_MODEL", "") or ""
            ).strip()
            or "text-embedding-3-small"
        ),
        ambiguous_lexicon_enabled=bool(
            getattr(settings, "CHATBOT_AMBIGUOUS_LEXICON_ENABLED", True)
        ),
        auto_include_keys=bool(getattr(settings, "CHATBOT_AUTO_INCLUDE_KEYS", False)),
        display_cols=display_cols,
        key_cols_override=key_cols_override,
        schema_keyword_gates=schema_keyword_gates,
    )
