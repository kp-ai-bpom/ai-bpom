import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import settings


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
}


@dataclass(frozen=True)
class SemanticMemoryConfig:
    allowed_tables: dict[str, list[str]]
    vector_table: str
    sql_timeout_ms: int
    top_k_per_keyword: int
    max_retrieved_tables: int
    table_weight: float
    column_weight: float
    retrieval_threshold: float
    column_similarity_threshold: float
    max_context_chars: int
    keyword_retries: int
    sample_rows_per_table: int



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

    return SemanticMemoryConfig(
        allowed_tables=allowed_tables,
        vector_table=settings.CHATBOT_VECTOR_TABLE,
        sql_timeout_ms=max(1000, settings.CHATBOT_SQL_TIMEOUT_MS),
        top_k_per_keyword=max(1, settings.CHATBOT_TOP_K_PER_KEYWORD),
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
    )
