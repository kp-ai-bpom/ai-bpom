"""Konfigurasi spesifik domain CHATBOT.

Semua setting di sini HANYA dipakai oleh ``app/domains/chatbot/`` — pisahkan
dari ``app/core/config.py`` (yang sekarang hanya berisi setting shared lintas
domain) supaya developer domain lain (``pemetaan_suksesor``,
``penilaian_suksesor``) tidak ikut terbeban perubahan setting chatbot.

Pola ini meniru ``app/domains/pemetaan_suksesor/core/config.py`` yang sudah
lebih dulu memakai konvensi config per-domain. Nama env var TIDAK berubah
(masih ``CHATBOT_*``) supaya kompatibel dengan ``eval/sweep_config.py`` dan
deployment lama.

Resolver temperature per-stage (``_resolve_stage_temperature``) ikut pindah ke
sini karena hanya dipakai oleh sub-config chatbot (rewrite, ambiguity,
schema-filtering, sql-validation).
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class ChatbotSettings(BaseSettings):
    # Default temperature untuk LLM yang dipanggil dari pipeline chatbot.
    # Dipakai sebagai fallback oleh ``_resolve_stage_temperature`` ketika env
    # per-stage (mis. ``CHATBOT_REWRITE_LLM_TEMPERATURE``) tidak diset.
    # Hierarki resolusi:
    #   1. Env var per-stage (mis. ``CHATBOT_AMBIGUITY_LLM_TEMPERATURE``).
    #   2. ``legacy_default`` dari caller (mis. ``CHATBOT_AMBIGUITY_TEMPERATURE``).
    #   3. ``CHATBOT_LLM_TEMPERATURE`` (default 0.7).
    CHATBOT_LLM_TEMPERATURE: float = float(
        os.getenv("CHATBOT_LLM_TEMPERATURE", "0.7")
    )

    # ── Semantic Memory Pipeline (Tahap 2 — schema filtering) ───────────────
    CHATBOT_VECTOR_TABLE: str = os.getenv("CHATBOT_VECTOR_TABLE", "knowledge_entities")
    CHATBOT_SQL_TIMEOUT_MS: int = int(os.getenv("CHATBOT_SQL_TIMEOUT_MS", "8000"))
    CHATBOT_TOP_K_PER_KEYWORD: int = int(os.getenv("CHATBOT_TOP_K_PER_KEYWORD", "15"))
    CHATBOT_MAX_RETRIEVED_TABLES: int = int(os.getenv("CHATBOT_MAX_RETRIEVED_TABLES", "5"))
    CHATBOT_TABLE_WEIGHT: float = float(os.getenv("CHATBOT_TABLE_WEIGHT", "1.5"))
    CHATBOT_COLUMN_WEIGHT: float = float(os.getenv("CHATBOT_COLUMN_WEIGHT", "1.0"))
    CHATBOT_RETRIEVAL_THRESHOLD: float = float(
        os.getenv("CHATBOT_RETRIEVAL_THRESHOLD", "0.3")
    )
    CHATBOT_COLUMN_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_COLUMN_SIMILARITY_THRESHOLD", "0.25")
    )
    CHATBOT_MAX_CONTEXT_CHARS: int = int(os.getenv("CHATBOT_MAX_CONTEXT_CHARS", "40000"))
    CHATBOT_SQL_GENERATION_RETRIES: int = int(
        os.getenv("CHATBOT_SQL_GENERATION_RETRIES", "4")
    )
    CHATBOT_SQL_GENERATOR_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_SQL_GENERATOR_MAX_TOKENS", "3072")
    )
    CHATBOT_SQL_DEFAULT_SCHEMA: str = os.getenv(
        "CHATBOT_SQL_DEFAULT_SCHEMA", "public"
    )
    CHATBOT_KEYWORD_RETRIES: int = int(os.getenv("CHATBOT_KEYWORD_RETRIES", "3"))
    CHATBOT_SAMPLE_ROWS_PER_TABLE: int = int(
        os.getenv("CHATBOT_SAMPLE_ROWS_PER_TABLE", "3")
    )
    CHATBOT_ALLOWED_TABLES_JSON: str = os.getenv(
        "CHATBOT_ALLOWED_TABLES_JSON",
        '{"public":["propinsi_tm","kabupaten_tm","kecamatan_tm","pangkat_tm","tipepegawai_tm","eselon_tm","pegawai_tm","disabilitas_tm","riwayatjabatan_th","SIAP_SATKER_TOP","jabatan_tm","sk_pegawai_v"],"siap":["R_FUNGSI","T_RIWAYAT_MUTASI","V_PENDIDIKAN_TERAKHIR"],"mantel":["period_employees","periods"]}',
    )

    # ── Question Rewriting (Tahap 1) ────────────────────────────────────────
    CHATBOT_REWRITE_ENABLED: bool = (
        os.getenv("CHATBOT_REWRITE_ENABLED", "true").lower() == "true"
    )
    CHATBOT_REWRITE_WORKING_MEMORY_WINDOW: int = int(
        os.getenv("CHATBOT_REWRITE_WORKING_MEMORY_WINDOW", "4")
    )
    CHATBOT_REWRITE_MAX_EPISODIC_MATCHES: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_EPISODIC_MATCHES", "3")
    )
    CHATBOT_REWRITE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_REWRITE_SIMILARITY_THRESHOLD", "0.3")
    )
    CHATBOT_REWRITE_MAX_EPISODIC_SNIPPET_CHARS: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_EPISODIC_SNIPPET_CHARS", "1500")
    )
    CHATBOT_REWRITE_MAX_WORKING_SNIPPET_CHARS: int = int(
        os.getenv("CHATBOT_REWRITE_MAX_WORKING_SNIPPET_CHARS", "1000")
    )
    CHATBOT_REWRITE_LLM_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_REWRITE_LLM_MAX_TOKENS", "1200")
    )
    CHATBOT_REWRITE_SOURCE: str = os.getenv(
        "CHATBOT_REWRITE_SOURCE", "chatbot_api"
    )
    CHATBOT_REWRITE_WM_EMBEDDING_FILTER_ENABLED: bool = (
        os.getenv("CHATBOT_REWRITE_WM_EMBEDDING_FILTER_ENABLED", "true").lower() == "true"
    )
    CHATBOT_REWRITE_WM_EMBEDDING_THRESHOLD: float = float(
        os.getenv("CHATBOT_REWRITE_WM_EMBEDDING_THRESHOLD", "0.3")
    )

    # ── Stage 1 UQ (Question Contextual Rewriting Uncertainty) ──────────────
    # FROZEN dari kalibrasi tests/stage1/scripts/kalibrasi_stage1.py.
    # Production HARUS mirror kalibrasi: prompt identik, temperature identik,
    # max_tokens identik, embedding model identik. Setiap perubahan di sini
    # meng-invalidate threshold τ_U dan wajib re-kalibrasi.
    CHATBOT_REWRITE_UQ_ENABLED: bool = (
        os.getenv("CHATBOT_REWRITE_UQ_ENABLED", "true").lower() == "true"
    )
    CHATBOT_REWRITE_UQ_M_SAMPLING: int = int(
        os.getenv("CHATBOT_REWRITE_UQ_M_SAMPLING", "10")
    )
    CHATBOT_REWRITE_UQ_T_SAMPLING: float = float(
        os.getenv("CHATBOT_REWRITE_UQ_T_SAMPLING", "1.0")
    )
    CHATBOT_REWRITE_UQ_TAU_CLUSTER: float = float(
        os.getenv("CHATBOT_REWRITE_UQ_TAU_CLUSTER", "0.80")
    )
    CHATBOT_REWRITE_UQ_TAU_U: float = float(
        os.getenv("CHATBOT_REWRITE_UQ_TAU_U", "0.40")
    )
    CHATBOT_REWRITE_UQ_SAMPLE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_REWRITE_UQ_SAMPLE_MAX_TOKENS", "1200")
    )

    # ── Ambiguity Handling (Tahap 3) ────────────────────────────────────────
    CHATBOT_AMBIGUITY_ENABLED: bool = (
        os.getenv("CHATBOT_AMBIGUITY_ENABLED", "true").lower() == "true"
    )
    CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_DETECTION_MAX_TOKENS", "1024")
    )
    CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_REFINE_MAX_TOKENS", "512")
    )
    CHATBOT_AMBIGUITY_TEMPERATURE: float = float(
        os.getenv("CHATBOT_AMBIGUITY_TEMPERATURE", "0.1")
    )
    CHATBOT_AMBIGUITY_TIMEOUT_SECONDS: float = float(
        os.getenv("CHATBOT_AMBIGUITY_TIMEOUT_SECONDS", "10")
    )
    CHATBOT_AMBIGUITY_RATE_LIMIT_RETRIES: int = int(
        os.getenv("CHATBOT_AMBIGUITY_RATE_LIMIT_RETRIES", "3")
    )
    CHATBOT_AMBIGUITY_SESSION_TTL_SECONDS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_SESSION_TTL_SECONDS", "300")
    )
    CHATBOT_AMBIGUITY_AUTO_RESOLVE_WINDOW: int = int(
        os.getenv("CHATBOT_AMBIGUITY_AUTO_RESOLVE_WINDOW", "3")
    )
    CHATBOT_AMBIGUITY_MAX_HISTORY_TURNS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_MAX_HISTORY_TURNS", "3")
    )
    CHATBOT_USE_TOON: bool = (
        os.getenv("CHATBOT_USE_TOON", "true").lower() == "true"
    )

    # ── Stage 1 UQ (Ambiguity) — FROZEN hasil kalibrasi 60 testcase ─────────
    # ROC AUC=0.8928, F1=0.873. JANGAN re-tune tanpa re-kalibrasi end-to-end.
    CHATBOT_AMBIGUITY_UQ_ENABLED: bool = (
        os.getenv("CHATBOT_AMBIGUITY_UQ_ENABLED", "true").lower() == "true"
    )
    CHATBOT_AMBIGUITY_UQ_M_SAMPLING: int = int(
        os.getenv("CHATBOT_AMBIGUITY_UQ_M_SAMPLING", "10")
    )
    CHATBOT_AMBIGUITY_UQ_T_SAMPLING: float = float(
        os.getenv("CHATBOT_AMBIGUITY_UQ_T_SAMPLING", "1.0")
    )
    CHATBOT_AMBIGUITY_UQ_TAU_CLUSTER: float = float(
        os.getenv("CHATBOT_AMBIGUITY_UQ_TAU_CLUSTER", "0.80")
    )
    CHATBOT_AMBIGUITY_UQ_TAU_U: float = float(
        os.getenv("CHATBOT_AMBIGUITY_UQ_TAU_U", "0.40")
    )
    CHATBOT_AMBIGUITY_UQ_SAMPLE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_AMBIGUITY_UQ_SAMPLE_MAX_TOKENS", "512")
    )
    CHATBOT_AMBIGUITY_EMBEDDING_MODEL: str = os.getenv(
        "CHATBOT_AMBIGUITY_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── Procedural Memory ───────────────────────────────────────────────────
    CHATBOT_PROCEDURAL_ENABLED: bool = (
        os.getenv("CHATBOT_PROCEDURAL_ENABLED", "true").lower() == "true"
    )
    CHATBOT_PROCEDURAL_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CHATBOT_PROCEDURAL_SIMILARITY_THRESHOLD", "0.85")
    )
    CHATBOT_PROCEDURAL_TTL_DAYS: int = int(
        os.getenv("CHATBOT_PROCEDURAL_TTL_DAYS", "90")
    )
    CHATBOT_PROCEDURAL_RESET_KEYWORDS: str = os.getenv(
        "CHATBOT_PROCEDURAL_RESET_KEYWORDS",
        "reset preferensi,lupakan preferensi,lupakan aturan,hapus preferensi",
    )

    # ── SQL Validation Pipeline (Tahap 5) ───────────────────────────────────
    CHATBOT_VALIDATION_LEVEL: str = os.getenv("CHATBOT_VALIDATION_LEVEL", "full")
    CHATBOT_VALIDATION_MAX_EXECUTION_ITERATIONS: int = int(
        os.getenv("CHATBOT_VALIDATION_MAX_EXECUTION_ITERATIONS", "3")
    )
    CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS: int = int(
        os.getenv("CHATBOT_VALIDATION_MAX_SEMANTIC_ITERATIONS", "1")
    )
    CHATBOT_VALIDATION_TIMEOUT_SECONDS: float = float(
        os.getenv("CHATBOT_VALIDATION_TIMEOUT_SECONDS", "30")
    )
    CHATBOT_VALIDATION_TRIVIAL_SKIP: bool = (
        os.getenv("CHATBOT_VALIDATION_TRIVIAL_SKIP", "true").lower() == "true"
    )
    CHATBOT_VALIDATION_REFINER_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_VALIDATION_REFINER_MAX_TOKENS", "2048")
    )
    CHATBOT_VALIDATION_JUDGE_MAX_TOKENS: int = int(
        os.getenv("CHATBOT_VALIDATION_JUDGE_MAX_TOKENS", "1536")
    )
    CHATBOT_VALIDATION_EXECUTION_TIMEOUT_MS: int = int(
        os.getenv("CHATBOT_VALIDATION_EXECUTION_TIMEOUT_MS", "5000")
    )


chatbot_settings = ChatbotSettings()


def _clamp_temperature(value: float) -> float:
    """Clamp temperature ke rentang valid provider [0.0, 2.0]."""
    return max(0.0, min(2.0, float(value)))


def _resolve_stage_temperature(
    stage_env: str,
    *,
    legacy_default: Optional[float] = None,
) -> float:
    """Resolve temperature LLM untuk satu tahap pipeline chatbot.

    Hierarki:
      1. ``os.getenv(stage_env)`` jika diset dan parse-able.
      2. ``legacy_default`` jika disediakan caller.
      3. ``chatbot_settings.CHATBOT_LLM_TEMPERATURE`` (default 0.7).

    Hasil di-clamp ke ``[0.0, 2.0]`` mengikuti rentang valid OpenAI/Anthropic.
    Parsing nilai env yang invalid (mis. ``"abc"``) jatuh ke ``legacy_default``
    lalu ke default global, bukan crash startup.
    """
    raw = os.getenv(stage_env)
    if raw is not None and raw.strip() != "":
        try:
            return _clamp_temperature(float(raw))
        except ValueError:
            pass
    if legacy_default is not None:
        return _clamp_temperature(legacy_default)
    return _clamp_temperature(chatbot_settings.CHATBOT_LLM_TEMPERATURE)
