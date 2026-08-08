"""Domain settings for penilaian_makalah."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PenilaianMakalahSettings:
    working_dir: str = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
    skj_json_folder: str = os.getenv(
        "SKJ_JSON_FOLDER",
        os.path.join(os.path.dirname(__file__), "..", "dto", "jabatan_rules"),
    )
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "password123")
    bucket_makalah: str = os.getenv("BUCKET_MAKALAH", "makalah")
    bucket_knowledge: str = os.getenv("BUCKET_KNOWLEDGE", "knowledge")
    bucket_riwayat: str = os.getenv(
        "BUCKET_RIWAYAT", "riwayat-penilaian-makalah"
    )
    bucket_tema: str = os.getenv("BUCKET_TEMA", "ketentuan-penulisan-makalah")
    supported_query_modes: tuple[str, ...] = (
        "hybrid",
        "mix",
        "local",
        "global",
        "naive",
    )
    default_query_mode: str = os.getenv("PENILAIAN_QUERY_MODE", "hybrid")


settings = PenilaianMakalahSettings()
