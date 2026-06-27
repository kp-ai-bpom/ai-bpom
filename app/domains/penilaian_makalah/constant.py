"""
Constants for Penilaian Makalah Domain.
"""

from typing import Dict, List

# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

MIN_SCORE: int = 40
MAX_SCORE: int = 100
SCORE_RANGE: int = MAX_SCORE - MIN_SCORE

SCORE_LABELS: Dict[str, str] = {
    "n1_kesesuaian_judul": "Kesesuaian Judul dengan Tema",
    "n2_kesesuaian_isi": "Kesesuaian Isi dengan Judul & Tema",
    "n3_sistematika": "Sistematika Penulisan",
    "n4_ketajaman_analisis": "Ketajaman Analisis",
    "n5_penggunaan_bahasa": "Penggunaan Bahasa",
}

SCORE_WEIGHTS: Dict[str, int] = {
    "n1_kesesuaian_judul": 1,
    "n2_kesesuaian_isi": 1,
    "n3_sistematika": 1,
    "n4_ketajaman_analisis": 2,
    "n5_penggunaan_bahasa": 1,
}

# =============================================================================
# CRITERIA
# =============================================================================

CRITERIA_KEYS: List[str] = [
    "n1_kesesuaian_judul",
    "n2_kesesuaian_isi",
    "n3_sistematika",
    "n4_ketajaman_analisis",
    "n5_penggunaan_bahasa",
]

CRITERIA_SHORT: List[str] = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
]

CRITERIA_LABELS: Dict[str, str] = {
    "n1": "Kesesuaian Judul",
    "n2": "Kesesuaian Isi",
    "n3": "Sistematika",
    "n4": "Ketajaman Analisis",
    "n5": "Penggunaan Bahasa",
}

# =============================================================================
# UNCERTAINTY SAMPLING
# =============================================================================

DEFAULT_M_SAMPLES: int = 7

DEFAULT_TEMPERATURE: float = 1.0

NSV_THRESHOLD: float = 0.10

SAMPLE_RETRY_LIMIT: int = 2

# =============================================================================
# QUERY MODE
# =============================================================================

DEFAULT_QUERY_MODE = "hybrid"

SUPPORTED_QUERY_MODE = [
    "hybrid",
    "mix",
    "local",
    "global",
    "naive",
]

# =============================================================================
# STORAGE BUCKET
# =============================================================================

BUCKET_MAKALAH = "makalah"

BUCKET_KNOWLEDGE = "knowledge"

BUCKET_RIWAYAT = "riwayat-penilaian-makalah"

BUCKET_TEMA = "ketentuan-penulisan-makalah"