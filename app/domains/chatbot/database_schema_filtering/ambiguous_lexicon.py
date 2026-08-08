"""Safeguard deterministik: kamus istilah-ambigu domain BPOM.

PENTING — ini MEKANISME TERPISAH dari metode UQ skripsi (semantic entropy atas
interpretasi skema, lihat ``schema_uq.py``). Tujuannya hanya satu: MENJAMIN
bahwa istilah yang secara domain memang ambigu (mis. "senior", "terbaik",
"terbanyak") SELALU memicu klarifikasi, walau sampler UQ kebetulan konvergen
(model terlalu yakin → 1 cluster). Sampling murni TIDAK bisa menjamin ini.

Untuk sidang: sajikan ini sebagai *rule-based safeguard* pelengkap, BUKAN bagian
dari kuantifikasi ketidakpastian. Verdict UQ tetap dilaporkan apa adanya; lexicon
hanya menambah pemicu klarifikasi deterministik bila UQ tidak memicunya.

Cara menambah istilah: cukup tambahkan entri ke ``AMBIGUOUS_TERMS``.
"""

import re

from ..semantic_disambiguation.types import (
    AmbiguityDetectionResult,
    InterpretationOption,
)


# Setiap entri: kata kunci kanonik → definisi ambiguitas.
#   aliases             : daftar bentuk kata yang memicu (dicocokkan word-boundary,
#                         spasi ditoleransi sebagai \s+).
#   ambiguity_type      : tipe ambiguitas (lihat AMBIGUITY_TYPES di types.py).
#   clarification_question : pertanyaan ramah, bahasa bisnis, tanpa jargon.
#   options             : daftar (label, deskripsi) pilihan interpretasi.
#
# Urutan list menentukan prioritas bila beberapa istilah cocok (yang lebih
# spesifik diletakkan lebih dahulu).
AMBIGUOUS_TERMS: list[dict] = [
    {
        "term": "senior",
        "aliases": ["senior", "paling senior", "tersenior", "lebih senior"],
        "ambiguity_type": "column",
        "clarification_question": (
            "Maksud Anda 'senior' yang seperti apa? Istilah ini bisa diartikan "
            "dengan beberapa cara berbeda."
        ),
        "options": [
            (
                "Masa kerja paling lama",
                "Pegawai yang paling lama bekerja, dihitung dari tanggal mulai "
                "bekerja.",
            ),
            (
                "Pangkat/golongan tertinggi",
                "Pegawai dengan pangkat atau golongan kepegawaian tertinggi.",
            ),
            (
                "Jabatan struktural tertinggi",
                "Pegawai dengan jabatan atau eselon struktural tertinggi.",
            ),
            (
                "Usia paling tua",
                "Pegawai dengan usia paling tua berdasarkan tanggal lahir.",
            ),
        ],
    },
    {
        "term": "junior",
        "aliases": ["junior", "paling junior", "terjunior"],
        "ambiguity_type": "column",
        "clarification_question": (
            "Maksud Anda 'junior' yang seperti apa?"
        ),
        "options": [
            (
                "Masa kerja paling baru",
                "Pegawai yang paling baru bekerja (masa kerja terpendek).",
            ),
            (
                "Pangkat/golongan terendah",
                "Pegawai dengan pangkat atau golongan kepegawaian terendah.",
            ),
            (
                "Usia paling muda",
                "Pegawai dengan usia paling muda berdasarkan tanggal lahir.",
            ),
        ],
    },
    {
        "term": "terbaik",
        "aliases": ["terbaik", "paling baik", "terunggul", "paling unggul"],
        "ambiguity_type": "vagueness",
        "clarification_question": (
            "Ukuran 'terbaik' yang Anda maksud berdasarkan apa?"
        ),
        "options": [
            (
                "Nilai kinerja tertinggi",
                "Berdasarkan hasil penilaian kinerja pegawai tertinggi.",
            ),
            (
                "Pendidikan tertinggi",
                "Berdasarkan jenjang pendidikan terakhir tertinggi.",
            ),
            (
                "Pangkat/jabatan tertinggi",
                "Berdasarkan pangkat atau jabatan tertinggi.",
            ),
        ],
    },
    {
        "term": "terbanyak",
        "aliases": ["terbanyak", "paling banyak", "jumlah terbesar"],
        "ambiguity_type": "scope",
        "clarification_question": (
            "'Terbanyak' dihitung berdasarkan pengelompokan apa?"
        ),
        "options": [
            (
                "Per unit kerja / satker",
                "Menghitung jumlah pegawai pada tiap unit kerja atau satuan "
                "kerja.",
            ),
            (
                "Per jabatan",
                "Menghitung jumlah pegawai pada tiap jabatan.",
            ),
            (
                "Per pangkat/golongan",
                "Menghitung jumlah pegawai pada tiap pangkat atau golongan.",
            ),
        ],
    },
]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # Spasi pada alias multi-kata ditoleransi sebagai 1+ whitespace.
    escaped = r"\s+".join(re.escape(part) for part in alias.split())
    return re.compile(rf"(?<![0-9a-z]){escaped}(?![0-9a-z])", re.IGNORECASE)


def match_ambiguous_term(query: str) -> dict | None:
    """Kembalikan entri lexicon pertama yang cocok di ``query`` (atau None).

    Pencocokan word-boundary, case-insensitive, mendukung alias multi-kata.
    """
    text = (query or "").strip().lower()
    if not text:
        return None
    for entry in AMBIGUOUS_TERMS:
        for alias in entry["aliases"]:
            if _alias_pattern(alias).search(text):
                return entry
    return None


def build_clarification(entry: dict) -> AmbiguityDetectionResult:
    """Susun ``AmbiguityDetectionResult`` deterministik dari entri lexicon.

    Field telemetri UQ (h_norm, τ_U, ...) sengaja DIBIARKAN None — klarifikasi
    ini BUKAN berasal dari kuantifikasi ketidakpastian, melainkan safeguard
    rule-based. Membiarkannya None menjaga trace tetap jujur.
    """
    options = [
        InterpretationOption(label=label, description=description)
        for label, description in entry["options"]
    ]
    return AmbiguityDetectionResult(
        is_ambiguous=True,
        ambiguity_type=entry.get("ambiguity_type") or "vagueness",
        clarification_question=entry["clarification_question"],
        interpretation_options=options,
    )


__all__ = [
    "AMBIGUOUS_TERMS",
    "match_ambiguous_term",
    "build_clarification",
]
