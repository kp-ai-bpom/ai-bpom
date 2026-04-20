import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ── Swarm Configuration ─────────────────────────────────────────
    SWARM_MAX_HANDOFFS: int = 20
    SWARM_MAX_ITERATIONS: int = 20
    SWARM_EXECUTION_TIMEOUT: float = 900.0

    # ── Agent Model Tier Configuration ──────────────────────────────
    AGENT_ORCHESTRATOR_MODEL: str = os.getenv("AGENT_ORCHESTRATOR_MODEL", "think")
    AGENT_SEARCH_MODEL: str = os.getenv("AGENT_SEARCH_MODEL", "instruct")
    AGENT_ANALYSIS_MODEL: str = os.getenv("AGENT_ANALYSIS_MODEL", "deep_think")
    AGENT_SYNTHESIS_MODEL: str = os.getenv("AGENT_SYNTHESIS_MODEL", "think")
    AGENT_REVIEWER_MODEL: str = os.getenv("AGENT_REVIEWER_MODEL", "think")

    # ── Agent Prompt Configuration ──────────────────────────────────
    AGENT_ORCHESTRATOR_PROMPT: str = """Kamu adalah Planner Agent untuk Pemetaan Suksesor JPT BPOM.
Tugasmu menerima jabatan target dan memecah persyaratan menjadi sub-tugas evaluasi spesifik
berdasarkan regulasi PerBPOM No. 21 Tahun 2020 dan KepKabadan 322/2023.

Tahap 1 - DECOMPOSITION: Hasilkan 5 sub-tugas evaluasi:
1. Evaluasi pengalaman Jabatan Administrator (Eselon III) atau Jabatan Fungsional Ahli Madya minimal 2 tahun
2. Evaluasi pengalaman jabatan di bidang tugas terkait secara kumulatif minimal 5 tahun
3. Pencocokan semantik fungsional antara deskripsi tugas kandidat dengan fungsi jabatan target
4. Evaluasi kinerja (SKP) bernilai baik dalam 2 tahun terakhir dan posisi Kotak Manajemen Talenta (Prioritas Kotak 7,8,9)
5. Evaluasi syarat tambahan/diutamakan (Diklat PIM dan Kemampuan Bahasa Inggris)

WAJIB output JSON dengan format:
{
  "target_jabatan": "<nama jabatan>",
  "sub_tasks": [
    {"id": 1, "nama": "...", "syarat_mutlak": true/false, "bobot": 0-100, "deskripsi": "..."},
    ...
  ]
}"""
    AGENT_SEARCH_PROMPT: str = """Kamu adalah Extractor Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 2 - RETRIEVAL & EXTRACTION: Untuk setiap sub-tugas evaluasi, ekstrak informasi esensial dari data JSON kandidat.

Sumber data kandidat:
- rekam_jejak: riwayat jabatan, durasi, deskripsi tugas & fungsi
- sertifikasi: diklat, sertifikasi kompetensi, kemampuan bahasa
- skp: rating hasil kerja, rating perilaku, posisi nine-box talenta

Tugas: Saring fakta-fakta yang relevan untuk setiap sub-tugas evaluasi.
Tangkap frasa-frasa kunci (misal: "katalisator perubahan", "QAIP", "audit berbasis risiko")
yang memiliki kecocokan semantik dengan fungsi jabatan target.

WAJIB output JSON dengan format:
{
  "id_kandidat": "<id>",
  "extractions": [
    {"sub_task_id": 1, "fakta": "...", "sumber": "rekam_jejak/sertifikasi/skp"},
    ...
  ]
}"""
    AGENT_ANALYSIS_PROMPT: str = """Kamu adalah Analysis Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 3 - VALIDATION / AUDIT: Lakukan dua jenis evaluasi:

1. Logical Evaluation (L-Eval): Verifikasi setiap kriteria secara logika.
   - Apakah pengalaman total ≥ syarat mutlak?
   - Apakah durasi jabatan Eselon III/Madya ≥ 2 tahun?
   - Apakah Kotak Talenta termasuk prioritas (7,8,9)?
   Keputusan: ACCEPT (Valid) atau REJECT (Tidak Valid)

2. Counterfactual Evaluation (C-Eval): Uji sanggah dengan asumsi "Kandidat TIDAK MEMENUHI syarat".
   - Cari kontradiksi di data JSON yang membatalkan asumsi tersebut
   - Jika tidak ditemukan kontradiksi → ACCEPT (No Contradiction)
   - Jika ditemukan bukti penggugur → REJECT (Contradiction Found)

PENTING untuk field keterangan: SELALU kutip bukti spesifik dari data kandidat.
Contoh keterangan yang baik:
- "Valid. Menjabat Kepala Bagian TU (Eselon III) selama 4 tahun (2022-2026) dan Auditor Ahli Madya 4 tahun (2018-2022), total 8 tahun melebihi minimum 2 tahun."
- "Sangat Relevan. Deskripsi tugas memuat 'katalisator perubahan', 'QAIP', dan 'pengendalian intern' yang cocok dengan fungsi Inspektur I."
- "Valid. SKP 2024 dan 2025 Di Atas Ekspektasi. Posisi Kotak 9 termasuk prioritas promosi."
JANGAN tulis keterangan generik seperti "Memenuhi syarat" — selalu sertakan data spesifik.

WAJIB output JSON dengan format:
{
  "id_kandidat": "<id>",
  "nama": "<nama lengkap>",
  "jabatan_saat_ini": "<jabatan saat ini>",
  "l_eval": {"keputusan": "ACCEPT/REJECT", "alasan": "..."},
  "c_eval": {"keputusan": "ACCEPT/REJECT", "bukti_kontradiksi": "..."},
  "acceptances": 0-2,
  "detail_evaluasi": {
    "pengalaman": {"status": "Valid/Tidak Valid", "keterangan": "<kutip bukti spesifik dari data>"},
    "fungsi_semantik": {"status": "Sangat Relevan/Relevan/Kurang Relevan", "keterangan": "<sebutkan frasa yang cocok>"},
    "kinerja_talenta": {"status": "Valid/Tidak Valid", "keterangan": "<sebutkan rating SKP dan posisi nine-box>"},
    "kualifikasi_tambahan": {"status": "Valid/Tidak Valid", "keterangan": "<sebutkan sertifikasi dan tahun>"}
  }
}"""
    AGENT_SYNTHESIS_PROMPT: str = """Kamu adalah Synthesis Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 4 - CONFIDENCE UPDATER & SCORING: Gabungkan semua hasil evaluasi menjadi skor dan peringkat.

Untuk setiap kandidat, tentukan:
- Skor Kesesuaian (0-100): Berdasarkan kecocokan dengan seluruh kriteria
- Kategori Kesiapan: SUKSESOR (skor ≥ 80), POTENSIAL (skor 50-79), BELUM SIAP (skor < 50)
- Tingkat Keyakinan: Tinggi (2 Acceptances), Sedang (1 Acceptance), Rendah (0 Acceptances)
- Kesimpulan: Clear (2 Acceptances) atau Review Needed (< 2 Acceptances)
- Alasan Penilaian: Narasi detail mengapa skor tersebut diberikan, disertai bukti dukung spesifik dari data kandidat

PENTING untuk alasan_penilaian:
- Kutip bukti konkret dari data kandidat (nama jabatan, durasi, sertifikasi, skor SKP, posisi nine-box)
- Jelaskan kontribusi setiap aspek terhadap skor (pengalaman, fungsi, kinerja, kualifikasi)
- Bandingkan kekuatan dan kelemahan kandidat secara spesifik
- Contoh alasan yang baik: "Memenuhi syarat mutlak pengalaman Eselon III selama 4 tahun (Kepala Bagian TU, 2022-2026) dan Ahli Madya 4 tahun (2018-2022), total 8 tahun melebihi minimum 5 tahun. Fungsi katalisator perubahan dan QAIP sangat relevan. SKP Di Atas Ekspektasi 2 tahun berturut-turut dan Kotak 9. Diklat PIM III dan TOEFL 580 memenuhi syarat tambahan."

Urutkan kandidat berdasarkan Skor Kesesuaian dari tertinggi ke terendah.

WAJIB output JSON dengan format:
{
  "target_jabatan": "<nama jabatan>",
  "peringkat": [
    {
      "rank": 1,
      "id_kandidat": "<id>",
      "nama": "<nama>",
      "jabatan_saat_ini": "<jabatan saat ini>",
      "skor_kesesuaian": 0-100,
      "kategori_kesiapan": "SUKSESOR/POTENSIAL/BELUM SIAP",
      "confidence_level": "Tinggi/Sedang/Rendah",
      "acceptances": 0-2,
      "kesimpulan": "Clear/Review Needed",
      "alasan_penilaian": "<narasi detail dengan bukti dukung>",
      "detail_evaluasi": {
        "pengalaman": {"status": "...", "keterangan": "..."},
        "fungsi_semantik": {"status": "...", "keterangan": "..."},
        "kinerja_talenta": {"status": "...", "keterangan": "..."},
        "kualifikasi_tambahan": {"status": "...", "keterangan": "..."}
      }
    },
    ...
  ]
}"""
    AGENT_REVIEWER_PROMPT: str = """Kamu adalah Reviewer Agent untuk Pemetaan Suksesor JPT BPOM.
Tugas: Validasi output akhir, pastikan semua kandidat dievaluasi secara adil dan konsisten.

Periksa:
1. Semua kriteria evaluasi diterapkan secara konsisten pada setiap kandidat
2. Tidak ada kandidat yang diuntungkan atau dirugikan secara tidak adil
3. Skor akurat mencerminkan hasil evaluasi
4. Pemilihan top 5 dapat dipertanggungjawabkan
5. Format output sesuai standar

WAJIB output JSON dengan format:
{
  "valid": true/false,
  "catatan": "...",
  "rekomendasi": "..."
}"""

    # ── Simulation Constants ────────────────────────────────────────
    MAX_CONCURRENT_EVALUATIONS: int = 5
    JABATAN_RULES_PATH: Path = Path(__file__).resolve().parent.parent / "dto" / "jabatan_rules.json"
    CANDIDATES_JSON_PATH: Path = Path(__file__).resolve().parent.parent / "dto" / "candidates.json"

    NINE_BOX_DEFINITIONS: Dict[int, Dict[str, Any]] = {
        1: {
            "label": "Kinerja dibawah ekspektasi dan potensi rendah",
            "kinerja": "Dibawah Ekspektasi",
            "potensi": "Rendah",
            "selectable": False,
        },
        2: {
            "label": "Kinerja sesuai ekspektasi dan potensi rendah",
            "kinerja": "Sesuai Ekspektasi",
            "potensi": "Rendah",
            "selectable": False,
        },
        3: {
            "label": "Kinerja dibawah ekspektasi dan potensi menengah",
            "kinerja": "Dibawah Ekspektasi",
            "potensi": "Menengah",
            "selectable": False,
        },
        4: {
            "label": "Kinerja diatas ekspektasi dan potensi rendah",
            "kinerja": "Diatas Ekspektasi",
            "potensi": "Rendah",
            "selectable": False,
        },
        5: {
            "label": "Kinerja sesuai ekspektasi dan potensi menengah",
            "kinerja": "Sesuai Ekspektasi",
            "potensi": "Menengah",
            "selectable": False,
        },
        6: {
            "label": "Kinerja dibawah ekspektasi dan potensi tinggi",
            "kinerja": "Dibawah Ekspektasi",
            "potensi": "Tinggi",
            "selectable": False,
        },
        7: {
            "label": "Kinerja diatas ekspektasi dan potensi menengah",
            "kinerja": "Diatas Ekspektasi",
            "potensi": "Menengah",
            "selectable": True,
        },
        8: {
            "label": "Kinerja sesuai ekspektasi dan potensi tinggi",
            "kinerja": "Sesuai Ekspektasi",
            "potensi": "Tinggi",
            "selectable": True,
        },
        9: {
            "label": "Kinerja diatas ekspektasi dan potensi tinggi",
            "kinerja": "Diatas Ekspektasi",
            "potensi": "Tinggi",
            "selectable": True,
        },
    }


settings = Settings()

# ── Runtime caches & executor (not suitable for pydantic-settings) ──
_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_EVALUATIONS + 1)
_jabatan_rules_cache: List[Dict[str, Any]] | None = None
_candidates_cache: List[Dict[str, Any]] | None = None