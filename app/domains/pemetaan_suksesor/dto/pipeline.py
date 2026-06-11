from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    PLANNER = "planner"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REVIEWER = "reviewer"


# ── Generic (untuk evaluate endpoint) ────────────────────────────────

class AgentEvaluateRequest(BaseModel):
    agent_output: Any
    input_json: Any
    tool_call_log: Optional[str] = None


# ── Planner ───────────────────────────────────────────────────────────

class CandidateInput(BaseModel):
    """Data kandidat dari SIASN — pass-through verbatim ke Planner Agent."""

    model_config = {"extra": "allow"}

    nip: str
    nama: str
    nama_lengkap: str
    jabatan_nama: str
    jabatan_terakhir: str
    fungsi_jabatan: list[str]
    riwayat_jabatan: list[str]
    riwayat_pendidikan: list[str]
    nilai_potensi: float
    nilai_mansoskul: int
    nilai_kinerja: int
    nilai_kinerja_label: str
    masa_kerja: int
    masa_kerja_total_tahun: Optional[int] = None
    pengalaman_struktural_tahun: str
    diklat_pim_level: Optional[str] = None
    jenjang_pendidikan_id: str
    current_eselon_id: Optional[str] = None
    target_eselon_id: str
    recommendation_label: str
    recommendation_type: str
    is_eligible: bool
    rhk: list[str]
    pool: Optional[int] = None
    pool_id: Optional[int] = None
    foto_url: Optional[str] = None


class PlannerRequest(BaseModel):
    target: str = Field(
        description="Nama jabatan target suksesi (contoh: 'Kepala Balai Besar POM')"
    )
    candidates: list[CandidateInput] = Field(
        description="Daftar kandidat yang akan dievaluasi"
    )
