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


# ── Analysis ────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Input untuk menjalankan Analysis Agent secara mandiri.

    Membutuhkan output dari Planner Agent (blueprint) dan data input asli.
    Biasanya diperoleh dari hasil job /agents/planner.
    """
    blueprint: Any = Field(
        description="Output XAI Blueprint dari Planner Agent (JSON dict)"
    )
    input: Any = Field(
        description="Data input asli: {target, candidates, ...} dari request Planner"
    )


# ── Synthesis ──────────────────────────────────────────────────────────

class SynthesisRequest(BaseModel):
    """Input untuk menjalankan Synthesis Agent secara mandiri.

    Membutuhkan output dari Analysis Agent dan blueprint context dari Planner.
    Biasanya diperoleh dari hasil job /agents/analysis dan /agents/planner.
    """
    xai_justification_report: Any = Field(
        description="Output XAI Justification Report dari Analysis Agent (JSON dict)"
    )
    blueprint_context: Any = Field(
        description="Output XAI Blueprint dari Planner Agent untuk referensi aturan_penilaian (JSON dict)"
    )


# ── Reviewer ────────────────────────────────────────────────────────────

class ReviewerRequest(BaseModel):
    """Input untuk menjalankan Reviewer Agent secara mandiri.

    Membutuhkan output dari ketiga agent sebelumnya.
    Biasanya diperoleh dari hasil job /agents/synthesis, /agents/analysis, /agents/planner.
    """
    synthesis_report: Any = Field(
        description="Output Synthesis Report dari Synthesis Agent (JSON dict)"
    )
    analysis_report: Any = Field(
        description="Output XAI Justification Report dari Analysis Agent (JSON dict)"
    )
    planner_blueprint: Any = Field(
        description="Output XAI Blueprint dari Planner Agent (JSON dict)"
    )
