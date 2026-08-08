"""Pydantic schemas for penilaian_makalah."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# REQUEST
# =============================================================================

class EvaluateRequest(BaseModel):
    """Request untuk melakukan penilaian makalah."""

    jabatan: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nama jabatan yang dipilih."
    )

    paper_filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nama file makalah di MinIO."
    )

    tema_filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nama file ketentuan tema."
    )

    query_mode: str = Field(
        default="hybrid",
        description="Mode retrieval LightRAG (hybrid|mix|local|global|naive)."
    )

    m_samples: int = Field(
        default=7,
        ge=3,
        le=15,
        description="Jumlah sampling untuk uncertainty analysis."
    )

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Temperature untuk LLM sampling (0.0-2.0)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "jabatan": "Kepala Pusat Pengembangan SDM POM",
                "paper_filename": "Agus Riyanto_Makalah Seleksi JFT.pdf",
                "tema_filename": "KETENTUAN PENULISAN MAKALAH JPT PRATAMA KEPALA BBPOM_2026.pdf",
                "query_mode": "hybrid",
                "m_samples": 7,
                "temperature": 1.0,
            }
        }


class UploadPaperResponse(BaseModel):

    filename: str

    uploaded: bool

    message: str


class UploadKnowledgeResponse(BaseModel):

    filename: str

    ingested: bool

    message: str


# =============================================================================
# SCORE
# =============================================================================

class ScoreSchema(BaseModel):
    """Penilaian per kriteria (40-100)."""

    n1_kesesuaian_judul: float = Field(ge=40, le=100, description="Score judul (40-100)")
    n2_kesesuaian_isi: float = Field(ge=40, le=100, description="Score isi (40-100)")
    n3_sistematika: float = Field(ge=40, le=100, description="Score sistematika (40-100)")
    n4_ketajaman_analisis: float = Field(ge=40, le=100, description="Score analisis (40-100, bobot 2x)")
    n5_penggunaan_bahasa: float = Field(ge=40, le=100, description="Score bahasa (40-100)")


# =============================================================================
# JUSTIFICATION
# =============================================================================

class JustificationSchema(BaseModel):
    """Justifikasi per kriteria."""

    n1_kesesuaian_judul: str = Field(min_length=1, max_length=500)
    n2_kesesuaian_isi: str = Field(min_length=1, max_length=500)
    n3_sistematika: str = Field(min_length=1, max_length=500)
    n4_ketajaman_analisis: str = Field(min_length=1, max_length=500)
    n5_penggunaan_bahasa: str = Field(min_length=1, max_length=500)


# =============================================================================
# EVIDENCE
# =============================================================================

class EvidenceSchema(BaseModel):
    """Evidence dari makalah per kriteria."""

    n1_kesesuaian_judul: str = Field(min_length=1, max_length=500)
    n2_kesesuaian_isi: str = Field(min_length=1, max_length=500)
    n3_sistematika: str = Field(min_length=1, max_length=500)
    n4_ketajaman_analisis: str = Field(min_length=1, max_length=500)
    n5_penggunaan_bahasa: str = Field(min_length=1, max_length=500)


# =============================================================================
# UNCERTAINTY
# =============================================================================

class CriteriaUncertainty(BaseModel):
    """Uncertainty metrics per kriteria."""

    mean: float = Field(ge=40, le=100, description="Mean score")
    std: float = Field(ge=0, description="Standard deviation")
    nsv: float = Field(ge=0, le=1, description="Normalized Standard Variation")
    status: str = Field(pattern="^(✅|⚠️|❌)", description="Status emoji indicator")
    raw_samples: List[float] = Field(description="Semua sampel score untuk kriteria ini")


class UncertaintySchema(BaseModel):
    """Uncertainty quantification untuk consensus."""

    per_criteria: Dict[str, CriteriaUncertainty] = Field(description="Uncertainty per kriteria")
    weighted_aggregate: float = Field(ge=0, le=1, description="WAU - Weighted Aggregate Uncertainty")
    overall_status: str = Field(description="Status keseluruhan penilaian")
    most_uncertain_criteria: str = Field(description="Kriteria dengan uncertainty tertinggi")


# =============================================================================
# RESPONSE
# =============================================================================

class EvaluateResponse(BaseModel):
    """Response evaluasi makalah."""

    ringkasan: str = Field(min_length=1, max_length=1000, description="Ringkasan hasil evaluasi")
    final_score: float = Field(ge=40, le=100, description="Final score berdasarkan weighted scoring")
    scores: Dict[str, float] = Field(description="Consensus scores per kriteria")
    justification: Dict[str, str] = Field(description="Justifikasi per kriteria")
    evidence: Dict[str, str] = Field(description="Evidence dari makalah per kriteria")
    uncertainty: Dict[str, Any] = Field(description="Uncertainty metrics")
    valid_samples: int = Field(ge=0, description="Jumlah sampel valid yang dihasilkan")
    total_samples: int = Field(ge=1, description="Total sampel yang diminta")

    class Config:
        json_schema_extra = {
            "example": {
                "ringkasan": "Makalah membahas metodologi...",
                "final_score": 75.5,
                "scores": {
                    "n1_kesesuaian_judul": 80,
                    "n2_kesesuaian_isi": 75,
                    "n3_sistematika": 70,
                    "n4_ketajaman_analisis": 75,
                    "n5_penggunaan_bahasa": 76,
                },
                "justification": {"n1_kesesuaian_judul": "Judul mencerminkan isi..."},
                "evidence": {"n1_kesesuaian_judul": "Dilihat di halaman 1..."},
                "uncertainty": {"per_criteria": {}, "weighted_aggregate": 0.08, "overall_status": "✅ YAKIN"},
                "valid_samples": 7,
                "total_samples": 7,
            }
        }


# =============================================================================
# HISTORY
# =============================================================================

class HistoryItem(BaseModel):
    """Item riwayat evaluasi."""

    id: int = Field(ge=1, description="ID evaluasi")
    paper_filename: str = Field(description="Nama file makalah")
    jabatan: str = Field(description="Jabatan yang dievaluasi")
    final_score: float = Field(ge=40, le=100, description="Final score")
    query_mode: str = Field(description="Query mode yang digunakan")
    created_at: datetime = Field(description="Waktu evaluasi")


class HistoryResponse(BaseModel):
    """Response daftar riwayat evaluasi."""

    total: int = Field(ge=0, description="Total evaluasi")
    data: List[HistoryItem] = Field(description="List evaluasi")


# =============================================================================
# EXPORT
# =============================================================================

class ExportResponse(BaseModel):

    metadata: Dict[str, Any]

    consensus_results: Dict[str, Any]

    all_samples: List[Any]

    best_sample_by_consensus: Optional[Dict[str, Any]] = None