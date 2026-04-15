from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class SuksesorDataResponse(BaseModel):
    """Response DTO for Suksesor data."""

    id: UUID
    nip: str
    nama: str
    unit_kerja: Optional[str] = None
    grade: Optional[str] = None
    kompetensi: Optional[str] = None
    potensi: Optional[str] = None
    readiness: Optional[int] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SuksesorResponse(BaseModel):
    """Standard response wrapper for single Suksesor."""

    message: str
    data: SuksesorDataResponse


class SuksesorListDataResponse(BaseModel):
    """Response DTO for list of Suksesor with pagination."""

    items: list[SuksesorDataResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class SuksesorListResponse(BaseModel):
    """Standard response wrapper for list of Suksesor."""

    message: str
    data: SuksesorListDataResponse


class SuksesorDeleteResponse(BaseModel):
    """Response DTO for delete operation."""

    message: str
    data: dict


# ── Simulation Response Schemas ─────────────────────────────────


class DetailEvaluasi(BaseModel):
    """Detail evaluasi per aspek."""

    status: str
    keterangan: str


class KandidatResult(BaseModel):
    """Hasil evaluasi satu kandidat."""

    rank: int
    id_kandidat: str
    nama: str
    jabatan_saat_ini: str
    skor_kesesuaian: float
    kategori_kesiapan: str
    confidence_level: str
    acceptances: int
    kesimpulan: str
    alasan_penilaian: str = ""
    detail_evaluasi: Optional[Dict[str, DetailEvaluasi]] = None


class SimulasiDataResponse(BaseModel):
    """Data response simulasi pemetaan suksesor."""

    target_jabatan: str
    total_kandidat: int
    top_kandidat: List[KandidatResult]
    sub_tugas: Optional[List[Dict]] = None
    catatan_reviewer: Optional[str] = None


class SimulasiResponse(BaseModel):
    """Response wrapper untuk simulasi pemetaan suksesor."""

    message: str
    data: SimulasiDataResponse


class NineBoxItem(BaseModel):
    """Data satu box dalam nine-box talenta grid."""

    box_number: int
    label: str
    kinerja: str
    potensi: str
    selectable: bool
    count: int
    candidates: List[str]


class NineBoxData(BaseModel):
    """Data response nine-box grid."""

    boxes: List[NineBoxItem]


class NineBoxResponse(BaseModel):
    """Response wrapper untuk nine-box grid."""

    message: str
    data: NineBoxData


class KandidatRingkasan(BaseModel):
    """Ringkasan kandidat untuk kartu UI (Step 3)."""

    total_pengalaman_tahun: int
    sertifikasi_top: List[str]
    skp_rating_terbaru: str


class KandidatCard(BaseModel):
    """Data kandidat untuk daftar pilihan di Step 3."""

    id: str
    nama: str
    jabatan_saat_ini: str
    unit_kerja: str
    box_number: int
    ringkasan: KandidatRingkasan


class KandidatListData(BaseModel):
    """Data response daftar kandidat terfilter berdasarkan box."""

    total: int
    filtered_boxes: List[int]
    kandidat: List[KandidatCard]


class KandidatListResponse(BaseModel):
    """Response wrapper untuk daftar kandidat."""

    message: str
    data: KandidatListData
