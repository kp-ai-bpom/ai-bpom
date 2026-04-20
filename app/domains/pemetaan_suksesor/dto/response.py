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
    input_token: str = "0 token"
    output_token: str = "0 token"


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


class RekamJejakItem(BaseModel):
    """Satu entri rekam jejak jabatan kandidat."""

    periode: str
    jabatan: str
    durasi_tahun: int
    deskripsi_tugas_dan_fungsi: str


class SertifikasiItem(BaseModel):
    """Satu entri sertifikasi kandidat."""

    nama_sertifikasi: str
    tahun: int
    keterangan: str


class SkpTahunItem(BaseModel):
    """Data SKP satu tahun."""

    rating_hasil_kerja: str
    rating_perilaku_kerja: str
    keterangan: str


class KandidatCard(BaseModel):
    """Data kandidat untuk daftar pilihan di Step 3."""

    id: str
    nama: str
    jabatan_saat_ini: str
    unit_kerja: str
    box_number: int
    rekam_jejak: List[RekamJejakItem]
    sertifikasi: List[SertifikasiItem]
    skp: Dict[str, SkpTahunItem]
    posisi_nine_box_talenta: str


class KandidatListData(BaseModel):
    """Data response daftar kandidat terfilter berdasarkan box."""

    total: int
    filtered_boxes: List[int]
    kandidat: List[KandidatCard]


class KandidatListResponse(BaseModel):
    """Response wrapper untuk daftar kandidat."""

    message: str
    data: KandidatListData


# ── Matching History Response Schemas ─────────────────────────────


class MatchingHistorySummary(BaseModel):
    """Ringkasan riwayat matching (tanpa JSON blobs)."""

    id: UUID
    target_jabatan: str
    total_kandidat: int
    created_at: datetime

    model_config = {"from_attributes": True}


class MatchingHistoryDetail(BaseModel):
    """Detail lengkap riwayat matching."""

    id: UUID
    target_jabatan: str
    total_kandidat: int
    top_kandidat: List[Dict]
    sub_tugas: Optional[List[Dict]] = None
    catatan_reviewer: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MatchingHistoryListData(BaseModel):
    """Data response daftar riwayat matching dengan paginasi."""

    items: List[MatchingHistorySummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class MatchingHistoryListResponse(BaseModel):
    """Response wrapper untuk daftar riwayat matching."""

    message: str
    data: MatchingHistoryListData


class MatchingHistoryDetailResponse(BaseModel):
    """Response wrapper untuk detail riwayat matching."""

    message: str
    data: MatchingHistoryDetail


class MatchingHistorySaveResponse(BaseModel):
    """Response wrapper untuk simpan riwayat matching."""

    message: str
    data: MatchingHistoryDetail
