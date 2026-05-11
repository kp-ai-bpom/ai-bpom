from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SuksesorCreateRequest(BaseModel):
    """Request DTO for creating a new Suksesor."""

    nip: str = Field(..., min_length=1, max_length=20, description="NIP pegawai")
    nama: str = Field(..., min_length=1, max_length=255, description="Nama lengkap")
    unit_kerja: Optional[str] = Field(None, max_length=255, description="Unit kerja")
    grade: Optional[str] = Field(None, max_length=5, description="Grade jabatan")
    kompetensi: Optional[str] = Field(None, description="Kompetensi (JSON format)")
    potensi: Optional[str] = Field(
        None, max_length=20, description="Potensi (High/Medium/Low)"
    )
    readiness: Optional[int] = Field(
        None, ge=0, le=100, description="Readiness level (0-100)"
    )
    is_active: bool = Field(True, description="Status aktif")


class SuksesorUpdateRequest(BaseModel):
    """Request DTO for updating an existing Suksesor."""

    nip: Optional[str] = Field(None, min_length=1, max_length=20)
    nama: Optional[str] = Field(None, min_length=1, max_length=255)
    unit_kerja: Optional[str] = Field(None, max_length=255)
    grade: Optional[str] = Field(None, max_length=5)
    kompetensi: Optional[str] = None
    potensi: Optional[str] = Field(None, max_length=20)
    readiness: Optional[int] = Field(None, ge=0, le=100)
    is_active: Optional[bool] = None


class SuksesorListRequest(BaseModel):
    """Request DTO for listing Suksesor with pagination."""

    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")
    search: Optional[str] = Field(None, description="Search by nama or nip")
    is_active: Optional[bool] = Field(None, description="Filter by active status")


# ── SIASN Candidate Schema ────────────────────────────────────────


class KandidatSIASN(BaseModel):
    """Data kandidat dalam format SIASN (Sistem Informasi Aparatur Sipil Negara)."""

    nip: str = Field(..., description="NIP pegawai")
    nama: str = Field(..., description="Nama singkat pegawai")
    nama_lengkap: str = Field(..., description="Nama lengkap dengan gelar")
    jabatan_nama: str = Field(..., description="Nama jabatan saat ini")
    foto_url: Optional[str] = Field(None, description="URL foto pegawai")
    pool: Optional[int] = Field(None, description="Nomor pool")
    pool_id: Optional[int] = Field(None, description="ID pool")
    fungsi_jabatan: List[str] = Field(default_factory=list, description="Daftar fungsi jabatan")
    riwayat_jabatan: List[str] = Field(default_factory=list, description="Daftar riwayat jabatan")
    jabatan_terakhir: str = Field(..., description="Jabatan terakhir")
    riwayat_pendidikan: List[str] = Field(default_factory=list, description="Daftar riwayat pendidikan")
    nilai_potensi: Optional[float] = Field(None, description="Nilai potensi")
    nilai_mansoskul: Optional[int] = Field(None, description="Nilai manajemen sosial kultural")
    nilai_kinerja: Optional[float] = Field(None, description="Nilai kinerja")
    nilai_kinerja_label: Optional[str] = Field(None, description="Label nilai kinerja")
    masa_kerja: Optional[int] = Field(None, description="Masa kerja dalam tahun")
    masa_kerja_total_tahun: Optional[int] = Field(None, description="Total masa kerja dalam tahun")
    diklat_pim_level: Optional[str] = Field(None, description="Level diklat kepemimpinan")
    jenjang_pendidikan_id: Optional[str] = Field(None, description="ID jenjang pendidikan")
    pengalaman_struktural_tahun: Optional[str] = Field(None, description="Pengalaman struktural dalam tahun")
    current_eselon_id: Optional[str] = Field(None, description="ID eselon saat ini")
    target_eselon_id: Optional[str] = Field(None, description="ID eselon target")
    recommendation_label: Optional[str] = Field(None, description="Label rekomendasi")
    recommendation_type: Optional[str] = Field(None, description="Tipe rekomendasi")
    is_eligible: Optional[bool] = Field(None, description="Status eligible")
    rhk: List[str] = Field(default_factory=list, description="Daftar Rencana Hasil Kerja")

    model_config = {"from_attributes": True}


class SimulasiRequest(BaseModel):
    """Request DTO untuk simulasi pemetaan suksesor."""

    target_jabatan: str = Field(
        ..., description="Jabatan target suksesi, e.g. Inspektur I"
    )
    kandidat: List[KandidatSIASN] = Field(
        ..., min_length=1, max_length=50, description="Daftar kandidat SIASN"
    )


class SaveMatchingRequest(BaseModel):
    """Request DTO untuk menyimpan hasil matching ke riwayat."""

    target_jabatan: str = Field(
        ..., description="Jabatan target suksesi yang digunakan saat matching"
    )
    total_kandidat: int = Field(
        ..., ge=1, description="Total kandidat yang dievaluasi"
    )
    top_kandidat: List[Dict] = Field(
        ..., min_length=1, description="Daftar top kandidat hasil matching"
    )
    sub_tugas: Optional[List[Dict]] = Field(
        None, description="Sub-tugas dekomposisi dari pipeline"
    )
    catatan_reviewer: Optional[str] = Field(
        None, description="Catatan dari reviewer agent"
    )


class IngestRequest(BaseModel):
    document_names: list[str] | None = None
    force_reingest: bool = False