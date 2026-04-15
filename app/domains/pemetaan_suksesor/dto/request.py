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


# ── Simulation Request Schemas ──────────────────────────────────


class KandidatProfil(BaseModel):
    """Profil kandidat suksesi."""

    id: str = Field(..., description="ID Kandidat, e.g. KANDIDAT-001")
    nama: str = Field(..., description="Nama lengkap kandidat")
    jabatan_saat_ini: str = Field(..., description="Jabatan saat ini")
    unit_kerja: str = Field(..., description="Unit kerja saat ini")


class RekamJejakEntry(BaseModel):
    """Satu entri rekam jejak karir."""

    periode: str = Field(..., description="Periode, e.g. 2022 - 2026")
    jabatan: str = Field(..., description="Nama jabatan")
    durasi_tahun: int = Field(..., description="Durasi dalam tahun")
    deskripsi_tugas_dan_fungsi: str = Field(
        ..., description="Deskripsi tugas dan fungsi"
    )


class SertifikasiEntry(BaseModel):
    """Satu entri sertifikasi."""

    nama_sertifikasi: str = Field(..., description="Nama sertifikasi atau diklat")
    tahun: int = Field(..., description="Tahun perolehan")
    keterangan: str = Field(..., description="Keterangan tambahan")


class SKPTahun(BaseModel):
    """Penilaian SKP satu tahun."""

    rating_hasil_kerja: str = Field(
        ..., description="Rating hasil kerja, e.g. Di Atas Ekspektasi"
    )
    rating_perilaku_kerja: str = Field(..., description="Rating perilaku kerja")
    keterangan: str = Field(..., description="Catatan penilaian")


class KandidatSuksesi(BaseModel):
    """Data lengkap satu kandidat suksesi — sesuai format input.json."""

    kandidat_suksesi: KandidatProfil
    rekam_jejak: List[RekamJejakEntry]
    sertifikasi: List[SertifikasiEntry]
    skp: Dict[str, SKPTahun] = Field(..., description="Key = tahun, value = SKP")
    posisi_nine_box_talenta: Optional[str] = Field(
        None,
        description="Posisi nine-box, e.g. Kotak 9",
        alias="posisi_nine_box_talenta",
    )

    model_config = {"populate_by_name": True}


class SimulasiRequest(BaseModel):
    """Request DTO untuk simulasi pemetaan suksesor."""

    target_jabatan: str = Field(
        ..., description="Jabatan target suksesi, e.g. Inspektur I"
    )
    kandidat: List[KandidatSuksesi] = Field(
        ..., min_length=1, max_length=50, description="Daftar kandidat"
    )
