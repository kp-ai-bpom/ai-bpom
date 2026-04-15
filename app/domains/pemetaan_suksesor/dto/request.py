from typing import Optional

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
