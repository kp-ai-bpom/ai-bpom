from datetime import datetime
from typing import Optional
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
