from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from .dto.request import (
    SuksesorCreateRequest,
    SuksesorUpdateRequest,
)
from .dto.response import (
    SuksesorDeleteResponse,
    SuksesorListResponse,
    SuksesorResponse,
)
from .services import SuksesorService, get_suksesor_service

router = APIRouter()


@router.post(
    "/",
    response_model=SuksesorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Suksesor",
    description="Create a new Suksesor (calon penerus jabatan) entry.",
)
async def create_suksesor(
    data: SuksesorCreateRequest,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Create a new Suksesor."""
    return await service.create(data)


@router.get(
    "/",
    response_model=SuksesorListResponse,
    summary="Get list of Suksesor",
    description="Get paginated list of Suksesor with optional filters.",
)
async def list_suksesor(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by nama or nip"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorListResponse:
    """Get paginated list of Suksesor."""
    return await service.get_list(
        page=page, page_size=page_size, search=search, is_active=is_active
    )


@router.get(
    "/nip/{nip}",
    response_model=SuksesorResponse,
    summary="Get Suksesor by NIP",
    description="Get a specific Suksesor by their NIP (Nomor Induk Pegawai).",
)
async def get_suksesor_by_nip(
    nip: str,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Get a Suksesor by NIP."""
    return await service.get_by_nip(nip)


@router.get(
    "/{suksesor_id}",
    response_model=SuksesorResponse,
    summary="Get Suksesor by ID",
    description="Get a specific Suksesor by their UUID.",
)
async def get_suksesor_by_id(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Get a Suksesor by ID."""
    return await service.get_by_id(suksesor_id)


@router.put(
    "/{suksesor_id}",
    response_model=SuksesorResponse,
    summary="Update a Suksesor",
    description="Update an existing Suksesor by ID.",
)
async def update_suksesor(
    suksesor_id: UUID,
    data: SuksesorUpdateRequest,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Update a Suksesor."""
    return await service.update(suksesor_id, data)


@router.delete(
    "/{suksesor_id}",
    response_model=SuksesorDeleteResponse,
    summary="Delete a Suksesor",
    description="Delete a Suksesor by ID (hard delete).",
)
async def delete_suksesor(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorDeleteResponse:
    """Delete a Suksesor."""
    return await service.delete(suksesor_id)


@router.patch(
    "/{suksesor_id}/deactivate",
    response_model=SuksesorResponse,
    summary="Soft delete a Suksesor",
    description="Deactivate a Suksesor by setting is_active to False.",
)
async def deactivate_suksesor(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Soft delete (deactivate) a Suksesor."""
    return await service.soft_delete(suksesor_id)
