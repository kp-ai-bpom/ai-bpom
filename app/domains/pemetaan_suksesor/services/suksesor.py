import math
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db

from ..dto.request import SuksesorCreateRequest, SuksesorUpdateRequest
from ..dto.response import (
    SuksesorDataResponse,
    SuksesorDeleteResponse,
    SuksesorListDataResponse,
    SuksesorListResponse,
    SuksesorResponse,
)
from ..repositories import SuksesorRepository


class SuksesorService:
    """Service for Suksesor business logic."""

    def __init__(self, repository: SuksesorRepository):
        self._repo = repository

    async def create(self, data: SuksesorCreateRequest) -> SuksesorResponse:
        """Create a new Suksesor."""
        if await self._repo.exists_by_nip(data.nip):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Suksesor with NIP '{data.nip}' already exists",
            )

        suksesor = await self._repo.create(data)
        return SuksesorResponse(
            message="Suksesor created successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_by_id(self, suksesor_id: UUID) -> SuksesorResponse:
        """Get a Suksesor by ID."""
        suksesor = await self._repo.get_by_id(suksesor_id)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor retrieved successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_by_nip(self, nip: str) -> SuksesorResponse:
        """Get a Suksesor by NIP."""
        suksesor = await self._repo.get_by_nip(nip)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with NIP '{nip}' not found",
            )
        return SuksesorResponse(
            message="Suksesor retrieved successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_list(
        self,
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> SuksesorListResponse:
        """Get paginated list of Suksesor."""
        items, total = await self._repo.get_list(
            page=page, page_size=page_size, search=search, is_active=is_active
        )

        total_pages = math.ceil(total / page_size) if total > 0 else 1

        return SuksesorListResponse(
            message="Suksesor list retrieved successfully",
            data=SuksesorListDataResponse(
                items=[SuksesorDataResponse.model_validate(item) for item in items],
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            ),
        )

    async def update(
        self, suksesor_id: UUID, data: SuksesorUpdateRequest
    ) -> SuksesorResponse:
        """Update a Suksesor."""
        if data.nip and await self._repo.exists_by_nip(
            data.nip, exclude_id=suksesor_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Suksesor with NIP '{data.nip}' already exists",
            )

        suksesor = await self._repo.update(suksesor_id, data)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor updated successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def delete(self, suksesor_id: UUID) -> SuksesorDeleteResponse:
        """Delete a Suksesor (hard delete)."""
        if not await self._repo.delete(suksesor_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorDeleteResponse(
            message=f"Suksesor '{suksesor_id}' deleted successfully",
            data={"id": str(suksesor_id)},
        )

    async def soft_delete(self, suksesor_id: UUID) -> SuksesorResponse:
        """Soft delete a Suksesor (set is_active to False)."""
        suksesor = await self._repo.soft_delete(suksesor_id)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor deactivated successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )


def get_suksesor_service(db: AsyncSession = Depends(get_db)) -> SuksesorService:
    """Dependency injection for SuksesorService."""
    repository = SuksesorRepository(db)
    return SuksesorService(repository)