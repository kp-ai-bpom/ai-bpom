import math
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db

from ..dto.request import SaveMatchingRequest
from ..dto.response import (
    MatchingHistoryDetail,
    MatchingHistoryDetailResponse,
    MatchingHistoryListData,
    MatchingHistoryListResponse,
    MatchingHistorySaveResponse,
    MatchingHistorySummary,
)
from ..repositories import MatchingHistoryRepository


class MatchingHistoryService:
    """Service for matching history business logic."""

    def __init__(self, repository: MatchingHistoryRepository):
        self._repo = repository

    async def save(self, data: SaveMatchingRequest) -> MatchingHistorySaveResponse:
        """Save a matching result to history."""
        history = await self._repo.create(data)
        return MatchingHistorySaveResponse(
            message="Riwayat matching berhasil disimpan",
            data=MatchingHistoryDetail.model_validate(history),
        )

    async def get_by_id(self, history_id: UUID) -> MatchingHistoryDetailResponse:
        """Get a matching history by ID."""
        history = await self._repo.get_by_id(history_id)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Matching history with ID '{history_id}' not found",
            )
        return MatchingHistoryDetailResponse(
            message="Riwayat matching berhasil diambil",
            data=MatchingHistoryDetail.model_validate(history),
        )

    async def get_list(
        self, page: int = 1, page_size: int = 10
    ) -> MatchingHistoryListResponse:
        """Get paginated list of matching history."""
        items, total = await self._repo.get_list(page=page, page_size=page_size)
        total_pages = math.ceil(total / page_size) if total > 0 else 1
        return MatchingHistoryListResponse(
            message="Daftar riwayat matching berhasil diambil",
            data=MatchingHistoryListData(
                items=[MatchingHistorySummary.model_validate(i) for i in items],
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            ),
        )


def get_matching_history_service(
    db: AsyncSession = Depends(get_db),
) -> MatchingHistoryService:
    """Dependency injection for MatchingHistoryService."""
    repository = MatchingHistoryRepository(db)
    return MatchingHistoryService(repository)