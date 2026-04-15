from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.pemetaan_suksesor.models import MatchingHistory, Suksesor

from .dto.request import (
    SaveMatchingRequest,
    SuksesorCreateRequest,
    SuksesorUpdateRequest,
)


class SuksesorRepository:
    """Repository for Suksesor data access."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def create(self, data: SuksesorCreateRequest) -> Suksesor:
        """Create a new Suksesor."""
        suksesor = Suksesor(**data.model_dump())
        self._db.add(suksesor)
        await self._db.commit()
        await self._db.refresh(suksesor)
        return suksesor

    async def get_by_id(self, suksesor_id: UUID) -> Optional[Suksesor]:
        """Get a Suksesor by ID."""
        result = await self._db.execute(
            select(Suksesor).where(Suksesor.id == suksesor_id)
        )
        return result.scalar_one_or_none()

    async def get_by_nip(self, nip: str) -> Optional[Suksesor]:
        """Get a Suksesor by NIP."""
        result = await self._db.execute(select(Suksesor).where(Suksesor.nip == nip))
        return result.scalar_one_or_none()

    async def get_list(
        self,
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> tuple[list[Suksesor], int]:
        """Get paginated list of Suksesor with optional filters."""
        query = select(Suksesor)

        # Apply filters
        if search:
            query = query.where(
                (Suksesor.nama.ilike(f"%{search}%"))
                | (Suksesor.nip.ilike(f"%{search}%"))
            )

        if is_active is not None:
            query = query.where(Suksesor.is_active == is_active)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self._db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        query = query.order_by(Suksesor.created_at.desc())

        result = await self._db.execute(query)
        items = list(result.scalars().all())

        return items, total

    async def update(
        self, suksesor_id: UUID, data: SuksesorUpdateRequest
    ) -> Optional[Suksesor]:
        """Update a Suksesor."""
        suksesor = await self.get_by_id(suksesor_id)
        if not suksesor:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(suksesor, field, value)

        await self._db.commit()
        await self._db.refresh(suksesor)
        return suksesor

    async def delete(self, suksesor_id: UUID) -> bool:
        """Delete a Suksesor."""
        suksesor = await self.get_by_id(suksesor_id)
        if not suksesor:
            return False

        await self._db.delete(suksesor)
        await self._db.commit()
        return True

    async def soft_delete(self, suksesor_id: UUID) -> Optional[Suksesor]:
        """Soft delete a Suksesor by setting is_active to False."""
        suksesor = await self.get_by_id(suksesor_id)
        if not suksesor:
            return None

        suksesor.is_active = False  # type: ignore[assignment]
        await self._db.commit()
        await self._db.refresh(suksesor)
        return suksesor

    async def exists_by_nip(self, nip: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if a Suksesor with the given NIP exists."""
        query = select(func.count()).where(Suksesor.nip == nip)
        if exclude_id:
            query = query.where(Suksesor.id != exclude_id)

        result = await self._db.execute(query)
        count = result.scalar() or 0
        return count > 0


class MatchingHistoryRepository:
    """Repository for MatchingHistory data access."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def create(self, data: SaveMatchingRequest) -> MatchingHistory:
        """Create a new MatchingHistory record."""
        history = MatchingHistory(**data.model_dump())
        self._db.add(history)
        await self._db.commit()
        await self._db.refresh(history)
        return history

    async def get_by_id(self, history_id: UUID) -> Optional[MatchingHistory]:
        """Get a MatchingHistory by ID."""
        result = await self._db.execute(
            select(MatchingHistory).where(MatchingHistory.id == history_id)
        )
        return result.scalar_one_or_none()

    async def get_list(
        self,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[list[MatchingHistory], int]:
        """Get paginated list of MatchingHistory ordered by created_at desc."""
        query = select(MatchingHistory)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self._db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.order_by(MatchingHistory.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self._db.execute(query)
        items = list(result.scalars().all())

        return items, total
