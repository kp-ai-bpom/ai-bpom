"""Async repository layer for penilaian_makalah."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.penilaian_makalah.models import EvaluationResult, IngestionLog


class EvaluationRepository:
    """Data access layer for evaluation results and ingestion logs."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_history(self, limit: int = 100) -> list[EvaluationResult]:
        """Return the newest evaluation results first."""
        stmt = (
            select(EvaluationResult)
            .order_by(EvaluationResult.created_at.desc(), EvaluationResult.id.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def save_evaluation(
        self,
        paper_filename: str,
        jabatan: str,
        result: dict[str, Any],
        query_mode: str,
    ) -> EvaluationResult:
        """Persist a completed evaluation result."""
        record = EvaluationResult(
            paper_filename=paper_filename,
            jabatan=jabatan,
            query_mode=query_mode,
            scores=result.get("scores", {}),
            justification=result.get("justification", {}),
            evidence=result.get("evidence", {}),
            uncertainty_metrics=result.get("uncertainty_metrics", {}),
            ringkasan=result.get("Ringkasan") or result.get("ringkasan") or "",
            final_score=float(result.get("final_score", 0.0)),
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)
        return record

    async def create_ingestion_log(
        self,
        filename: str,
        status: str,
        error_message: str | None = None,
    ) -> IngestionLog:
        """Persist an ingestion log entry."""
        record = IngestionLog(
            filename=filename,
            status=status,
            error_message=error_message,
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)
        return record
