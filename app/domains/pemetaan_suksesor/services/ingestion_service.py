from app.core.logger import log

from ..rag.graph.main import GraphRAG
from ..rag.ingestion.minio_client import MinioClient
from ..rag.ingestion.pipeline import run_ingestion
from ..rag.vector.main import VectorRAG
from ..dto.response import (
    IngestLogDetailResponse,
    IngestLogListResponse,
    IngestLogSummary,
    IngestResponse,
    IngestResult,
    UploadResponse,
)


class IngestionService:
    def __init__(self, vector_rag: VectorRAG, graph_rag: GraphRAG, minio_client: MinioClient):
        self._vector_rag = vector_rag
        self._graph_rag = graph_rag
        self._minio = minio_client

    async def ingest(self, document_names: list[str] | None = None, force_reingest: bool = False) -> IngestResponse:
        results = await run_ingestion(
            vector_rag=self._vector_rag,
            graph_rag=self._graph_rag,
            minio_client=self._minio,
            document_names=document_names,
            force_reingest=force_reingest,
        )
        return IngestResponse(
            total_documents=len(results),
            results=[IngestResult(**r) for r in results],
        )

    async def upload_and_ingest(self, filename: str, data: bytes, force_reingest: bool = False) -> UploadResponse:
        self._minio.upload_file(filename, data)
        results = await run_ingestion(
            vector_rag=self._vector_rag,
            graph_rag=self._graph_rag,
            minio_client=self._minio,
            document_names=[filename],
            force_reingest=force_reingest,
        )
        return UploadResponse(
            filename=filename,
            minio_path=f"{self._minio._bucket}/{filename}",
            ingest_result=IngestResult(**results[0]) if results else IngestResult(
                filename=filename, status="error", chunk_count=0, entity_count=0,
                content_hash="", message="No results returned",
            ),
        )

    async def list_logs(self, offset: int = 0, limit: int = 50) -> IngestLogListResponse:
        from ..repositories import IngestionLogRepository
        from app.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            repo = IngestionLogRepository(session)
            logs = await repo.get_list(offset=offset, limit=limit)
            summaries = [
                IngestLogSummary(
                    id=l.id,
                    filename=l.filename,
                    content_hash=l.content_hash,
                    status=l.status,
                    chunk_count=l.chunk_count,
                    entity_count=l.entity_count,
                    ingested_at=str(l.ingested_at),
                )
                for l in logs
            ]
            return IngestLogListResponse(message="Ingestion logs", data=summaries)

    async def get_log(self, log_id: int) -> IngestLogDetailResponse | None:
        from ..repositories import IngestionLogRepository
        from app.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            repo = IngestionLogRepository(session)
            log_entry = await repo.get_by_id(log_id)
            if not log_entry:
                return None
            return IngestLogDetailResponse(
                message="Ingestion log detail",
                data=IngestLogSummary(
                    id=log_entry.id,
                    filename=log_entry.filename,
                    content_hash=log_entry.content_hash,
                    status=log_entry.status,
                    chunk_count=log_entry.chunk_count,
                    entity_count=log_entry.entity_count,
                    ingested_at=str(log_entry.ingested_at),
                ),
            )


def get_ingestion_service() -> IngestionService:
    from ..rag.graph.main import GraphRAG
    from ..rag.ingestion.minio_client import MinioClient
    from ..rag.vector.main import VectorRAG
    return IngestionService(
        vector_rag=VectorRAG(),
        graph_rag=GraphRAG(),
        minio_client=MinioClient(),
    )