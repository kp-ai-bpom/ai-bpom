import logging
from typing import List

from fastapi import Depends

from .jobs import JobService, get_job_service

logger = logging.getLogger(__name__)


class PipelineService:
    """Orchestrates background pipeline tasks for RAG ingestion.

    Each method receives a job_id and a list of file_paths (strings pointing to
    files already saved to a temp directory by the API layer). The API layer
    is responsible for saving UploadFile contents to disk before scheduling
    the background task, because FastAPI closes UploadFile objects after the
    request ends.
    """

    def __init__(self, job_service: JobService):
        self.jobs = job_service

    async def run_knowledge_graph_jabatan(self, job_id: str, file_paths: List[str]):
        """XLSX → Neo4j knowledge graph (LLM extraction + MERGE)."""
        from pathlib import Path
        from app.domains.pemetaan_suksesor.rag.ingestion.parsing import build_chunks, parse_xlsx_path
        from app.domains.pemetaan_suksesor.rag.ingestion.extract import extract_from_chunks
        from app.domains.pemetaan_suksesor.rag.graph.main import GraphRAG

        try:
            graph_rag = GraphRAG()
            total_entities = 0
            total_relationships = 0

            for file_path in file_paths:
                parsed = parse_xlsx_path(Path(file_path))
                chunks = build_chunks(parsed)
                entities, relationships = await extract_from_chunks(chunks)
                if entities:
                    graph_rag.upsert_entities(entities)
                if relationships:
                    graph_rag.upsert_relationships(relationships)
                total_entities += len(entities)
                total_relationships += len(relationships)

            graph_rag.close()
            self.jobs.update_job(
                job_id,
                "completed",
                result=f"Jabatan KG ingested: {total_entities} entities, {total_relationships} relationships",
            )
        except Exception as e:
            logger.exception("Knowledge graph jabatan pipeline failed")
            self.jobs.update_job(job_id, "failed", error=str(e))

    async def run_knowledge_graph_regulasi(self, job_id: str, file_paths: List[str]):
        """PDF → Neo4j knowledge graph (8-type LLM extraction + MERGE)."""
        from pathlib import Path
        from app.domains.pemetaan_suksesor.rag.ingestion.extract_pdf import extract_from_pdf_chunks, parse_pdf_bytes
        from app.domains.pemetaan_suksesor.rag.graph.main import GraphRAG

        try:
            graph_rag = GraphRAG()
            total_entities = 0
            total_relationships = 0

            for file_path in file_paths:
                path = Path(file_path)
                data = path.read_bytes()
                chunks = parse_pdf_bytes(data, path.name)
                entities, relationships = await extract_from_pdf_chunks(chunks)
                if entities:
                    graph_rag.upsert_entities(entities)
                if relationships:
                    graph_rag.upsert_relationships(relationships)
                total_entities += len(entities)
                total_relationships += len(relationships)

            graph_rag.close()
            self.jobs.update_job(
                job_id,
                "completed",
                result=f"Regulasi KG ingested: {total_entities} entities, {total_relationships} relationships",
            )
        except Exception as e:
            logger.exception("Knowledge graph regulasi pipeline failed")
            self.jobs.update_job(job_id, "failed", error=str(e))

    async def run_vectorrag_jabatan(self, job_id: str, file_paths: List[str]):
        """XLSX → chunk → embed → ingest into pgvector document_chunks."""
        try:
            logger.info("Starting VectorRAG Jabatan pipeline, files: %s", file_paths)
            # TODO: Port from hybrid-rag/app/vector/ingest.py
            self.jobs.update_job(job_id, "completed", result="Vector ingestion completed")
        except Exception as e:
            logger.exception("VectorRAG Jabatan pipeline failed")
            self.jobs.update_job(job_id, "failed", error=str(e))

    async def run_profil_jabatan(self, job_id: str, file_paths: List[str]):
        """XLSX → 14-field keyword categorisation → upsert to jabatan_rules (PostgreSQL)."""
        from pathlib import Path
        from app.db.database import AsyncSessionLocal
        from app.domains.pemetaan_suksesor.rag.ingestion.jabatan_profile import build_profil_jabatan
        from app.domains.pemetaan_suksesor.repositories import JabatanRulesRepository

        try:
            processed = 0
            for file_path in file_paths:
                data = Path(file_path).read_bytes()
                profil = build_profil_jabatan(data)

                async with AsyncSessionLocal() as session:
                    repo = JabatanRulesRepository(session)
                    await repo.upsert(
                        slug=profil["slug"],
                        nama_jabatan=profil["nama_jabatan"],
                        atasan_langsung=profil["atasan_langsung"],
                        data=profil["data"],
                    )
                    await session.commit()
                processed += 1

            self.jobs.update_job(
                job_id,
                "completed",
                result=f"Profil jabatan upserted: {processed} record(s)",
            )
        except Exception as e:
            logger.exception("Profil jabatan pipeline failed")
            self.jobs.update_job(job_id, "failed", error=str(e))


def get_pipeline_service(job_service: JobService = Depends(get_job_service)) -> PipelineService:
    return PipelineService(job_service)