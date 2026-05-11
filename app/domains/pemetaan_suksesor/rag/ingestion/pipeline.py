import hashlib

from app.core.logger import log

from ..graph.main import GraphRAG
from ..vector.main import VectorRAG
from ..vector.schema import init_vector_store
from .extract import extract_from_chunks
from .minio_client import MinioClient
from .parsing import build_chunks, parse_xlsx_bytes


async def run_ingestion(
    vector_rag: VectorRAG,
    graph_rag: GraphRAG,
    minio_client: MinioClient,
    document_names: list[str] | None = None,
    force_reingest: bool = False,
) -> list[dict]:
    await init_vector_store()

    if document_names:
        filenames = document_names
    else:
        filenames = minio_client.list_documents()

    if not filenames:
        return []

    results = []
    for filename in filenames:
        result = await _ingest_one(
            filename=filename,
            vector_rag=vector_rag,
            graph_rag=graph_rag,
            minio_client=minio_client,
            force_reingest=force_reingest,
        )
        results.append(result)

    return results


async def _ingest_one(
    filename: str,
    vector_rag: VectorRAG,
    graph_rag: GraphRAG,
    minio_client: MinioClient,
    force_reingest: bool = False,
) -> dict:
    result = {
        "filename": filename,
        "status": "new",
        "chunk_count": 0,
        "entity_count": 0,
        "content_hash": "",
        "message": None,
    }

    try:
        file_bytes = minio_client.download_file(filename)
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Failed to download from MinIO: {e}"
        log.exception(f"❌ Download failed: {filename}")
        return result

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    result["content_hash"] = content_hash

    if not force_reingest:
        from app.domains.pemetaan_suksesor.repositories import IngestionLogRepository
        from app.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            repo = IngestionLogRepository(session)
            existing = await repo.get_by_filename(filename)

            if existing and existing.content_hash == content_hash:
                result["status"] = "skipped"
                log.info(f"⏭️ Skipped (unchanged): {filename}")
                return result

            if existing and existing.content_hash != content_hash:
                result["status"] = "updated"
                log.info(f"🔄 Updating (changed): {filename}")
                await _delete_existing_data(filename, vector_rag, graph_rag)

    try:
        parsed = parse_xlsx_bytes(file_bytes, filename)
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Failed to parse XLSX: {e}"
        log.exception(f"❌ Parse failed: {filename}")
        return result

    if not parsed.get("nama_jabatan"):
        result["status"] = "error"
        result["message"] = "No nama_jabatan found in document"
        log.warning(f"⚠️ No nama_jabatan: {filename}")
        return result

    chunks = build_chunks(parsed)
    for i, chunk in enumerate(chunks):
        chunk["doc_filename"] = filename
        chunk["nama_jabatan"] = parsed["nama_jabatan"]

    # Vector ingestion
    try:
        chunk_count = await vector_rag.add_chunks(chunks)
        result["chunk_count"] = chunk_count
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Vector ingestion failed: {e}"
        log.exception(f"❌ Vector ingest failed: {filename}")
        return result

    # Graph extraction + ingestion
    try:
        entities, relationships = await extract_from_chunks(chunks)
        if entities:
            graph_rag.upsert_entities(entities)
        if relationships:
            graph_rag.upsert_relationships(relationships)
        result["entity_count"] = len(entities)
    except Exception as e:
        log.warning(f"⚠️ Graph extraction failed for {filename}: {e}")
        result["entity_count"] = 0

    # Log to IngestionLog
    await _save_ingestion_log(filename, content_hash, result["status"], result["chunk_count"], result["entity_count"])

    log.info(f"✅ Ingested: {filename} — {result['chunk_count']} chunks, {result['entity_count']} entities ({result['status']})")
    return result


async def _delete_existing_data(filename: str, vector_rag: VectorRAG, graph_rag: GraphRAG):
    await vector_rag.delete_by_filename(filename)


async def _save_ingestion_log(filename: str, content_hash: str, status: str, chunk_count: int, entity_count: int):
    from app.domains.pemetaan_suksesor.repositories import IngestionLogRepository
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        repo = IngestionLogRepository(session)
        await repo.upsert(
            filename=filename,
            content_hash=content_hash,
            status=status,
            chunk_count=chunk_count,
            entity_count=entity_count,
        )
        await session.commit()