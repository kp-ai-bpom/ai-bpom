import asyncpg

from app.core.config import settings
from app.core.logger import log


async def init_vector_store():
    """Create document_chunks table and HNSW index if not exist."""
    conn = await asyncpg.connect(settings.POSTGRES_URI.replace("+asyncpg", ""))
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                doc_filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                nama_jabatan TEXT NOT NULL,
                embedding VECTOR({settings.RAG_EMBEDDING_DIMENSIONS}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(doc_filename, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_jabatan
            ON document_chunks (nama_jabatan)
        """)
        log.info("✅ Vector store schema initialized")
    finally:
        await conn.close()