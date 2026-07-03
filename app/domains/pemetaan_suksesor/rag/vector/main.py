import asyncio
import concurrent.futures

import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.core.logger import log

from .embed import embed_single, embed_texts


class VectorRAG:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=settings.POSTGRES_URI.replace("+asyncpg", ""),
                min_size=2,
                max_size=10,
            )
            async with self._pool.acquire() as conn:
                await register_vector(conn)
        return self._pool

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def add_chunks(self, chunks: list[dict]) -> int:
        texts = [c["chunk_text"] for c in chunks]
        embeddings = await embed_texts(texts)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            count = 0
            for chunk, embedding in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO document_chunks
                        (doc_filename, chunk_index, chunk_text, nama_jabatan, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (doc_filename, chunk_index) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        nama_jabatan = EXCLUDED.nama_jabatan,
                        embedding = EXCLUDED.embedding
                    """,
                    chunk["doc_filename"],
                    chunk["chunk_index"],
                    chunk["chunk_text"],
                    chunk["nama_jabatan"],
                    embedding,
                )
                count += 1
        return count

    async def _retrieve_async(self, query: str, top_k: int | None = None) -> str:
        top_k = top_k or settings.RAG_VECTOR_TOP_K
        query_embedding = await embed_single(query)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT doc_filename, nama_jabatan, chunk_text,
                       embedding <=> $1 AS distance
                FROM document_chunks
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                top_k,
            )
            if not rows:
                return "Tidak ada hasil dari Vector RAG."

            results = []
            for row in rows:
                results.append(
                    f"[{row['nama_jabatan']} | {row['doc_filename']} | distance={row['distance']:.4f}]\n{row['chunk_text']}"
                )
        return "\n---\n".join(results)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._retrieve_async(query, top_k))
                return future.result()
        else:
            return asyncio.run(self._retrieve_async(query, top_k))

    async def delete_by_filename(self, doc_filename: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM document_chunks WHERE doc_filename = $1",
                doc_filename,
            )
            count = int(result.split()[-1])
            log.info(f"🗑️ Deleted {count} chunks for '{doc_filename}'")
            return count