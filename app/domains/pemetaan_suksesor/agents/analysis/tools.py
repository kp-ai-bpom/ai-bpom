import asyncio

import asyncpg
from pgvector.asyncpg import register_vector
from strands import tool

from app.core.config import settings
from app.domains.pemetaan_suksesor.rag.vector.embed import embed_single


@tool
def search_vector_rag(query: str, top_k: int = 5) -> str:
    """
    Semantic search pada document_chunks di pgvector.
    Gunakan tool ini untuk menemukan dokumen RHK jabatan yang semantically similar
    dengan fungsi_utama atau kompetensi_spesifik target jabatan.

    Args:
        query: Teks query untuk semantic search (contoh: "audit kinerja", "manajemen risiko")
        top_k: Jumlah hasil teratas yang dikembalikan (default: 5)
    """

    async def _search():
        query_embedding = await embed_single(query)

        dsn = settings.POSTGRES_URI
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

        conn = await asyncpg.connect(dsn=dsn)
        try:
            await register_vector(conn)
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

            lines = [f"=== Hasil pencarian vektor untuk '{query}' (top {top_k}) ==="]
            for row in rows:
                lines.append(
                    f"[{row['nama_jabatan']} | {row['doc_filename']} | distance={row['distance']:.4f}]\n{row['chunk_text']}"
                )
                lines.append("---")
        finally:
            await conn.close()

        return "\n".join(lines)

    return asyncio.run(_search())