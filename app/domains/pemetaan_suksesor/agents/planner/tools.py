import asyncio
import json
import logging
from typing import Optional

import asyncpg
from neo4j import GraphDatabase, Driver
from strands import tool

from app.core.config import settings

logger = logging.getLogger(__name__)

_neo4j_driver: Optional[Driver] = None


def _get_neo4j_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _pg_dsn() -> str:
    dsn = settings.POSTGRES_URI
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
    return dsn


@tool
def query_jabatan_profile(nama_jabatan: str) -> str:
    """
    Retrieve profil jabatan terstruktur dari PostgreSQL tabel jabatan_rules.
    Gunakan tool ini untuk mendapatkan data profil jabatan berdasarkan nama jabatan.

    Args:
        nama_jabatan: Nama jabatan yang dicari (contoh: "Inspektur I", "Direktur Registrasi Obat")
    """

    async def _query():
        conn = await asyncpg.connect(dsn=_pg_dsn())
        try:
            row = await conn.fetchrow(
                """
                SELECT nama_jabatan, atasan_langsung, data
                FROM jabatan_rules
                WHERE nama_jabatan ILIKE '%' || $1 || '%'
                LIMIT 1
                """,
                nama_jabatan,
            )
            if not row:
                return f"Profil jabatan '{nama_jabatan}' tidak ditemukan di database."
            result = {
                "nama_jabatan": row["nama_jabatan"],
                "atasan_langsung": row["atasan_langsung"],
                "data": json.loads(row["data"])
                if isinstance(row["data"], str)
                else row["data"],
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            await conn.close()

    return asyncio.run(_query())


@tool
def query_neo4j_depth2(entity_name: str) -> str:
    """
    Traversal depth-2 dari entity yang diberikan di Neo4j context KG.
    Mengambil entity langsung (depth-1) dan tetangga dari tetangga (depth-2).
    Gunakan tool ini untuk mendapatkan aturan penilaian dan relasi regulasi dari knowledge graph.

    Args:
        entity_name: Nama entity yang akan di-traverse (contoh: "JPT Pratama", "Seleksi Terbuka JPT")
    """
    driver = _get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {name: $name})-[r1:RELATED_TO]-(b:Entity)
                OPTIONAL MATCH (b)-[r2:RELATED_TO]-(c:Entity)
                WHERE b <> c AND a <> c
                RETURN a.name AS source,
                       b.name AS b_name, b.type AS b_type, b.description AS b_desc,
                       c.name AS c_name, c.type AS c_type, c.description AS c_desc,
                       r1.description AS r1_desc, r1.weight AS r1_weight,
                       r2.description AS r2_desc, r2.weight AS r2_weight
                LIMIT 80
                """,
                name=entity_name,
            )
            records = list(result)

        if not records:
            return f"Entity '{entity_name}' tidak ditemukan di Knowledge Graph atau tidak memiliki relasi."

        lines = [f"=== Depth-2 Subgraph dari '{entity_name}' ==="]
        depth1_entities: set = set()
        depth2_entities: set = set()

        for rec in records:
            b_key = rec["b_name"]
            if b_key not in depth1_entities:
                depth1_entities.add(b_key)
                lines.append(
                    f"[Depth-1] {rec['source']} --[{rec.get('r1_weight', '')}]--> {b_key} ({rec.get('b_type', '')}): {rec.get('b_desc', '')}"
                )
            c_name = rec["c_name"]
            if c_name and c_name not in depth2_entities:
                depth2_entities.add(c_name)
                lines.append(
                    f"  [Depth-2] {b_key} --[{rec.get('r2_weight', '')}]--> {c_name} ({rec.get('c_type', '')}): {rec.get('c_desc', '')}"
                )

        return "\n".join(lines)
    except Exception as e:
        logger.error("Error querying Neo4j depth2: %s", e)
        return f"Error querying Neo4j: {str(e)}"


@tool
def search_neo4j_entities(keyword: str, limit: int = 10) -> str:
    """
    Fuzzy search entity di Neo4j berdasarkan keyword. Mengembalikan daftar entity yang namanya mengandung keyword.
    Gunakan tool ini saat nama entity tidak diketahui secara exact match.

    Args:
        keyword: Kata kunci untuk mencari entity (contoh: "seleksi", "talenta", "penilaian")
        limit: Jumlah maksimum hasil yang dikembalikan (default: 10)
    """
    driver = _get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $keyword
                RETURN e.name AS name, e.type AS type, e.description AS description
                LIMIT $limit
                """,
                keyword=keyword,
                limit=limit,
            )
            records = list(result)

        if not records:
            return f"Tidak ditemukan entity dengan keyword '{keyword}'."

        lines = [f"=== Hasil pencarian entity untuk '{keyword}' ==="]
        for rec in records:
            lines.append(
                f"- {rec['name']} ({rec.get('type', '')}): {rec.get('description', '')}"
            )

        return "\n".join(lines)
    except Exception as e:
        logger.error("Error searching Neo4j entities: %s", e)
        return f"Error searching Neo4j: {str(e)}"
