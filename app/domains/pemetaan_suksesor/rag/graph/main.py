from neo4j import GraphDatabase

from app.core.config import settings
from app.core.logger import log


class GraphRAG:
    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    def close(self):
        self._driver.close()

    def retrieve(self, entity: str) -> str:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {name: $name})-[r:RELATED_TO]-(b:Entity)
                RETURN a.name AS source, b.name AS target,
                       r.description AS description, r.weight AS weight
                LIMIT $limit
                """,
                name=entity,
                limit=settings.RAG_GRAPH_LIMIT,
            )
            records = list(result)
            if not records:
                return f"Entitas '{entity}' tidak ditemukan di Knowledge Graph."

            lines = []
            for rec in records:
                lines.append(
                    f"{rec['source']} --[{rec['weight']}]--> {rec['target']}: {rec['description']}"
                )
        return "\n".join(lines)

    def search_entities(self, keyword: str, limit: int = 10) -> list[str]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $keyword
                RETURN e.name AS name, e.type AS type
                LIMIT $limit
                """,
                keyword=keyword,
                limit=limit,
            )
            return [f"{row['name']} ({row['type']})" for row in result]

    def upsert_entities(self, entities: list[tuple[str, str, str]]):
        with self._driver.session() as session:
            for name, entity_type, description in entities:
                session.run(
                    """
                    MERGE (n:Entity {name: $name})
                    SET n.type = $type, n.description = $description
                    """,
                    name=name,
                    type=entity_type,
                    description=description,
                )
        log.info(f"✅ Upserted {len(entities)} entities")

    def upsert_relationships(self, relationships: list[tuple[str, str, str, int]]):
        with self._driver.session() as session:
            for source, target, description, weight in relationships:
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:RELATED_TO]->(b)
                    SET r.description = $description, r.weight = $weight
                    """,
                    source=source,
                    target=target,
                    description=description,
                    weight=weight,
                )
        log.info(f"✅ Upserted {len(relationships)} relationships")

    def delete_by_entity_names(self, entity_names: list[str]):
        with self._driver.session() as session:
            for name in entity_names:
                session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    DETACH DELETE e
                    """,
                    name=name,
                )
        log.info(f"🗑️ Deleted {len(entity_names)} entities from graph")