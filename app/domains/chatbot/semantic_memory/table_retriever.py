from app.core.llm import LLMAdapter

from ..repositories import ChatbotRepository
from .config import SemanticMemoryConfig
from .types import RetrievedTable


class TableRetriever:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: SemanticMemoryConfig,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._config = config

    async def retrieve(self, keywords: list[str]) -> dict[str, RetrievedTable]:
        if not keywords:
            return {}

        try:
            table_available = await self._repository.is_vector_table_available(
                self._config.vector_table
            )
        except Exception:
            return {}

        if not table_available:
            return {}

        entity_map: dict[int, dict] = {}

        for keyword in keywords:
            try:
                vector = self._llm_adapter.embeddings.embed_query(keyword)
            except Exception:
                continue

            rows = await self._repository.retrieve_entities_by_vector(
                vector_table=self._config.vector_table,
                vector=vector,
                top_k=self._config.top_k_per_keyword,
            )
            for row in rows:
                row_id = int(row.get("id") or 0)
                similarity = float(row.get("similarity") or 0.0)
                previous = entity_map.get(row_id)
                if previous is None or similarity > float(previous.get("similarity") or 0.0):
                    entity_map[row_id] = row

        if not entity_map:
            return {}

        table_scores: dict[str, list[float]] = {}
        column_scores: dict[str, dict[str, float]] = {}

        for entity in entity_map.values():
            schema_name = str(entity.get("schema_name") or "")
            table_name = str(entity.get("table_name") or "")
            if not schema_name or not table_name:
                continue

            table_key = f"{schema_name}.{table_name}"
            similarity = float(entity.get("similarity") or 0.0)
            entity_type = str(entity.get("entity_type") or "")
            weight = (
                self._config.table_weight
                if entity_type == "table"
                else self._config.column_weight
            )

            weighted_score = similarity * weight
            table_scores.setdefault(table_key, []).append(weighted_score)

            column_name = entity.get("column_name")
            if column_name:
                col_scores = column_scores.setdefault(table_key, {})
                col_name = str(column_name)
                if similarity > col_scores.get(col_name, 0.0):
                    col_scores[col_name] = similarity

        ranked: list[tuple[str, RetrievedTable]] = []
        for table_key, scores in table_scores.items():
            max_score = max(scores)
            if max_score < self._config.retrieval_threshold:
                continue

            schema_name, table_name = table_key.split(".", 1)
            ranked.append(
                (
                    table_key,
                    RetrievedTable(
                        schema=schema_name,
                        table=table_name,
                        score=max_score,
                        column_scores=column_scores.get(table_key, {}),
                    ),
                )
            )

        ranked.sort(key=lambda item: item[1].score, reverse=True)
        limited = ranked[: self._config.max_retrieved_tables]
        return {table_key: table for table_key, table in limited}
