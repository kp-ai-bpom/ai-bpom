import csv
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional
from uuid import uuid4

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import log
from app.core.llm import LLMAdapter, init_llm
from app.db.database import get_db

from .question_rewriting import (
    QuestionRewritingService,
    get_question_rewriting_config,
)
from .repositories import ChatbotRepository
from .semantic_memory import SemanticMemoryPipeline, get_semantic_memory_config


class ChatbotService:
    """
    Service for handling chatbot LLM interactions.
    Uses LLMAdapter to access instruct, think, and deep_think models.
    """

    _chat_storage_ready = False
    _chat_storage_lock = Lock()

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository | None = None,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._semantic_pipeline: SemanticMemoryPipeline | None = None
        self._question_rewriting_service: QuestionRewritingService | None = None
        if self._repository is not None:
            self._semantic_pipeline = SemanticMemoryPipeline(
                llm_adapter=self._llm_adapter,
                repository=self._repository,
                config=get_semantic_memory_config(),
            )
            self._question_rewriting_service = QuestionRewritingService(
                llm_adapter=self._llm_adapter,
                repository=self._repository,
                config=get_question_rewriting_config(),
            )

    async def _ensure_chat_storage_ready(self) -> None:
        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        if self.__class__._chat_storage_ready:
            return

        await self._repository.ensure_chat_memory_tables()
        with self.__class__._chat_storage_lock:
            self.__class__._chat_storage_ready = True

    @staticmethod
    def _resolve_vector_table_name(table_name: str | None) -> str:
        default_table = get_semantic_memory_config().vector_table
        resolved = (table_name or default_table).strip()

        if resolved.lower() == "string":
            log.warning(
                "Invalid placeholder table_name='string' received; fallback to default table=%s",
                default_table,
            )
            return default_table

        return resolved

    def instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the instruct model for general purpose conversation"""
        response = self._llm_adapter.instruct.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    def think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the think model for reasoning-focused responses"""
        response = self._llm_adapter.think.bind(max_tokens=max_tokens).invoke(messages)
        return str(response.content)

    def deep_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the deep_think model for in-depth analysis"""
        response = self._llm_adapter.deep_think.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    async def a_instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the instruct model"""
        response = await self._llm_adapter.instruct.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the think model"""
        response = await self._llm_adapter.think.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_deep_think(
        self, messages: List, max_tokens: Optional[int] = None
    ) -> str:
        """Async invoke the deep_think model"""
        response = await self._llm_adapter.deep_think.bind(
            max_tokens=max_tokens
        ).ainvoke(messages)
        return str(response.content)

    async def send_message(
        self, user_id: str, message: str, session_id: Optional[str] = None
    ) -> dict[str, str]:
        """Run semantic pipeline and persist chat session/message in database."""
        normalized_user_id = str(user_id).strip()
        normalized_message = message.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_message:
            raise ValueError("message cannot be empty")

        normalized_session_id = session_id.strip() if session_id is not None else None
        if session_id is not None and not normalized_session_id:
            raise ValueError("session_id cannot be empty")
        resolved_session_id = normalized_session_id or str(uuid4())

        if self._semantic_pipeline is None:
            raise RuntimeError("Semantic pipeline is not configured")
        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()

        standalone_query = normalized_message
        if self._question_rewriting_service is not None:
            rewrite_result = await self._question_rewriting_service.rewrite(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                current_query=normalized_message,
            )
            rewritten_query = rewrite_result.rewritten_query.strip()
            if rewritten_query:
                standalone_query = rewritten_query

        pipeline_result = await self._semantic_pipeline.run(standalone_query)
        query = pipeline_result.sql
        explanation = pipeline_result.explanation

        await self._repository.persist_chat_message(
            user_id=normalized_user_id,
            session_id=resolved_session_id,
            title=normalized_message[:255],
            question=normalized_message,
            standalone_question=standalone_query,
            query=query,
            explanation=explanation,
        )

        return {
            "query": query,
            "explanation": explanation,
            "user_id": normalized_user_id,
            "session_id": resolved_session_id,
        }

    async def get_session_messages(
        self, user_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        """Get persisted session details and conversations for a user."""
        normalized_user_id = str(user_id).strip()
        normalized_session_id = session_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()
        return await self._repository.get_chat_session_messages(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

    async def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """List all persisted sessions that belong to a user."""
        normalized_user_id = str(user_id).strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()
        return await self._repository.list_chat_sessions(user_id=normalized_user_id)

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a persisted session if it belongs to the requesting user."""
        normalized_user_id = str(user_id).strip()
        normalized_session_id = session_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()

        deleted = await self._repository.delete_chat_session(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )
        if not deleted:
            return False

        QuestionRewritingService.clear_session_memory(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )
        return True

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text using the embeddings model"""
        vector = self._llm_adapter.embeddings.embed_query(text)
        return vector

    async def import_base_knowledge_csv(
        self,
        csv_path: str = "data/base_knowledge.csv",
        table_name: str | None = None,
        batch_size: int = 50,
        truncate_before_insert: bool = True,
    ) -> dict[str, Any]:
        """Import CSV base knowledge, embed its content, then upsert to pgvector table."""
        if self._repository is None:
            raise RuntimeError("Semantic pipeline is not configured")

        resolved_table = self._resolve_vector_table_name(table_name)
        if not resolved_table:
            raise ValueError("table_name cannot be empty")

        log.info(
            "🚀 Starting CSV import to vector table=%s csv_path=%s batch_size=%s truncate=%s",
            resolved_table,
            csv_path,
            batch_size,
            truncate_before_insert,
        )

        csv_file = Path(csv_path).expanduser()
        if not csv_file.is_absolute():
            csv_file = (Path.cwd() / csv_file).resolve()
        if not csv_file.exists() or not csv_file.is_file():
            raise ValueError(f"CSV file not found: {csv_file}")

        resolved_batch_size = max(1, min(500, int(batch_size)))

        await self._repository.ensure_base_knowledge_vector_table(resolved_table)
        if truncate_before_insert:
            await self._repository.truncate_vector_table(resolved_table)

        expected_dimensions = await self._repository.get_vector_table_embedding_dimensions(
            resolved_table
        )

        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)

        total_rows = len(csv_rows)
        targeted_rows = total_rows

        processed_rows = 0
        inserted_rows = 0
        failed_rows = 0
        sample_errors: list[str] = []
        pending_rows: list[dict[str, Any]] = []

        for row in csv_rows[:targeted_rows]:
            processed_rows += 1
            content = str(row.get("content") or "").strip()

            if not content:
                failed_rows += 1
                if len(sample_errors) < 10:
                    sample_errors.append(
                        f"row={processed_rows}: empty content"
                    )
                continue

            try:
                embedding = self._llm_adapter.embeddings.embed_query(content)
                if not embedding:
                    raise ValueError("empty embedding returned")
                pending_rows.append(
                    {
                        "entity_type": str(row.get("entity_type") or "").strip() or None,
                        "schema_name": str(row.get("schema_name") or "").strip() or None,
                        "table_name": str(row.get("table_name") or "").strip() or None,
                        "table_description": str(
                            row.get("table_description") or ""
                        ).strip()
                        or None,
                        "column_name": str(row.get("column_name") or "").strip() or None,
                        "column_alias": str(row.get("column_alias") or "").strip() or None,
                        "column_description": str(
                            row.get("column_description") or ""
                        ).strip()
                        or None,
                        "value_description": str(
                            row.get("value_description") or ""
                        ).strip()
                        or None,
                        "content": content,
                        "embedding": embedding,
                    }
                )
            except Exception as exc:
                failed_rows += 1
                if len(sample_errors) < 10:
                    sample_errors.append(f"row={processed_rows}: {exc}")

            if len(pending_rows) >= resolved_batch_size:
                try:
                    inserted_rows += await self._repository.insert_base_knowledge_rows(
                        vector_table=resolved_table,
                        rows=pending_rows,
                        expected_dimensions=expected_dimensions,
                    )
                except Exception as exc:
                    failed_rows += len(pending_rows)
                    if len(sample_errors) < 10:
                        sample_errors.append(f"batch@row={processed_rows}: {exc}")
                finally:
                    pending_rows = []

        if pending_rows:
            try:
                inserted_rows += await self._repository.insert_base_knowledge_rows(
                    vector_table=resolved_table,
                    rows=pending_rows,
                    expected_dimensions=expected_dimensions,
                )
            except Exception as exc:
                failed_rows += len(pending_rows)
                if len(sample_errors) < 10:
                    sample_errors.append(f"final-batch: {exc}")

        result = {
            "table_name": resolved_table,
            "csv_path": str(csv_file),
            "embedding_model": settings.AI_EMBEDDINGS_MODEL_NAME,
            "total_rows": total_rows,
            "targeted_rows": targeted_rows,
            "processed_rows": processed_rows,
            "inserted_rows": inserted_rows,
            "failed_rows": failed_rows,
            "truncated": bool(truncate_before_insert),
            "expected_dimensions": expected_dimensions,
            "sample_errors": sample_errors,
        }

        log.info(
            "✅ CSV import finished table=%s inserted=%s failed=%s targeted=%s",
            resolved_table,
            inserted_rows,
            failed_rows,
            targeted_rows,
        )
        if failed_rows > 0 and sample_errors:
            log.warning(
                "CSV import encountered errors table=%s samples=%s",
                resolved_table,
                sample_errors[:3],
            )

        return result


def get_llm_adapter() -> LLMAdapter:
    """Dependency untuk mendapatkan LLMAdapter"""
    return init_llm()


# dependencies injection
def get_chatbot_service(
    llm_adapter: LLMAdapter = Depends(get_llm_adapter),
) -> ChatbotService:
    """Dependency untuk mendapatkan ChatbotService"""
    return ChatbotService(llm_adapter)


async def get_chatbot_pipeline_service(
    llm_adapter: LLMAdapter = Depends(get_llm_adapter),
    db: AsyncSession = Depends(get_db),
) -> ChatbotService:
    """Dependency untuk mendapatkan ChatbotService dengan semantic pipeline."""
    repository = ChatbotRepository(db)
    return ChatbotService(llm_adapter=llm_adapter, repository=repository)
