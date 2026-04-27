import json
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import log


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUESTION_REWRITING_TABLE = "question_rewriting_episodes"
_CHAT_SESSIONS_TABLE = "chat_sessions"
_CHAT_MESSAGES_TABLE = "chat_messages"
_PENDING_CLARIFICATIONS_TABLE = "chat_pending_clarifications"
_PROCEDURAL_RULES_TABLE = "chat_user_procedural_rules"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _safe_table_reference(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 2:
        raise ValueError("Invalid table reference")

    safe_parts: list[str] = []
    for part in parts:
        if not _IDENTIFIER_PATTERN.fullmatch(part):
            raise ValueError("Invalid table identifier")
        safe_parts.append(_quote_identifier(part))
    return ".".join(safe_parts)


def _split_table_reference(name: str) -> tuple[str, str]:
    parts = name.split(".")
    if len(parts) == 1:
        schema_name = "public"
        table_name = parts[0]
    elif len(parts) == 2:
        schema_name, table_name = parts
    else:
        raise ValueError("Invalid table reference")

    if not _IDENTIFIER_PATTERN.fullmatch(schema_name):
        raise ValueError("Invalid schema identifier")
    if not _IDENTIFIER_PATTERN.fullmatch(table_name):
        raise ValueError("Invalid table identifier")
    return schema_name, table_name


def _build_index_name(prefix: str, table_ref: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", table_ref).strip("_").lower()
    if not normalized:
        normalized = "vector_table"
    max_prefix = 20
    base_prefix = prefix[:max_prefix]
    name = f"{base_prefix}_{normalized}"
    return name[:63]


def _embedding_to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def _align_embedding_dimensions(
    embedding: list[float],
    expected_dimensions: int | None,
) -> list[float]:
    if not embedding:
        return []
    if expected_dimensions is None or expected_dimensions <= 0:
        return embedding

    current_dimensions = len(embedding)
    if current_dimensions == expected_dimensions:
        return embedding
    if current_dimensions > expected_dimensions:
        return embedding[:expected_dimensions]
    return embedding + [0.0] * (expected_dimensions - current_dimensions)


class ChatbotRepository:
    """Repository layer for chatbot semantic memory and SQL execution."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def ensure_chat_memory_tables(self) -> None:
        ddl_statements = [
            f"""
            CREATE TABLE IF NOT EXISTS {_CHAT_SESSIONS_TABLE} (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                title_source VARCHAR(50) DEFAULT 'first_user_message',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS title VARCHAR(255)
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS title_source VARCHAR(50) DEFAULT 'first_user_message'
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ALTER COLUMN created_at SET DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_CHAT_SESSIONS_TABLE}
            ALTER COLUMN updated_at SET DEFAULT NOW()
            """,
            f"""
            UPDATE {_CHAT_SESSIONS_TABLE}
            SET
                title = COALESCE(NULLIF(title, ''), LEFT(session_id, 255)),
                title_source = COALESCE(NULLIF(title_source, ''), 'first_user_message'),
                created_at = COALESCE(created_at, NOW()),
                updated_at = COALESCE(updated_at, created_at, NOW())
            WHERE
                title IS NULL
                OR title = ''
                OR title_source IS NULL
                OR title_source = ''
                OR created_at IS NULL
                OR updated_at IS NULL
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
            ON {_CHAT_SESSIONS_TABLE} (user_id, updated_at DESC)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
            ON {_CHAT_SESSIONS_TABLE} (user_id)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {_CHAT_MESSAGES_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL
                    REFERENCES {_CHAT_SESSIONS_TABLE}(session_id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                standalone_question TEXT,
                query TEXT NOT NULL,
                explanation TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS session_id VARCHAR(255)
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS question TEXT
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS standalone_question TEXT
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS query TEXT
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS explanation TEXT
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS pipeline_trace JSONB
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_CHAT_MESSAGES_TABLE}
            ALTER COLUMN created_at SET DEFAULT NOW()
            """,
            f"""
            UPDATE {_CHAT_MESSAGES_TABLE}
            SET
                created_at = COALESCE(created_at, NOW())
            WHERE
                created_at IS NULL
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
            ON {_CHAT_MESSAGES_TABLE} (session_id, created_at, id)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {_PENDING_CLARIFICATIONS_TABLE} (
                pending_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                standalone_question TEXT NOT NULL,
                schema_context TEXT NOT NULL,
                relevant_schema JSONB NOT NULL,
                clarification_question TEXT NOT NULL,
                options JSONB NOT NULL,
                ambiguity_type VARCHAR(64) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_chat_pending_user_session
            ON {_PENDING_CLARIFICATIONS_TABLE} (user_id, session_id)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_chat_pending_expires
            ON {_PENDING_CLARIFICATIONS_TABLE} (expires_at)
            """,
        ]

        try:
            for statement in ddl_statements:
                await self._db.execute(text(statement))
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def persist_chat_message(
        self,
        user_id: str,
        session_id: str,
        title: str,
        question: str,
        standalone_question: str,
        query: str,
        explanation: str,
        pipeline_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        session_insert = text(
            f"""
            INSERT INTO {_CHAT_SESSIONS_TABLE}
                (session_id, user_id, title, title_source, created_at, updated_at)
            VALUES
                (:session_id, :user_id, :title, :title_source, NOW(), NOW())
            ON CONFLICT (session_id) DO NOTHING
            """
        )
        session_owner_query = text(
            f"""
            SELECT user_id
            FROM {_CHAT_SESSIONS_TABLE}
            WHERE session_id = :session_id
            LIMIT 1
            """
        )
        session_title_update = text(
            f"""
            UPDATE {_CHAT_SESSIONS_TABLE}
            SET
                title = COALESCE(NULLIF(title, ''), :title),
                title_source = COALESCE(NULLIF(title_source, ''), :title_source)
            WHERE session_id = :session_id
            """
        )
        message_insert = text(
            f"""
            INSERT INTO {_CHAT_MESSAGES_TABLE}
                (session_id, question, standalone_question, query, explanation,
                 pipeline_trace, created_at)
            VALUES
                (:session_id, :question, :standalone_question, :query, :explanation,
                 CAST(:pipeline_trace AS JSONB), NOW())
            """
        )
        session_touch = text(
            f"""
            UPDATE {_CHAT_SESSIONS_TABLE}
            SET updated_at = NOW()
            WHERE session_id = :session_id
            """
        )

        params = {
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "title_source": "first_user_message",
            "question": question,
            "standalone_question": standalone_question,
            "query": query,
            "explanation": explanation,
            "pipeline_trace": (
                json.dumps(pipeline_trace, ensure_ascii=False, default=str)
                if pipeline_trace
                else None
            ),
        }

        try:
            await self._db.execute(session_insert, params)

            owner_result = await self._db.execute(
                session_owner_query,
                {"session_id": session_id},
            )
            existing_owner = owner_result.scalar_one_or_none()
            if existing_owner is None:
                raise ValueError("session_id was not persisted")
            if str(existing_owner) != user_id:
                raise ValueError("session_id is not owned by this user")

            await self._db.execute(session_title_update, params)
            await self._db.execute(message_insert, params)
            await self._db.execute(session_touch, {"session_id": session_id})
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def get_chat_session_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        session_query = text(
            f"""
            SELECT session_id, user_id, title, created_at, updated_at
            FROM {_CHAT_SESSIONS_TABLE}
            WHERE session_id = :session_id
              AND user_id = :user_id
            LIMIT 1
            """
        )
        message_query = text(
            f"""
            SELECT question, standalone_question, query, explanation,
                   pipeline_trace, created_at
            FROM {_CHAT_MESSAGES_TABLE}
            WHERE session_id = :session_id
            ORDER BY created_at, id
            """
        )

        session_row = (
            await self._db.execute(
                session_query,
                {
                    "session_id": session_id,
                    "user_id": user_id,
                },
            )
        ).mappings().first()

        if session_row is None:
            return None

        message_rows = (
            await self._db.execute(message_query, {"session_id": session_id})
        ).mappings().all()

        def _decode_pipeline_trace(value: Any) -> list[dict[str, Any]] | None:
            # Driver asyncpg umumnya mengembalikan kolom JSONB sebagai
            # ``list``/``dict`` Python langsung. Namun beberapa konfigurasi
            # (mis. driver lain atau migrasi data) bisa menyimpannya sebagai
            # string JSON. Kasus dict di-bungkus menjadi single-element list
            # agar UI tetap bisa render konsisten.
            if value is None:
                return None
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    return None
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    return None
                if isinstance(decoded, list):
                    return decoded
                if isinstance(decoded, dict):
                    return [decoded]
                return None
            return None

        conversations = [
            {
                "question": str(row["question"]),
                "standalone_question": str(
                    row["standalone_question"] or row["question"]
                ),
                "query": str(row["query"]),
                "explanation": (
                    None if row["explanation"] is None else str(row["explanation"])
                ),
                "pipeline_trace": _decode_pipeline_trace(row["pipeline_trace"]),
                "created_at": row["created_at"],
            }
            for row in message_rows
        ]

        return {
            "user_id": str(session_row["user_id"]),
            "session_id": str(session_row["session_id"]),
            "title": str(session_row["title"]),
            "created_at": session_row["created_at"],
            "updated_at": session_row["updated_at"],
            "conversations": conversations,
        }

    async def list_chat_sessions(self, user_id: str) -> list[dict[str, Any]]:
        query = text(
            f"""
            SELECT session_id, title, created_at, updated_at
            FROM {_CHAT_SESSIONS_TABLE}
            WHERE user_id = :user_id
            ORDER BY updated_at DESC, created_at DESC
            """
        )
        rows = (
            await self._db.execute(
                query,
                {
                    "user_id": user_id,
                },
            )
        ).mappings().all()
        return [
            {
                "session_id": str(row["session_id"]),
                "title": str(row["title"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def delete_chat_session(self, user_id: str, session_id: str) -> bool:
        query = text(
            f"""
            DELETE FROM {_CHAT_SESSIONS_TABLE}
            WHERE session_id = :session_id
              AND user_id = :user_id
            RETURNING session_id
            """
        )

        try:
            deleted = await self._db.execute(
                query,
                {
                    "session_id": session_id,
                    "user_id": user_id,
                },
            )
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

        return deleted.scalar_one_or_none() is not None

    async def create_pending_clarification(
        self,
        user_id: str,
        session_id: str,
        standalone_question: str,
        schema_context: str,
        relevant_schema: dict[str, Any],
        clarification_question: str,
        options: list[dict[str, Any]] | list[str],
        ambiguity_type: str,
        expires_at: datetime,
    ) -> str:
        pending_id = str(uuid4())
        query = text(
            f"""
            INSERT INTO {_PENDING_CLARIFICATIONS_TABLE}
                (
                    pending_id,
                    user_id,
                    session_id,
                    standalone_question,
                    schema_context,
                    relevant_schema,
                    clarification_question,
                    options,
                    ambiguity_type,
                    created_at,
                    expires_at
                )
            VALUES
                (
                    :pending_id,
                    :user_id,
                    :session_id,
                    :standalone_question,
                    :schema_context,
                    CAST(:relevant_schema AS JSONB),
                    :clarification_question,
                    CAST(:options AS JSONB),
                    :ambiguity_type,
                    NOW(),
                    :expires_at
                )
            """
        )

        params = {
            "pending_id": pending_id,
            "user_id": user_id,
            "session_id": session_id,
            "standalone_question": standalone_question,
            "schema_context": schema_context,
            "relevant_schema": json.dumps(relevant_schema, ensure_ascii=True),
            "clarification_question": clarification_question,
            "options": json.dumps(options, ensure_ascii=True),
            "ambiguity_type": ambiguity_type,
            "expires_at": expires_at,
        }

        try:
            await self._db.execute(query, params)
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

        return pending_id

    async def load_latest_pending_clarification(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Load the latest pending clarification for a session (auto-detect mode)."""
        query = text(
            f"""
            SELECT
                pending_id,
                user_id,
                session_id,
                standalone_question,
                schema_context,
                relevant_schema,
                clarification_question,
                options,
                ambiguity_type,
                created_at,
                expires_at
            FROM {_PENDING_CLARIFICATIONS_TABLE}
            WHERE user_id = :user_id
              AND session_id = :session_id
              AND expires_at > NOW()
            ORDER BY created_at DESC
            LIMIT 1
            """
        )

        row = (
            await self._db.execute(
                query,
                {
                    "user_id": user_id,
                    "session_id": session_id,
                },
            )
        ).mappings().first()

        if row is None:
            return None
        return dict(row)

    async def load_pending_clarification(
        self,
        user_id: str,
        session_id: str,
        pending_id: str,
    ) -> dict[str, Any] | None:
        """Load a specific pending clarification by ID."""
        query = text(
            f"""
            SELECT
                pending_id,
                user_id,
                session_id,
                standalone_question,
                schema_context,
                relevant_schema,
                clarification_question,
                options,
                ambiguity_type,
                created_at,
                expires_at
            FROM {_PENDING_CLARIFICATIONS_TABLE}
            WHERE pending_id = :pending_id
              AND user_id = :user_id
              AND session_id = :session_id
            LIMIT 1
            """
        )

        row = (
            await self._db.execute(
                query,
                {
                    "pending_id": pending_id,
                    "user_id": user_id,
                    "session_id": session_id,
                },
            )
        ).mappings().first()

        if row is None:
            return None
        return dict(row)

    async def delete_pending_clarification(self, pending_id: str) -> None:
        query = text(
            f"""
            DELETE FROM {_PENDING_CLARIFICATIONS_TABLE}
            WHERE pending_id = :pending_id
            """
        )

        try:
            await self._db.execute(query, {"pending_id": pending_id})
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def _get_embedding_dimensions(self, table_name: str) -> int | None:
        query = text(
            """
            SELECT
                a.atttypmod AS atttypmod,
                format_type(a.atttypid, a.atttypmod) AS formatted_type
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass(:table_name)
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1
            """
        )
        try:
            result = await self._db.execute(
                query,
                {"table_name": table_name},
            )
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning(
                "Failed to read embedding dimensions for table %s: %s",
                table_name,
                exc,
            )
            return None

        row = result.mappings().first()
        if row is None:
            return None

        formatted_type = str(row.get("formatted_type") or "").strip().lower()
        vector_match = re.fullmatch(r"vector\((\d+)\)", formatted_type)
        if vector_match:
            return int(vector_match.group(1))

        raw_value = row.get("atttypmod")

        try:
            resolved = int(raw_value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    async def _get_question_rewriting_embedding_dimensions(self) -> int | None:
        return await self._get_embedding_dimensions(_QUESTION_REWRITING_TABLE)

    async def ensure_question_rewriting_episodes_table(self) -> None:
        ddl_statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS {_QUESTION_REWRITING_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                conversation TEXT NOT NULL,
                conversation_summary TEXT,
                message_count INTEGER DEFAULT 0,
                last_message_at TIMESTAMP WITH TIME ZONE,
                context_tags TEXT[],
                what_worked TEXT,
                what_to_avoid TEXT,
                source VARCHAR(100) DEFAULT 'chatbot_api',
                embedding VECTOR,
                ambiguity_metadata JSONB
            )
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS session_id VARCHAR(255)
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS conversation TEXT
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS conversation_summary TEXT
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMP WITH TIME ZONE
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS context_tags TEXT[]
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS what_worked TEXT
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS what_to_avoid TEXT
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS source VARCHAR(100) DEFAULT 'chatbot_api'
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS embedding VECTOR
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ADD COLUMN IF NOT EXISTS ambiguity_metadata JSONB
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ALTER COLUMN created_at SET DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ALTER COLUMN updated_at SET DEFAULT NOW()
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ALTER COLUMN source SET DEFAULT 'chatbot_api'
            """,
            f"""
            ALTER TABLE {_QUESTION_REWRITING_TABLE}
            ALTER COLUMN message_count SET DEFAULT 0
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET conversation = COALESCE(NULLIF(conversation, ''), conversation_summary, 'conversation')
            WHERE conversation IS NULL
               OR conversation = ''
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET created_at = COALESCE(created_at, NOW())
            WHERE created_at IS NULL
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET updated_at = COALESCE(updated_at, created_at, NOW())
            WHERE updated_at IS NULL
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET source = COALESCE(source, 'chatbot_api')
            WHERE source IS NULL
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET message_count = COALESCE(message_count, 0)
            WHERE message_count IS NULL
            """,
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET last_message_at = COALESCE(last_message_at, updated_at, created_at)
            WHERE last_message_at IS NULL
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_qre_embedding_hnsw
            ON {_QUESTION_REWRITING_TABLE}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_qre_user_session
            ON {_QUESTION_REWRITING_TABLE} (user_id, session_id)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_qre_user_id
            ON {_QUESTION_REWRITING_TABLE} (user_id)
            """,
        ]

        try:
            for statement in ddl_statements:
                await self._db.execute(text(statement))
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def upsert_question_rewriting_episode(
        self,
        user_id: str,
        session_id: str,
        conversation: str,
        conversation_summary: str,
        message_count: int,
        last_message_at: datetime | None,
        context_tags: list[str],
        what_worked: str,
        what_to_avoid: str,
        source: str,
        embedding: list[float],
        ambiguity_metadata: dict[str, Any] | None = None,
    ) -> int:
        normalized_embedding = [float(value) for value in embedding] if embedding else []
        expected_dimensions = await self._get_question_rewriting_embedding_dimensions()
        aligned_embedding = _align_embedding_dimensions(
            normalized_embedding,
            expected_dimensions,
        )
        vector_literal = (
            _embedding_to_pgvector_literal(aligned_embedding)
            if aligned_embedding
            else None
        )
        normalized_tags = [
            str(tag).strip() for tag in context_tags if str(tag).strip()
        ]
        update_query = text(
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET
                updated_at = NOW(),
                conversation = :conversation,
                conversation_summary = :conversation_summary,
                message_count = :message_count,
                last_message_at = :last_message_at,
                context_tags = CAST(:context_tags AS TEXT[]),
                what_worked = :what_worked,
                what_to_avoid = :what_to_avoid,
                source = :source,
                embedding = CAST(:embedding AS vector),
                ambiguity_metadata = COALESCE(CAST(:ambiguity_metadata AS JSONB), ambiguity_metadata)
            WHERE id = (
                SELECT id
                FROM {_QUESTION_REWRITING_TABLE}
                WHERE user_id = :user_id
                  AND session_id = :session_id
                ORDER BY id DESC
                LIMIT 1
            )
            RETURNING id
            """
        )

        insert_query = text(
            f"""
            INSERT INTO {_QUESTION_REWRITING_TABLE}
                (
                    user_id,
                    session_id,
                    conversation,
                    conversation_summary,
                    message_count,
                    last_message_at,
                    context_tags,
                    what_worked,
                    what_to_avoid,
                    source,
                    embedding,
                    ambiguity_metadata
                )
            VALUES
                (
                    :user_id,
                    :session_id,
                    :conversation,
                    :conversation_summary,
                    :message_count,
                    :last_message_at,
                    CAST(:context_tags AS TEXT[]),
                    :what_worked,
                    :what_to_avoid,
                    :source,
                    CAST(:embedding AS vector),
                    CAST(:ambiguity_metadata AS JSONB)
                )
            RETURNING id
            """
        )

        params = {
            "user_id": user_id,
            "session_id": session_id,
            "conversation": conversation,
            "conversation_summary": conversation_summary,
            "message_count": max(0, int(message_count)),
            "last_message_at": last_message_at,
            "context_tags": normalized_tags,
            "what_worked": what_worked,
            "what_to_avoid": what_to_avoid,
            "source": source or "chatbot_api",
            "embedding": vector_literal,
            "ambiguity_metadata": (
                json.dumps(ambiguity_metadata, ensure_ascii=True)
                if ambiguity_metadata is not None
                else None
            ),
        }

        try:
            updated = await self._db.execute(update_query, params)
            episode_id = updated.scalar_one_or_none()
            if episode_id is None:
                inserted = await self._db.execute(insert_query, params)
                episode_id = inserted.scalar_one()
            await self._db.commit()
            return int(episode_id)
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def retrieve_question_rewriting_episodes(
        self,
        user_id: str,
        current_session_id: str | None,
        embedding: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        normalized_embedding = [float(value) for value in embedding] if embedding else []
        if not normalized_embedding:
            return []

        expected_dimensions = await self._get_question_rewriting_embedding_dimensions()
        aligned_embedding = _align_embedding_dimensions(
            normalized_embedding,
            expected_dimensions,
        )
        if not aligned_embedding:
            return []

        vector_literal = _embedding_to_pgvector_literal(aligned_embedding)
        max_distance = 1.0 - similarity_threshold
        query = text(
            f"""
            SELECT
                id,
                user_id,
                session_id,
                conversation,
                conversation_summary,
                message_count,
                last_message_at,
                context_tags,
                what_worked,
                what_to_avoid,
                1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM {_QUESTION_REWRITING_TABLE}
            WHERE (user_id = :user_id OR user_id IS NULL)
                            AND (
                                        CAST(:current_session_id AS VARCHAR) IS NULL
                                        OR session_id <> CAST(:current_session_id AS VARCHAR)
                                    )
              AND embedding IS NOT NULL
              AND (embedding <=> CAST(:vec AS vector)) <= :max_distance
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        )

        try:
            rows = (
                await self._db.execute(
                    query,
                    {
                        "vec": vector_literal,
                        "user_id": user_id,
                        "current_session_id": current_session_id,
                        "max_distance": max_distance,
                        "k": max(1, int(top_k)),
                    },
                )
            ).mappings().all()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning(
                "Failed retrieving episodic rows for user_id=%s session_id=%s: %s",
                user_id,
                current_session_id,
                exc,
            )
            return []

        return [dict(row) for row in rows]

    async def update_latest_episode_ambiguity(
        self,
        user_id: str,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        query = text(
            f"""
            UPDATE {_QUESTION_REWRITING_TABLE}
            SET
                updated_at = NOW(),
                ambiguity_metadata = CAST(:ambiguity_metadata AS JSONB)
            WHERE id = (
                SELECT id
                FROM {_QUESTION_REWRITING_TABLE}
                WHERE user_id = :user_id
                  AND session_id = :session_id
                ORDER BY id DESC
                LIMIT 1
            )
            """
        )

        try:
            await self._db.execute(
                query,
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "ambiguity_metadata": json.dumps(metadata, ensure_ascii=True),
                },
            )
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def get_recent_ambiguity_resolutions(
        self,
        user_id: str,
        ambiguity_type: str,
        limit: int,
    ) -> list[str]:
        query = text(
            f"""
            SELECT
                ambiguity_metadata ->> 'interpretation_chosen' AS interpretation_chosen
            FROM {_QUESTION_REWRITING_TABLE}
            WHERE user_id = :user_id
              AND ambiguity_metadata IS NOT NULL
              AND ambiguity_metadata ->> 'ambiguity_type' = :ambiguity_type
              AND ambiguity_metadata ->> 'interpretation_chosen' IS NOT NULL
              AND btrim(ambiguity_metadata ->> 'interpretation_chosen') <> ''
            ORDER BY updated_at DESC, id DESC
            LIMIT :k
            """
        )

        try:
            rows = (
                await self._db.execute(
                    query,
                    {
                        "user_id": user_id,
                        "ambiguity_type": ambiguity_type,
                        "k": max(1, int(limit)),
                    },
                )
            ).mappings().all()
        except SQLAlchemyError:
            await self._db.rollback()
            return []

        candidates: list[str] = []
        for row in rows:
            value = str(row.get("interpretation_chosen") or "").strip()
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    # ------------------------------------------------------------------
    # Procedural memory (per user_id) — learned rules to bypass repeated
    # clarification cycles for semantically-equivalent vague questions.
    # ------------------------------------------------------------------

    async def _get_procedural_embedding_dimensions(self) -> int | None:
        return await self._get_embedding_dimensions(_PROCEDURAL_RULES_TABLE)

    async def ensure_procedural_rules_table(self) -> None:
        ddl_statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS {_PROCEDURAL_RULES_TABLE} (
                rule_id VARCHAR(64) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                question_pattern TEXT NOT NULL,
                question_pattern_embedding VECTOR,
                canonical_resolution TEXT NOT NULL,
                ambiguity_type VARCHAR(64),
                source_clarification_question TEXT,
                source_options JSONB,
                confidence_score REAL DEFAULT 1.0,
                hit_count INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                superseded_by VARCHAR(64),
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                archived_at TIMESTAMP WITH TIME ZONE
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_procedural_user_status
            ON {_PROCEDURAL_RULES_TABLE} (user_id, status)
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_procedural_last_used
            ON {_PROCEDURAL_RULES_TABLE} (last_used_at)
            """,
        ]
        embedding_index_statement = f"""
            CREATE INDEX IF NOT EXISTS idx_procedural_embedding_hnsw
            ON {_PROCEDURAL_RULES_TABLE}
            USING hnsw (question_pattern_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        try:
            for statement in ddl_statements:
                await self._db.execute(text(statement))

            # HNSW index requires fixed-dimension vector(n). Skip on plain vector columns.
            embedding_dimensions = await self._get_procedural_embedding_dimensions()
            if embedding_dimensions and embedding_dimensions > 0:
                await self._db.execute(text(embedding_index_statement))

            await self._db.commit()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to ensure procedural rules table: %s", exc)

    async def find_matching_procedural_rule(
        self,
        user_id: str,
        embedding: list[float],
        similarity_threshold: float,
        ttl_days: int,
    ) -> dict[str, Any] | None:
        """Find best active procedural rule for user_id whose pattern embedding
        is within the cosine similarity threshold and not stale (last_used_at
        within ttl_days). Returns None when nothing qualifies.
        """
        if not embedding or not user_id:
            return None

        expected_dimensions = await self._get_procedural_embedding_dimensions()
        aligned = _align_embedding_dimensions(
            [float(v) for v in embedding], expected_dimensions
        )
        if not aligned:
            return None

        vector_literal = _embedding_to_pgvector_literal(aligned)
        max_distance = max(0.0, 1.0 - float(similarity_threshold))

        query = text(
            f"""
            SELECT
                rule_id,
                question_pattern,
                canonical_resolution,
                ambiguity_type,
                hit_count,
                version,
                confidence_score,
                last_used_at,
                1 - (question_pattern_embedding <=> CAST(:embedding AS vector))
                    AS similarity
            FROM {_PROCEDURAL_RULES_TABLE}
            WHERE user_id = :user_id
              AND status = 'active'
              AND question_pattern_embedding IS NOT NULL
              AND last_used_at >= NOW() - make_interval(days => :ttl_days)
              AND (question_pattern_embedding <=> CAST(:embedding AS vector))
                  <= :max_distance
            ORDER BY question_pattern_embedding <=> CAST(:embedding AS vector) ASC
            LIMIT 1
            """
        )
        try:
            row = (
                await self._db.execute(
                    query,
                    {
                        "user_id": user_id,
                        "embedding": vector_literal,
                        "max_distance": max_distance,
                        "ttl_days": int(max(1, ttl_days)),
                    },
                )
            ).mappings().first()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.exception("Failed to query procedural rule: %s", exc)
            return None
        log.info(
            "procedural_match: user_id=%s threshold=%s max_distance=%s -> %s",
            user_id,
            similarity_threshold,
            max_distance,
            (
                f"hit rule_id={row.get('rule_id')} similarity={row.get('similarity'):.4f}"
                if row is not None
                else "miss"
            ),
        )

        if row is None:
            return None
        return dict(row)

    async def record_procedural_rule_hit(self, rule_id: str) -> None:
        if not rule_id:
            return
        query = text(
            f"""
            UPDATE {_PROCEDURAL_RULES_TABLE}
            SET hit_count = COALESCE(hit_count, 0) + 1,
                last_used_at = NOW(),
                confidence_score = LEAST(1.0, COALESCE(confidence_score, 1.0) + 0.01)
            WHERE rule_id = :rule_id
            """
        )
        try:
            await self._db.execute(query, {"rule_id": rule_id})
            await self._db.commit()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to record procedural rule hit: %s", exc)

    async def find_existing_rule_for_pattern(
        self,
        user_id: str,
        embedding: list[float],
        similarity_threshold: float,
    ) -> dict[str, Any] | None:
        """Same as find_matching but scoped only to active rules and ignores
        TTL — used when deciding to supersede vs. insert a fresh rule.
        """
        if not embedding or not user_id:
            return None
        expected_dimensions = await self._get_procedural_embedding_dimensions()
        aligned = _align_embedding_dimensions(
            [float(v) for v in embedding], expected_dimensions
        )
        if not aligned:
            return None

        vector_literal = _embedding_to_pgvector_literal(aligned)
        max_distance = max(0.0, 1.0 - float(similarity_threshold))

        query = text(
            f"""
            SELECT rule_id, version, canonical_resolution
            FROM {_PROCEDURAL_RULES_TABLE}
            WHERE user_id = :user_id
              AND status = 'active'
              AND question_pattern_embedding IS NOT NULL
              AND (question_pattern_embedding <=> CAST(:embedding AS vector))
                  <= :max_distance
            ORDER BY question_pattern_embedding <=> CAST(:embedding AS vector) ASC
            LIMIT 1
            """
        )
        try:
            row = (
                await self._db.execute(
                    query,
                    {
                        "user_id": user_id,
                        "embedding": vector_literal,
                        "max_distance": max_distance,
                    },
                )
            ).mappings().first()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to query existing rule: %s", exc)
            return None
        return dict(row) if row else None

    async def insert_procedural_rule(
        self,
        user_id: str,
        question_pattern: str,
        embedding: list[float],
        canonical_resolution: str,
        ambiguity_type: str | None,
        source_clarification_question: str | None,
        source_options: list[str] | None,
        version: int = 1,
    ) -> str | None:
        """Insert a new procedural rule. Returns rule_id, or None on failure."""
        if not user_id or not question_pattern.strip() or not canonical_resolution.strip():
            return None
        if not embedding:
            return None

        expected_dimensions = await self._get_procedural_embedding_dimensions()
        aligned = _align_embedding_dimensions(
            [float(v) for v in embedding], expected_dimensions
        )
        if not aligned:
            return None

        rule_id = str(uuid4())
        vector_literal = _embedding_to_pgvector_literal(aligned)
        options_payload = json.dumps(
            [str(opt) for opt in (source_options or [])], ensure_ascii=True
        )

        query = text(
            f"""
            INSERT INTO {_PROCEDURAL_RULES_TABLE} (
                rule_id, user_id, question_pattern, question_pattern_embedding,
                canonical_resolution, ambiguity_type,
                source_clarification_question, source_options,
                confidence_score, hit_count, version, status,
                created_at, last_used_at
            ) VALUES (
                :rule_id, :user_id, :question_pattern, CAST(:embedding AS vector),
                :canonical_resolution, :ambiguity_type,
                :source_clarification_question, CAST(:source_options AS JSONB),
                1.0, 0, :version, 'active',
                NOW(), NOW()
            )
            """
        )
        try:
            await self._db.execute(
                query,
                {
                    "rule_id": rule_id,
                    "user_id": user_id,
                    "question_pattern": question_pattern.strip(),
                    "embedding": vector_literal,
                    "canonical_resolution": canonical_resolution.strip(),
                    "ambiguity_type": ambiguity_type,
                    "source_clarification_question": source_clarification_question,
                    "source_options": options_payload,
                    "version": int(max(1, version)),
                },
            )
            await self._db.commit()
            return rule_id
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to insert procedural rule: %s", exc)
            return None

    async def supersede_procedural_rule(
        self, old_rule_id: str, new_rule_id: str
    ) -> None:
        if not old_rule_id or not new_rule_id:
            return
        query = text(
            f"""
            UPDATE {_PROCEDURAL_RULES_TABLE}
            SET status = 'superseded',
                superseded_by = :new_rule_id,
                archived_at = NOW()
            WHERE rule_id = :old_rule_id
            """
        )
        try:
            await self._db.execute(
                query, {"old_rule_id": old_rule_id, "new_rule_id": new_rule_id}
            )
            await self._db.commit()
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to supersede procedural rule: %s", exc)

    async def archive_procedural_rules_by_pattern(
        self,
        user_id: str,
        embedding: list[float],
        similarity_threshold: float,
    ) -> int:
        """Archive all active rules for user_id whose pattern is similar to
        the supplied embedding. Used by reset-preference command.
        Returns count archived.
        """
        if not embedding or not user_id:
            return 0
        expected_dimensions = await self._get_procedural_embedding_dimensions()
        aligned = _align_embedding_dimensions(
            [float(v) for v in embedding], expected_dimensions
        )
        if not aligned:
            return 0

        vector_literal = _embedding_to_pgvector_literal(aligned)
        max_distance = max(0.0, 1.0 - float(similarity_threshold))

        query = text(
            f"""
            UPDATE {_PROCEDURAL_RULES_TABLE}
            SET status = 'archived', archived_at = NOW()
            WHERE user_id = :user_id
              AND status = 'active'
              AND question_pattern_embedding IS NOT NULL
              AND (question_pattern_embedding <=> CAST(:embedding AS vector))
                  <= :max_distance
            """
        )
        try:
            result = await self._db.execute(
                query,
                {
                    "user_id": user_id,
                    "embedding": vector_literal,
                    "max_distance": max_distance,
                },
            )
            await self._db.commit()
            return int(result.rowcount or 0)
        except SQLAlchemyError as exc:
            await self._db.rollback()
            log.warning("Failed to archive procedural rules: %s", exc)
            return 0

    async def is_vector_table_available(self, vector_table: str) -> bool:
        query = text("SELECT to_regclass(:table_name)")
        result = await self._db.execute(query, {"table_name": vector_table})
        return result.scalar_one_or_none() is not None

    async def load_schema(
        self, allowed_tables: dict[str, list[str]]
    ) -> list[dict[str, Any]]:
        if not allowed_tables:
            return []

        schema_params = {
            f"schema_{index}": schema_name
            for index, schema_name in enumerate(allowed_tables.keys())
        }
        schema_placeholders = ", ".join(f":{key}" for key in schema_params)
        query = text(
            f"""
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.udt_name,
                CASE
                    WHEN c.data_type = 'USER-DEFINED' THEN (
                        SELECT string_agg(e.enumlabel, ', ' ORDER BY e.enumsortorder)
                        FROM pg_type t
                        JOIN pg_enum e ON t.oid = e.enumtypid
                        WHERE t.typname = c.udt_name
                    )
                    ELSE NULL
                END AS enum_values
            FROM information_schema.columns c
            WHERE c.table_schema IN ({schema_placeholders})
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """
        )
        rows = (await self._db.execute(query, schema_params)).mappings().all()

        table_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            schema_name = str(row["table_schema"])
            table_name = str(row["table_name"])
            if table_name not in allowed_tables.get(schema_name, []):
                continue

            table_key = f"{schema_name}.{table_name}"
            if table_key not in table_map:
                table_map[table_key] = {
                    "schema": schema_name,
                    "name": table_name,
                    "columns": [],
                }

            enum_values_raw = row["enum_values"]
            column_info: dict[str, Any] = {
                "name": str(row["column_name"]),
                "type": str(row["data_type"]),
                "udt_name": str(row["udt_name"]),
            }
            if enum_values_raw:
                column_info["enum_values"] = str(enum_values_raw).split(", ")
            table_map[table_key]["columns"].append(column_info)

        return list(table_map.values())

    async def retrieve_entities_by_vector(
        self,
        vector_table: str,
        vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not vector:
            return []

        normalized_vector = [float(value) for value in vector]
        expected_dimensions = await self._get_embedding_dimensions(vector_table)
        aligned_vector = _align_embedding_dimensions(
            normalized_vector,
            expected_dimensions,
        )
        if not aligned_vector:
            return []

        table_ref = _safe_table_reference(vector_table)
        vector_literal = _embedding_to_pgvector_literal(aligned_vector)

        query = text(
            f"""
            SELECT
                id,
                entity_type,
                schema_name,
                table_name,
                column_name,
                content,
                1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM {table_ref}
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        )

        rows = (
            await self._db.execute(query, {"vec": vector_literal, "k": int(top_k)})
        ).mappings().all()
        return [dict(row) for row in rows]

    async def get_vector_table_embedding_dimensions(self, vector_table: str) -> int | None:
        _safe_table_reference(vector_table)
        return await self._get_embedding_dimensions(vector_table)

    async def get_vector_table_columns(self, vector_table: str) -> set[str]:
        schema_name, table_name = _split_table_reference(vector_table)
        query = text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema_name
              AND table_name = :table_name
            """
        )
        rows = (
            await self._db.execute(
                query,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            )
        ).scalars().all()
        return {str(row) for row in rows}

    async def ensure_base_knowledge_vector_table(self, vector_table: str) -> None:
        table_ref = _safe_table_reference(vector_table)
        embedding_index = _build_index_name("idx_bk_embedding", vector_table)
        entity_index = _build_index_name("idx_bk_entity", vector_table)

        ddl_statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS {table_ref} (
                id BIGSERIAL PRIMARY KEY,
                entity_type VARCHAR(50),
                schema_name VARCHAR(100),
                table_name VARCHAR(255),
                table_description TEXT,
                column_name VARCHAR(255),
                column_alias VARCHAR(255),
                column_description TEXT,
                value_description TEXT,
                content TEXT NOT NULL,
                embedding VECTOR
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS {entity_index}
            ON {table_ref} (entity_type, schema_name, table_name)
            """,
        ]

        embedding_index_statement = f"""
            CREATE INDEX IF NOT EXISTS {embedding_index}
            ON {table_ref}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """

        try:
            for statement in ddl_statements:
                await self._db.execute(text(statement))

            # HNSW index requires fixed-dimension vector(n). Skip on plain vector columns.
            embedding_dimensions = await self._get_embedding_dimensions(vector_table)
            if embedding_dimensions and embedding_dimensions > 0:
                await self._db.execute(text(embedding_index_statement))

            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def truncate_vector_table(self, vector_table: str) -> None:
        table_ref = _safe_table_reference(vector_table)
        query = text(f"TRUNCATE TABLE {table_ref} RESTART IDENTITY")
        try:
            await self._db.execute(query)
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

    async def count_vector_rows_with_content(self, vector_table: str) -> int:
        table_ref = _safe_table_reference(vector_table)
        query = text(
            f"""
            SELECT COUNT(*)
            FROM {table_ref}
            WHERE content IS NOT NULL
              AND btrim(content) <> ''
            """
        )

        result = await self._db.execute(query)
        return int(result.scalar_one_or_none() or 0)

    async def load_vector_rows_with_content(
        self,
        vector_table: str,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        table_ref = _safe_table_reference(vector_table)
        query = text(
            f"""
            SELECT id, content
            FROM {table_ref}
            WHERE content IS NOT NULL
              AND btrim(content) <> ''
            ORDER BY id
            LIMIT :limit OFFSET :offset
            """
        )

        rows = (
            await self._db.execute(
                query,
                {
                    "limit": max(1, int(limit)),
                    "offset": max(0, int(offset)),
                },
            )
        ).mappings().all()
        return [dict(row) for row in rows]

    async def update_vector_embeddings_by_id(
        self,
        vector_table: str,
        embeddings_by_id: list[tuple[Any, list[float]]],
        expected_dimensions: int | None = None,
    ) -> int:
        if not embeddings_by_id:
            return 0

        table_ref = _safe_table_reference(vector_table)
        params: list[dict[str, Any]] = []
        for row_id, embedding in embeddings_by_id:
            if row_id is None or not embedding:
                continue

            normalized_embedding = [float(value) for value in embedding]
            aligned_embedding = _align_embedding_dimensions(
                normalized_embedding,
                expected_dimensions,
            )
            if not aligned_embedding:
                continue

            params.append(
                {
                    "row_id": row_id,
                    "embedding": _embedding_to_pgvector_literal(aligned_embedding),
                }
            )

        if not params:
            return 0

        query = text(
            f"""
            UPDATE {table_ref}
            SET embedding = CAST(:embedding AS vector)
            WHERE id = :row_id
            """
        )

        try:
            await self._db.execute(query, params)
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

        return len(params)

    async def insert_base_knowledge_rows(
        self,
        vector_table: str,
        rows: list[dict[str, Any]],
        expected_dimensions: int | None = None,
    ) -> int:
        if not rows:
            return 0

        table_ref = _safe_table_reference(vector_table)
        table_columns = await self.get_vector_table_columns(vector_table)
        candidate_columns = [
            "entity_type",
            "schema_name",
            "table_name",
            "table_description",
            "column_name",
            "column_alias",
            "column_description",
            "value_description",
            "content",
            "embedding",
        ]
        insert_columns = [
            column_name
            for column_name in candidate_columns
            if column_name in table_columns
        ]

        if "content" not in insert_columns or "embedding" not in insert_columns:
            raise ValueError(
                "Vector table must have at least content and embedding columns"
            )

        params: list[dict[str, Any]] = []
        for row in rows:
            content = str(row.get("content") or "").strip()
            embedding = row.get("embedding")
            if not content or not embedding:
                continue

            normalized_embedding = [float(value) for value in embedding]
            aligned_embedding = _align_embedding_dimensions(
                normalized_embedding,
                expected_dimensions,
            )
            if not aligned_embedding:
                continue

            row_params: dict[str, Any] = {}
            for column_name in insert_columns:
                if column_name == "content":
                    row_params[column_name] = content
                elif column_name == "embedding":
                    row_params[column_name] = _embedding_to_pgvector_literal(
                        aligned_embedding
                    )
                else:
                    value = row.get(column_name)
                    if isinstance(value, str):
                        stripped = value.strip()
                        row_params[column_name] = stripped or None
                    else:
                        row_params[column_name] = value

            params.append(row_params)

        if not params:
            return 0

        column_list_sql = ", ".join(insert_columns)
        value_list_sql = ", ".join(
            [
                f"CAST(:{column_name} AS vector)"
                if column_name == "embedding"
                else f":{column_name}"
                for column_name in insert_columns
            ]
        )

        query = text(
            f"""
            INSERT INTO {table_ref} ({column_list_sql})
            VALUES ({value_list_sql})
            """
        )

        try:
            await self._db.execute(query, params)
            await self._db.commit()
        except SQLAlchemyError:
            await self._db.rollback()
            raise

        return len(params)

    async def load_table_descriptions(self, vector_table: str) -> dict[str, str]:
        table_ref = _safe_table_reference(vector_table)
        query = text(
            f"""
            SELECT schema_name, table_name, table_description
            FROM {table_ref}
            WHERE entity_type = 'table'
            """
        )
        rows = (await self._db.execute(query)).mappings().all()
        descriptions: dict[str, str] = {}
        for row in rows:
            key = f"{row['schema_name']}.{row['table_name']}"
            descriptions[key] = str(row["table_description"] or "")
        return descriptions

    async def load_column_samples(
        self,
        schema_tables: list[dict[str, Any]],
        n_samples: int,
    ) -> dict[tuple[str, str, str], list[Any]]:
        samples: dict[tuple[str, str, str], list[Any]] = {}

        for table in schema_tables:
            schema_name = str(table["schema"])
            table_name = str(table["name"])
            sql_ref = f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"
            query = text(f"SELECT * FROM {sql_ref} LIMIT :n")

            try:
                result = await self._db.execute(query, {"n": int(n_samples)})
            except SQLAlchemyError:
                continue

            rows = result.mappings().all()
            if not rows:
                continue

            for column_name in rows[0].keys():
                key = (schema_name, table_name, str(column_name))
                values: list[Any] = []
                for row in rows:
                    value = row[column_name]
                    if value is None:
                        continue
                    if value not in values:
                        values.append(value)
                    if len(values) >= n_samples:
                        break
                if values:
                    samples[key] = values

        return samples

    async def execute_sql(
        self, sql: str, timeout_ms: int
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        try:
            await self._db.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
            result = await self._db.execute(text(sql))
            rows = [dict(row) for row in result.mappings().all()]
            return rows, None
        except SQLAlchemyError as exc:
            await self._db.rollback()
            return None, str(exc)
