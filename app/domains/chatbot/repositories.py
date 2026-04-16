import re
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import log


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUESTION_REWRITING_TABLE = "question_rewriting_episodes"
_CHAT_SESSIONS_TABLE = "chat_sessions"
_CHAT_MESSAGES_TABLE = "chat_messages"


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
                (session_id, question, standalone_question, query, explanation, created_at)
            VALUES
                (:session_id, :question, :standalone_question, :query, :explanation, NOW())
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
            SELECT question, standalone_question, query, explanation, created_at
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
                embedding VECTOR
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
                embedding = CAST(:embedding AS vector)
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
                    embedding
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
                    CAST(:embedding AS vector)
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
