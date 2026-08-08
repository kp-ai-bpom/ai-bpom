"""Create chat sessions and messages tables

Revision ID: 003
Revises: 002
Create Date: 2026-04-15

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, Sequence[str], None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SESSIONS_TABLE = "chat_sessions"
_MESSAGES_TABLE = "chat_messages"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_SESSIONS_TABLE} (
            session_id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            title VARCHAR(255) NOT NULL,
            title_source VARCHAR(50) DEFAULT 'first_user_message',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
    )

    columns: list[tuple[str, str]] = [
        ("user_id", "VARCHAR(255)"),
        ("title", "VARCHAR(255)"),
        ("title_source", "VARCHAR(50) DEFAULT 'first_user_message'"),
        ("created_at", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
        ("updated_at", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
    ]

    for column_name, column_type in columns:
        op.execute(
            f"""
            ALTER TABLE {_SESSIONS_TABLE}
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """
        )

    op.execute(
        f"""
        ALTER TABLE {_SESSIONS_TABLE}
        ALTER COLUMN created_at SET DEFAULT NOW()
        """
    )
    op.execute(
        f"""
        ALTER TABLE {_SESSIONS_TABLE}
        ALTER COLUMN updated_at SET DEFAULT NOW()
        """
    )

    op.execute(
        f"""
        UPDATE {_SESSIONS_TABLE}
        SET
            title = COALESCE(NULLIF(title, ''), LEFT(session_id, 80)),
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
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
        ON {_SESSIONS_TABLE} (user_id, updated_at DESC)
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
        ON {_SESSIONS_TABLE} (user_id)
        """
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_MESSAGES_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL REFERENCES {_SESSIONS_TABLE}(session_id) ON DELETE CASCADE,
            question TEXT NOT NULL,
            standalone_question TEXT,
            query TEXT NOT NULL,
            explanation TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
    )

    message_columns: list[tuple[str, str]] = [
        ("session_id", "VARCHAR(255)"),
        ("question", "TEXT"),
        ("standalone_question", "TEXT"),
        ("query", "TEXT"),
        ("explanation", "TEXT"),
        ("created_at", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
    ]

    for column_name, column_type in message_columns:
        op.execute(
            f"""
            ALTER TABLE {_MESSAGES_TABLE}
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """
        )

    op.execute(
        f"""
        ALTER TABLE {_MESSAGES_TABLE}
        ALTER COLUMN created_at SET DEFAULT NOW()
        """
    )

    op.execute(
        f"""
        UPDATE {_MESSAGES_TABLE}
        SET created_at = COALESCE(created_at, NOW())
        WHERE created_at IS NULL
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
        ON {_MESSAGES_TABLE} (session_id, created_at, id)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_chat_messages_session_created")
    op.execute("DROP TABLE IF EXISTS chat_messages")
    op.execute("DROP INDEX IF EXISTS idx_chat_sessions_user")
    op.execute("DROP INDEX IF EXISTS idx_chat_sessions_user_updated")
    op.execute("DROP TABLE IF EXISTS chat_sessions")
