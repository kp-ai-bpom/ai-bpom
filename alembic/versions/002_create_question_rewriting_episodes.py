"""Create question rewriting episodes table

Revision ID: 002
Revises: 001
Create Date: 2026-04-15

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "question_rewriting_episodes"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            conversation TEXT NOT NULL,
            conversation_summary TEXT,
            context_tags TEXT[],
            what_worked TEXT,
            what_to_avoid TEXT,
            source VARCHAR(100) DEFAULT 'chatbot_api',
            embedding VECTOR
        )
        """
    )

    columns: list[tuple[str, str]] = [
        ("user_id", "VARCHAR(255)"),
        ("session_id", "VARCHAR(255)"),
        ("created_at", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
        ("updated_at", "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
        ("conversation", "TEXT"),
        ("conversation_summary", "TEXT"),
        ("context_tags", "TEXT[]"),
        ("what_worked", "TEXT"),
        ("what_to_avoid", "TEXT"),
        ("source", "VARCHAR(100) DEFAULT 'chatbot_api'"),
        ("embedding", "VECTOR"),
    ]

    for column_name, column_type in columns:
        op.execute(
            f"""
            ALTER TABLE {_TABLE}
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """
        )

    op.execute(
        f"""
        ALTER TABLE {_TABLE}
        ALTER COLUMN created_at SET DEFAULT NOW()
        """
    )
    op.execute(
        f"""
        ALTER TABLE {_TABLE}
        ALTER COLUMN updated_at SET DEFAULT NOW()
        """
    )
    op.execute(
        f"""
        ALTER TABLE {_TABLE}
        ALTER COLUMN source SET DEFAULT 'chatbot_api'
        """
    )

    op.execute(
        f"""
        UPDATE {_TABLE}
        SET created_at = COALESCE(created_at, NOW())
        WHERE created_at IS NULL
        """
    )
    op.execute(
        f"""
        UPDATE {_TABLE}
        SET updated_at = COALESCE(updated_at, created_at, NOW())
        WHERE updated_at IS NULL
        """
    )
    op.execute(
        f"""
        UPDATE {_TABLE}
        SET source = COALESCE(source, 'chatbot_api')
        WHERE source IS NULL
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_qre_embedding_hnsw
        ON {_TABLE}
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_qre_user_session
        ON {_TABLE} (user_id, session_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_qre_user_id
        ON {_TABLE} (user_id)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_qre_user_id")
    op.execute("DROP INDEX IF EXISTS idx_qre_user_session")
    op.execute("DROP INDEX IF EXISTS idx_qre_embedding_hnsw")
    op.execute(f"DROP TABLE IF EXISTS {_TABLE}")
