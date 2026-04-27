"""Add user procedural rules (procedural memory layer)

Revision ID: 007
Revises: 006
Create Date: 2026-04-23

"""

from typing import Sequence, Union

from alembic import op


revision: str = "007"
down_revision: Union[str, Sequence[str], None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PROCEDURAL_TABLE = "chat_user_procedural_rules"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PROCEDURAL_TABLE} (
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
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_procedural_user_status
        ON {_PROCEDURAL_TABLE} (user_id, status)
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_procedural_last_used
        ON {_PROCEDURAL_TABLE} (last_used_at)
        """
    )

    # NOTE: HNSW index requires fixed-dimension vector(n). Column is created as
    # plain VECTOR here; the runtime ensure_procedural_rules_table() will create
    # the HNSW index lazily once the embedding dimension is known.


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_procedural_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS idx_procedural_last_used")
    op.execute("DROP INDEX IF EXISTS idx_procedural_user_status")
    op.execute(f"DROP TABLE IF EXISTS {_PROCEDURAL_TABLE}")
