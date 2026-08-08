"""Add chatbot ambiguity handling persistence

Revision ID: 006
Revises: 005
Create Date: 2026-04-22

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, Sequence[str], None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PENDING_TABLE = "chat_pending_clarifications"
_QRE_TABLE = "question_rewriting_episodes"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PENDING_TABLE} (
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
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_pending_user_session
        ON {_PENDING_TABLE} (user_id, session_id)
        """
    )

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_chat_pending_expires
        ON {_PENDING_TABLE} (expires_at)
        """
    )

    op.execute(
        f"""
        ALTER TABLE {_QRE_TABLE}
        ADD COLUMN IF NOT EXISTS ambiguity_metadata JSONB
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(f"DROP INDEX IF EXISTS idx_chat_pending_expires")
    op.execute(f"DROP INDEX IF EXISTS idx_chat_pending_user_session")
    op.execute(f"DROP TABLE IF EXISTS {_PENDING_TABLE}")
    op.execute(f"ALTER TABLE {_QRE_TABLE} DROP COLUMN IF EXISTS ambiguity_metadata")
