"""Add compact episodic metadata columns

Revision ID: 004
Revises: 003
Create Date: 2026-04-15

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, Sequence[str], None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "question_rewriting_episodes"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"""
        ALTER TABLE IF EXISTS {_TABLE}
        ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0
        """
    )
    op.execute(
        f"""
        ALTER TABLE IF EXISTS {_TABLE}
        ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMP WITH TIME ZONE
        """
    )
    op.execute(
        f"""
        ALTER TABLE IF EXISTS {_TABLE}
        ALTER COLUMN message_count SET DEFAULT 0
        """
    )

    op.execute(
        f"""
        UPDATE {_TABLE}
        SET conversation = COALESCE(NULLIF(conversation, ''), conversation_summary, 'conversation')
        WHERE conversation IS NULL
           OR conversation = ''
        """
    )
    op.execute(
        f"""
        UPDATE {_TABLE}
        SET message_count = COALESCE(message_count, 0)
        WHERE message_count IS NULL
        """
    )
    op.execute(
        f"""
        UPDATE {_TABLE}
        SET last_message_at = COALESCE(last_message_at, updated_at, created_at)
        WHERE last_message_at IS NULL
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        f"""
        ALTER TABLE IF EXISTS {_TABLE}
        DROP COLUMN IF EXISTS last_message_at
        """
    )
    op.execute(
        f"""
        ALTER TABLE IF EXISTS {_TABLE}
        DROP COLUMN IF EXISTS message_count
        """
    )
