"""Expand effective session title handling and backfill truncated titles

Revision ID: 005
Revises: 004
Create Date: 2026-04-16

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, Sequence[str], None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SESSIONS_TABLE = "chat_sessions"
_MESSAGES_TABLE = "chat_messages"


def upgrade() -> None:
    """Upgrade schema."""
    # Keep title within existing varchar(255) while removing old 80-char truncation behavior.
    op.execute(
        f"""
        UPDATE {_SESSIONS_TABLE}
        SET title = LEFT(session_id, 255)
        WHERE title IS NULL
           OR btrim(title) = ''
        """
    )

    # Backfill only rows that match the previous 80-char truncation pattern.
    op.execute(
        f"""
        WITH first_messages AS (
            SELECT DISTINCT ON (m.session_id)
                m.session_id,
                m.question
            FROM {_MESSAGES_TABLE} m
            WHERE m.question IS NOT NULL
              AND btrim(m.question) <> ''
            ORDER BY m.session_id, m.created_at, m.id
        )
        UPDATE {_SESSIONS_TABLE} s
        SET title = LEFT(f.question, 255)
        FROM first_messages f
        WHERE s.session_id = f.session_id
          AND s.title_source = 'first_user_message'
          AND char_length(f.question) > 80
          AND s.title = LEFT(f.question, 80)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    # No safe lossless downgrade for data backfill.
    pass
