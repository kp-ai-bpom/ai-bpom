"""Bridge chatbot migration chain

Revision ID: 001
Revises: 76c30486c443
Create Date: 2026-04-22

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, Sequence[str], None] = "76c30486c443"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # This revision intentionally bridges historical migration chains.
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
