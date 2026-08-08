"""feat penilaian makalah

Revision ID: 6777772db000
Revises: 5f20b3f7c939
Create Date: 2026-06-27 09:14:13.542757

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6777772db000'
down_revision: Union[str, Sequence[str], None] = '5f20b3f7c939'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
