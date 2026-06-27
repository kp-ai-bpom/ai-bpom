"""feat penilaian makalah

Revision ID: 5f20b3f7c939
Revises: 007, a1b2c3d4e5f6
Create Date: 2026-06-27 09:03:04.528941

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f20b3f7c939'
down_revision: Union[str, Sequence[str], None] = ('007', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
