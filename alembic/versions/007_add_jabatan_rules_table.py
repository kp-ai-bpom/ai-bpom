"""add jabatan_rules table

Revision ID: 007
Revises: 006
Create Date: 2026-06-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'b00000000002'
down_revision: Union[str, Sequence[str], None] = 'b00000000001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'jabatan_rules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('slug', sa.String(length=255), nullable=False),
        sa.Column('nama_jabatan', sa.String(length=255), nullable=False),
        sa.Column('atasan_langsung', sa.String(length=255), server_default=sa.text("''"), nullable=False),
        sa.Column('data', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_jabatan_rules_slug', 'jabatan_rules', ['slug'], unique=True)
    op.create_index('ix_jabatan_rules_nama_jabatan', 'jabatan_rules', ['nama_jabatan'], unique=False)
    op.execute("CREATE INDEX IF NOT EXISTS ix_jabatan_rules_data ON jabatan_rules USING GIN (data)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_jabatan_rules_data")
    op.drop_index('ix_jabatan_rules_nama_jabatan', table_name='jabatan_rules')
    op.drop_index('ix_jabatan_rules_slug', table_name='jabatan_rules')
    op.drop_table('jabatan_rules')
