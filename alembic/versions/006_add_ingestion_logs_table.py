"""Add pemetaan_ingestion_logs table

Revision ID: 006
Revises: 005
Create Date: 2026-05-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b00000000001'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'pemetaan_ingestion_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('entity_count', sa.Integer(), nullable=True),
        sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_pemetaan_ingestion_logs_filename', 'pemetaan_ingestion_logs', ['filename'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_pemetaan_ingestion_logs_filename', table_name='pemetaan_ingestion_logs')
    op.drop_table('pemetaan_ingestion_logs')
