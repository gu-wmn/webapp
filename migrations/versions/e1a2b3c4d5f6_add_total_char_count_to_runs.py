"""add total_char_count to runs

Revision ID: e1a2b3c4d5f6
Revises: c7d24f8b1a63
Create Date: 2026-08-25

"""
from alembic import op
import sqlalchemy as sa


revision = 'e1a2b3c4d5f6'
down_revision = 'c7d24f8b1a63'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('total_char_count', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_column('total_char_count')
