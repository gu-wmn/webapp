"""add last_error to runs

Revision ID: c7d24f8b1a63
Revises: b3e7a1c9d5f0
Create Date: 2026-08-19

"""
from alembic import op
import sqlalchemy as sa


revision = 'c7d24f8b1a63'
down_revision = 'b3e7a1c9d5f0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('last_error', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_column('last_error')
