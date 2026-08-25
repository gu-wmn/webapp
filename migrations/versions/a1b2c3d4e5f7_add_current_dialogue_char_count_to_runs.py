"""add current_dialogue_char_count to runs

Revision ID: a1b2c3d4e5f7
Revises: f2b3c4d5e6a7
Create Date: 2026-08-25

"""
from alembic import op
import sqlalchemy as sa


revision = 'a1b2c3d4e5f7'
down_revision = 'f2b3c4d5e6a7'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('current_dialogue_char_count', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_column('current_dialogue_char_count')
