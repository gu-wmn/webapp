"""add dialogue_char_count and duration_seconds to run_results and regex_run_results

Revision ID: b3e7a1c9d5f0
Revises: 9f1c4a7e2b56
Create Date: 2026-08-18

"""
from alembic import op
import sqlalchemy as sa


revision = 'b3e7a1c9d5f0'
down_revision = '9f1c4a7e2b56'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('run_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dialogue_char_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('duration_seconds', sa.Float(), nullable=True))

    with op.batch_alter_table('regex_run_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dialogue_char_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('duration_seconds', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('regex_run_results', schema=None) as batch_op:
        batch_op.drop_column('duration_seconds')
        batch_op.drop_column('dialogue_char_count')

    with op.batch_alter_table('run_results', schema=None) as batch_op:
        batch_op.drop_column('duration_seconds')
        batch_op.drop_column('dialogue_char_count')
