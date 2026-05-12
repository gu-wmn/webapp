"""add regex stage to experiments

Revision ID: 4a1f9c3e82b0
Revises: 831d44d2e7ae
Create Date: 2026-05-12

"""
from alembic import op
import sqlalchemy as sa

revision = '4a1f9c3e82b0'
down_revision = '831d44d2e7ae'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('experiments', sa.Column('regex_enabled', sa.Boolean(), nullable=False, server_default='0'))
    op.add_column('experiments', sa.Column('regex_patterns', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('experiments', 'regex_patterns')
    op.drop_column('experiments', 'regex_enabled')
