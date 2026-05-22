"""add free_text_appendix to user_settings

Revision ID: d3f7a291b4e0
Revises: 831d44d2e7ae
Create Date: 2026-05-22 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd3f7a291b4e0'
down_revision = '2c9c3f8f1d11'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('user_settings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('free_text_appendix', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('user_settings', schema=None) as batch_op:
        batch_op.drop_column('free_text_appendix')
