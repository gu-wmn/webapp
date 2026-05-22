"""add include_global_template to prompts

Revision ID: 2c9c3f8f1d11
Revises: 7c3d1e6a0f52
Create Date: 2026-05-22

"""
from alembic import op
import sqlalchemy as sa


revision = '2c9c3f8f1d11'
down_revision = '7c3d1e6a0f52'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'include_global_template',
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )


def downgrade():
    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.drop_column('include_global_template')
