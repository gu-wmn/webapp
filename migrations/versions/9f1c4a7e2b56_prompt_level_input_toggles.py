"""prompt-level input toggles, remove experiment regex fields, add input instructions

Revision ID: 9f1c4a7e2b56
Revises: d3f7a291b4e0
Create Date: 2026-08-18

"""
from alembic import op
import sqlalchemy as sa


revision = '9f1c4a7e2b56'
down_revision = 'd3f7a291b4e0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('include_dialogue', sa.Boolean(), nullable=False, server_default=sa.true())
        )
        batch_op.add_column(
            sa.Column('include_regex_candidates', sa.Boolean(), nullable=False, server_default=sa.false())
        )

    with op.batch_alter_table('user_settings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dialogue_input_instructions', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('regex_input_instructions', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('previous_output_instructions', sa.Text(), nullable=True))

    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.drop_column('regex_patterns')
        batch_op.drop_column('regex_enabled')


def downgrade():
    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.add_column(sa.Column('regex_enabled', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('regex_patterns', sa.Text(), nullable=True))

    with op.batch_alter_table('user_settings', schema=None) as batch_op:
        batch_op.drop_column('previous_output_instructions')
        batch_op.drop_column('regex_input_instructions')
        batch_op.drop_column('dialogue_input_instructions')

    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.drop_column('include_regex_candidates')
        batch_op.drop_column('include_dialogue')
