"""add regex run tables

Revision ID: 7c3d1e6a0f52
Revises: 4a1f9c3e82b0
Create Date: 2026-05-12

"""
from alembic import op
import sqlalchemy as sa

revision = '7c3d1e6a0f52'
down_revision = '4a1f9c3e82b0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'regex_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('total_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('processed_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_regex_runs_experiment_id', 'regex_runs', ['experiment_id'])

    op.create_table(
        'regex_run_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('regex_run_id', sa.Integer(), nullable=False),
        sa.Column('dialogue_external_id', sa.String(), nullable=False),
        sa.Column('corpus_codename', sa.String(), nullable=False),
        sa.Column('output', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['regex_run_id'], ['regex_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_regex_run_results_regex_run_id', 'regex_run_results', ['regex_run_id'])


def downgrade():
    op.drop_index('ix_regex_run_results_regex_run_id', table_name='regex_run_results')
    op.drop_table('regex_run_results')
    op.drop_index('ix_regex_runs_experiment_id', table_name='regex_runs')
    op.drop_table('regex_runs')
