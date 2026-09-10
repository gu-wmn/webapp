"""add dialogue url and label type

Somewhere to keep the thread url a dialogue came from, and the type chosen for
a span where the annotation scheme has one - MLCR, explification and the rest
on the Swedish forums material.

Revision ID: 92599e8a5092
Revises:
Create Date: 2026-09-10 15:10:39.397498

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '92599e8a5092'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('dialogues', schema=None) as batch_op:
        batch_op.add_column(sa.Column('url', sa.String(length=1024), nullable=True))

    with op.batch_alter_table('annotation_labels', schema=None) as batch_op:
        batch_op.add_column(sa.Column('type', sa.String(length=64), nullable=True))


def downgrade():
    with op.batch_alter_table('annotation_labels', schema=None) as batch_op:
        batch_op.drop_column('type')

    with op.batch_alter_table('dialogues', schema=None) as batch_op:
        batch_op.drop_column('url')
