"""add owner, tags, default_policy_id to workflows

Revision ID: add_workflow_meta
Revises: add_playbooks
Create Date: 2025-08-12 15:00:01

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = 'add_workflow_meta'
down_revision = 'add_playbooks'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('workflows', sa.Column('owner', sa.String(length=255), nullable=True))
    op.add_column('workflows', sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('workflows', sa.Column('default_policy_id', sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column('workflows', 'default_policy_id')
    op.drop_column('workflows', 'tags')
    op.drop_column('workflows', 'owner')


