"""add indexes on workflows.owner and workflows.tags

Revision ID: add_workflow_indexes
Revises: add_workflow_meta
Create Date: 2025-08-12 15:00:02

"""
from alembic import op
import sqlalchemy as sa


revision = 'add_workflow_indexes'
down_revision = 'add_workflow_meta'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index('ix_workflows_owner', 'workflows', ['owner'])
    # Note: For JSONB tags, a GIN index would be ideal. If available, uncomment below:
    # op.execute("CREATE INDEX IF NOT EXISTS ix_workflows_tags_gin ON workflows USING GIN (tags)")


def downgrade() -> None:
    op.drop_index('ix_workflows_owner', table_name='workflows')
    # op.execute("DROP INDEX IF EXISTS ix_workflows_tags_gin")


