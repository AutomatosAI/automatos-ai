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
    # #840: tags was plain JSON here, which Postgres gives no default GIN opclass to, so
    # the GIN index could not be added in this revision. It is built in
    # workflows_tags_jsonb, once that later revision has converted the column to JSONB.
    # (A commented-out call to the execute operation used to sit here as a reminder; it
    # is removed rather than left dormant because build_schema's stage 5 re-runs every
    # upgrade() body as plain text, comments included, and was matching and retrying it
    # as live SQL against the still-JSON column on every fresh build.)


def downgrade() -> None:
    op.drop_index('ix_workflows_owner', table_name='workflows')


