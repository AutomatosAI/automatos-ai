"""Make workspace_id nullable for marketplace agents

Revision ID: 20260131_workspace_nullable
Revises: 20260131_merge
Create Date: 2026-01-31

Marketplace agents don't belong to a workspace, so workspace_id should be nullable.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260131_workspace_nullable'
down_revision = '20260131_merge'
branch_labels = None
depends_on = None


def upgrade():
    """Make workspace_id nullable for marketplace agents."""

    # Drop the NOT NULL constraint on workspace_id
    op.alter_column('agents', 'workspace_id',
                   existing_type=sa.dialects.postgresql.UUID(),
                   nullable=True)


def downgrade():
    """Restore NOT NULL constraint on workspace_id."""

    # This will fail if there are any marketplace agents with NULL workspace_id
    # Clean those up first if needed
    op.alter_column('agents', 'workspace_id',
                   existing_type=sa.dialects.postgresql.UUID(),
                   nullable=False)
