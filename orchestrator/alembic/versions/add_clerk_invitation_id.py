"""Add clerk_invitation_id to workspace_invitations.

Tracks the Clerk-side invitation ID so that local revoke can also revoke
the Clerk invitation, keeping the two systems in sync (and allowing
re-invitation of the same email).

Standalone migration: down_revision = None.
"""

from alembic import op
from sqlalchemy import text

revision = "add_clerk_invitation_id"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute(text("""
        ALTER TABLE workspace_invitations
        ADD COLUMN IF NOT EXISTS clerk_invitation_id VARCHAR(255)
    """))


def downgrade():
    op.execute(text("ALTER TABLE workspace_invitations DROP COLUMN IF EXISTS clerk_invitation_id"))
