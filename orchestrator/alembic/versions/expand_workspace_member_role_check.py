"""Expand workspace_members role CHECK to allow all WorkspaceRole enum values.

The original PRD-37 schema defined ``CHECK (role IN ('owner', 'member'))``,
but the application's ``WorkspaceRole`` enum has grown to include
``admin``, ``editor`` and ``viewer``. Inviting a user as ``admin`` now
fails the constraint at insert time inside ``/api/team/accept-invitation``
with ``psycopg2.errors.CheckViolation``.

This migration drops the old constraint and adds the expanded one. Same
shape used by ``workspace_invitations`` and the FastAPI permission layer.
"""

from alembic import op
from sqlalchemy import text

revision = "expand_ws_member_role_check"
down_revision = "add_clerk_invitation_id"
branch_labels = None
depends_on = None


ALLOWED_ROLES = ("owner", "admin", "editor", "viewer", "member")


def upgrade():
    op.execute(text("ALTER TABLE workspace_members DROP CONSTRAINT IF EXISTS workspace_members_role_check"))
    op.execute(
        text(
            "ALTER TABLE workspace_members ADD CONSTRAINT workspace_members_role_check "
            "CHECK (role IN ('owner', 'admin', 'editor', 'viewer', 'member'))"
        )
    )


def downgrade():
    op.execute(text("ALTER TABLE workspace_members DROP CONSTRAINT IF EXISTS workspace_members_role_check"))
    op.execute(
        text(
            "ALTER TABLE workspace_members ADD CONSTRAINT workspace_members_role_check "
            "CHECK (role IN ('owner', 'member'))"
        )
    )
