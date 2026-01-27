"""Add unique constraint on (workspace_id, user_id) for workspace_members

Revision ID: 20260127_workspace_member_uq
Revises: 102088374664
Create Date: 2026-01-27

Prevents duplicate workspace memberships.
"""
from alembic import op

revision = "20260127_workspace_member_uq"
down_revision = "102088374664"
branch_labels = None
depends_on = None


def upgrade():
    op.create_unique_constraint(
        "uq_workspace_member_workspace_user",
        "workspace_members",
        ["workspace_id", "user_id"],
    )


def downgrade():
    op.drop_constraint(
        "uq_workspace_member_workspace_user",
        "workspace_members",
        type_="unique",
    )
