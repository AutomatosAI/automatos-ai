"""Add admin lifecycle fields to workspaces table.

Supports the Workspace Admin console (pause / soft-delete):
- paused_at:     when a workspace was paused (e.g. non-payment, abuse review)
- paused_reason: short string describing why it was paused (admin note)
- deleted_at:    soft-delete marker (GDPR request, demo cleanup, etc.)

Hard-delete with S3 cascade is deferred to a worker — this migration only
adds the columns + index used by the admin list view.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_ws_admin_lifecycle"
down_revision = None  # standalone — applied manually
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workspaces",
        sa.Column("paused_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column("paused_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )

    # Index for admin list view filtering out deleted workspaces
    op.create_index(
        "ix_workspaces_deleted_at",
        "workspaces",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_workspaces_deleted_at", table_name="workspaces")
    op.drop_column("workspaces", "deleted_at")
    op.drop_column("workspaces", "paused_reason")
    op.drop_column("workspaces", "paused_at")
