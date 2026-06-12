"""PRD-161: board dispatch lease — add lease_until + attempts to board_tasks

Extends the existing board_tasks table (no new table) so the dispatch spine can
claim/lease/requeue. Branches off board_blocked_sla — the latest board_tasks
head — so the alembic head count is unchanged.

Revision ID: prd161_dispatch_lease
Revises: board_blocked_sla
Create Date: 2026-06-12
"""
from alembic import op
import sqlalchemy as sa

revision = "prd161_dispatch_lease"
down_revision = "board_blocked_sla"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "board_tasks",
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "board_tasks",
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("board_tasks", "attempts")
    op.drop_column("board_tasks", "lease_until")
