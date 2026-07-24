"""PRD-161 S5: add sla_breach_notified to board_tasks

Lets the sweeper flag an SLA-breached task (notification) exactly once. Chains
off prd161_review_feedback so the board_tasks head stays single-threaded.

Revision ID: prd161_sla_breach
Revises: prd161_review_feedback
Create Date: 2026-06-12
"""
from alembic import op
import sqlalchemy as sa

revision = "prd161_sla_breach"
down_revision = "prd161_review_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "board_tasks",
        sa.Column(
            "sla_breach_notified",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("board_tasks", "sla_breach_notified")
