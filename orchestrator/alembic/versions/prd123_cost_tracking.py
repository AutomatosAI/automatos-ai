"""PRD-123 Pattern #12: Add cost tracking columns to tool_execution_logs

Adds estimated_cost, rate_limit_remaining, execution_ms columns.

Revision ID: prd123_cost_tracking
Revises: prd123_checkpoint_count
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa

revision = "prd123_cost_tracking"
down_revision = "prd123_checkpoint_count"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool_execution_logs",
        sa.Column("estimated_cost", sa.Float(), server_default="0.0", nullable=True),
    )
    op.add_column(
        "tool_execution_logs",
        sa.Column("rate_limit_remaining", sa.Integer(), nullable=True),
    )
    op.add_column(
        "tool_execution_logs",
        sa.Column("execution_ms", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tool_execution_logs", "execution_ms")
    op.drop_column("tool_execution_logs", "rate_limit_remaining")
    op.drop_column("tool_execution_logs", "estimated_cost")
