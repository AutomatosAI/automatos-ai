"""PRD-123 Pattern #8: Add checkpoint_count to orchestration_runs

Revision ID: prd123_checkpoint_count
Revises: prd123_tool_tier
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa

revision = "prd123_checkpoint_count"
down_revision = "prd123_tool_tier"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orchestration_runs",
        sa.Column("checkpoint_count", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("orchestration_runs", "checkpoint_count")
