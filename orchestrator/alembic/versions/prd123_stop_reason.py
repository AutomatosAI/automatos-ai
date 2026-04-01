"""PRD-123 Pattern #6: Add stop_reason and stop_detail to orchestration_runs

Adds:
  - stop_reason column (varchar(50), nullable)
  - stop_detail column (text, nullable)

Revision ID: prd123_stop_reason
Revises: prd82c_parallel_schema
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa

revision = "prd123_stop_reason"
down_revision = "prd82c_parallel_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orchestration_runs",
        sa.Column("stop_reason", sa.String(50), nullable=True),
    )
    op.add_column(
        "orchestration_runs",
        sa.Column("stop_detail", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("orchestration_runs", "stop_detail")
    op.drop_column("orchestration_runs", "stop_reason")
