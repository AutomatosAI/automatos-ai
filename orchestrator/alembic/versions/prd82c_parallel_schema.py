"""PRD-82C: Add parallel execution and budget governance columns

Adds to orchestration_tasks:
  - complexity VARCHAR(10) DEFAULT 'moderate'
  - parallel_group VARCHAR(50) DEFAULT NULL
  - estimated_tokens INTEGER DEFAULT 4000

Alters orchestration_runs:
  - max_concurrent default changes from 1 to 3

Revision ID: prd82c_parallel_schema
Revises: prd82b_us009_archive
Create Date: 2026-03-24
"""
from alembic import op
import sqlalchemy as sa

revision = "prd82c_parallel_schema"
down_revision = "prd82b_us009_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add complexity column to orchestration_tasks
    op.add_column(
        "orchestration_tasks",
        sa.Column("complexity", sa.String(10), nullable=False, server_default="moderate"),
    )

    # Add parallel_group column to orchestration_tasks
    op.add_column(
        "orchestration_tasks",
        sa.Column("parallel_group", sa.String(50), nullable=True),
    )

    # Add estimated_tokens column to orchestration_tasks
    op.add_column(
        "orchestration_tasks",
        sa.Column("estimated_tokens", sa.Integer(), nullable=False, server_default="4000"),
    )

    # Change max_concurrent default from 1 to 3 on orchestration_runs
    op.alter_column(
        "orchestration_runs",
        "max_concurrent",
        server_default="3",
    )


def downgrade() -> None:
    # Revert max_concurrent default to 1
    op.alter_column(
        "orchestration_runs",
        "max_concurrent",
        server_default="1",
    )

    # Drop new columns
    op.drop_column("orchestration_tasks", "estimated_tokens")
    op.drop_column("orchestration_tasks", "parallel_group")
    op.drop_column("orchestration_tasks", "complexity")
