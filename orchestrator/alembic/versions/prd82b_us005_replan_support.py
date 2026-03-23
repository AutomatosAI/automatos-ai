"""PRD-82B US-005: Add replanning support to orchestration_runs

Adds:
  - replan_count column (integer, default 0)
  - Updates state CHECK constraint to include 'replanning'

Revision ID: prd82b_us005_replan
Revises: prd82a_orchestration_tables
Create Date: 2026-03-16
"""
from alembic import op
import sqlalchemy as sa

revision = "prd82b_us005_replan"
down_revision = "prd82a_orchestration_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add replan_count column
    op.add_column(
        "orchestration_runs",
        sa.Column("replan_count", sa.Integer(), nullable=False, server_default="0"),
    )

    # Drop old state CHECK constraint and recreate with 'replanning' included
    op.drop_constraint(
        "ck_orchestration_runs_state",
        "orchestration_runs",
        type_="check",
    )
    op.create_check_constraint(
        "ck_orchestration_runs_state",
        "orchestration_runs",
        "state IN ('pending', 'planning', 'awaiting_approval', 'running', "
        "'paused', 'replanning', 'verifying', 'awaiting_human', "
        "'completed', 'failed', 'cancelled')",
    )


def downgrade() -> None:
    # Restore original CHECK constraint (without 'replanning')
    op.drop_constraint(
        "ck_orchestration_runs_state",
        "orchestration_runs",
        type_="check",
    )
    op.create_check_constraint(
        "ck_orchestration_runs_state",
        "orchestration_runs",
        "state IN ('pending', 'planning', 'awaiting_approval', 'running', "
        "'paused', 'verifying', 'awaiting_human', "
        "'completed', 'failed', 'cancelled')",
    )

    # Remove replan_count column
    op.drop_column("orchestration_runs", "replan_count")
