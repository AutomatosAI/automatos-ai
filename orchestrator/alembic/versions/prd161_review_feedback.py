"""PRD-161 Q44: add review_feedback to board_tasks

Carries reviewer feedback into the next execution when a task is rejected back
to the same agent. Chains off prd161_dispatch_lease (the S1 lease migration) so
the board_tasks head stays single-threaded.

Revision ID: prd161_review_feedback
Revises: prd161_dispatch_lease
Create Date: 2026-06-12
"""
from alembic import op
import sqlalchemy as sa

revision = "prd161_review_feedback"
down_revision = "prd161_dispatch_lease"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "board_tasks",
        sa.Column("review_feedback", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("board_tasks", "review_feedback")
