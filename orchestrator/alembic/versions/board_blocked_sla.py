"""Add blocked status, SLA deadline, and blocked metadata to board_tasks

Revision ID: board_blocked_sla
Revises: governance_blueprints
Create Date: 2026-03-29
"""
from alembic import op
import sqlalchemy as sa

revision = "board_blocked_sla"
down_revision = "governance_blueprints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("board_tasks", sa.Column("sla_deadline", sa.DateTime(timezone=True), nullable=True))
    op.add_column("board_tasks", sa.Column("blocked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("board_tasks", sa.Column("blocked_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("board_tasks", "blocked_reason")
    op.drop_column("board_tasks", "blocked_at")
    op.drop_column("board_tasks", "sla_deadline")
