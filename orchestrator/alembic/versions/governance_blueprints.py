"""Add agent_blueprints table and budget columns on orchestration_runs.

Revision ID: governance_blueprints
Revises: mission_zero_org_fields
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "governance_blueprints"
down_revision = "mission_zero_org_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_blueprints",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("rules", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.add_column(
        "orchestration_runs",
        sa.Column("budget_config", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "orchestration_runs",
        sa.Column("budget_spent", postgresql.JSONB(), server_default="{}", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("orchestration_runs", "budget_spent")
    op.drop_column("orchestration_runs", "budget_config")
    op.drop_table("agent_blueprints")
