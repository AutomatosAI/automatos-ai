"""Add team, job_title, reports_to_id to agents for Mission Zero org chart.

Revision ID: mission_zero_org_fields
Revises: prd82c_parallel_schema
"""

revision = "mission_zero_org_fields"
down_revision = "prd82c_parallel_schema"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    op.add_column("agents", sa.Column("team", sa.String(100), nullable=True))
    op.add_column("agents", sa.Column("job_title", sa.String(200), nullable=True))
    op.add_column(
        "agents",
        sa.Column(
            "reports_to_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_agents_team", "agents", ["team"])
    op.create_index("ix_agents_reports_to_id", "agents", ["reports_to_id"])


def downgrade() -> None:
    op.drop_index("ix_agents_reports_to_id", table_name="agents")
    op.drop_index("ix_agents_team", table_name="agents")
    op.drop_column("agents", "reports_to_id")
    op.drop_column("agents", "job_title")
    op.drop_column("agents", "team")
