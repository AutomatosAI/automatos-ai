"""PRD-76: Make agent_reports.agent_id nullable + add agent_name column

Orchestrator heartbeats have no agent_id. This allows orchestrator-level
reports to be stored alongside agent reports.

Standalone migration (down_revision = None).
"""

from alembic import op
import sqlalchemy as sa

revision = "prd76_nullable_agent"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Make agent_id nullable (orchestrator reports have no agent)
    op.alter_column(
        "agent_reports",
        "agent_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )

    # Add agent_name column so we don't need JOIN for display
    op.add_column(
        "agent_reports",
        sa.Column("agent_name", sa.String(255), nullable=True),
    )

    # Backfill agent_name from agents table
    op.execute("""
        UPDATE agent_reports r
        SET agent_name = a.name
        FROM agents a
        WHERE r.agent_id = a.id AND r.agent_name IS NULL
    """)


def downgrade():
    op.drop_column("agent_reports", "agent_name")
    op.alter_column(
        "agent_reports",
        "agent_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
