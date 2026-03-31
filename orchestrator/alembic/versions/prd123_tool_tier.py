"""PRD-123 Pattern #4: Add tier column to tools table

Adds tier column (varchar(20), default 'marketplace') and backfills
system/platform tiers for known internal tools.

Revision ID: prd123_tool_tier
Revises: prd123_stop_reason
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa

revision = "prd123_tool_tier"
down_revision = "prd123_stop_reason"
branch_labels = None
depends_on = None

# Internal tools that should be system tier
SYSTEM_TOOL_NAMES = ("RAG", "MEMORY", "NL2SQL", "CODEGRAPH")


def upgrade() -> None:
    op.add_column(
        "tools",
        sa.Column("tier", sa.String(20), server_default="marketplace", nullable=True),
    )
    op.create_index("ix_tools_tier", "tools", ["tier"])

    # Backfill system-tier tools
    op.execute(
        f"UPDATE tools SET tier = 'system' "
        f"WHERE name IN ({', '.join(repr(n) for n in SYSTEM_TOOL_NAMES)})"
    )
    # Backfill platform-tier tools (Automatos-sourced, not system)
    op.execute(
        "UPDATE tools SET tier = 'platform' "
        "WHERE provider = 'automatos' AND tier = 'marketplace'"
    )


def downgrade() -> None:
    op.drop_index("ix_tools_tier", table_name="tools")
    op.drop_column("tools", "tier")
