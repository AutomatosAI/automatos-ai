"""PRD-67: Add is_system_agent and required_role columns to agents table

Revision ID: 20260226_cto_agent
Revises: 20260226_merge_heads
Create Date: 2026-02-26

Adds:
- is_system_agent (boolean) — system agents are global, seeded by platform
- required_role (varchar) — if set, agent only visible to users with this system_role
- slug (varchar, unique) — stable identifier for system agents (idempotent seeding)
"""

from alembic import op
import sqlalchemy as sa

revision = "20260226_cto_agent"
down_revision = "20260226_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column("is_system_agent", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "agents",
        sa.Column("required_role", sa.String(50), nullable=True),
    )
    op.add_column(
        "agents",
        sa.Column("slug", sa.String(100), nullable=True),
    )

    # Index for quick system agent lookups
    op.create_index(
        "idx_agents_system_agent",
        "agents",
        ["is_system_agent"],
        postgresql_where=sa.text("is_system_agent = true"),
    )
    # Unique slug for idempotent seeding
    op.create_index(
        "idx_agents_slug_unique",
        "agents",
        ["slug"],
        unique=True,
        postgresql_where=sa.text("slug IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_agents_slug_unique", table_name="agents")
    op.drop_index("idx_agents_system_agent", table_name="agents")
    op.drop_column("agents", "slug")
    op.drop_column("agents", "required_role")
    op.drop_column("agents", "is_system_agent")
