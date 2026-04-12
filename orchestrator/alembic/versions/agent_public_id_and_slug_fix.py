"""Add public_id UUID to agents, fix slug uniqueness to per-workspace

Adds a `public_id` UUID column for external/widget-facing agent identification.
Sequential integer IDs are guessable and unsafe for public widget configs on
Shopify storefronts. Widgets will use public_id (UUID) instead of id (int).

Also changes the slug uniqueness constraint from table-wide to per-workspace.
Two workspaces can both have a "researcher" agent — they're in different
workspaces, no conflict. The old global unique constraint blocked this.

Revision ID: agent_public_id_and_slug_fix
Revises: drop_agents_model_config_default
Create Date: 2026-04-12
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "agent_public_id_and_slug_fix"
down_revision = "drop_agents_model_config_default"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add public_id column as nullable first
    op.add_column("agents", sa.Column("public_id", UUID(as_uuid=True), nullable=True))

    # 2. Backfill existing rows with random UUIDs
    op.execute("UPDATE agents SET public_id = gen_random_uuid() WHERE public_id IS NULL")

    # 3. Make non-nullable now that all rows have values
    op.alter_column("agents", "public_id", nullable=False)

    # 4. Add unique index on public_id
    op.create_index("ix_agents_public_id", "agents", ["public_id"], unique=True)

    # 5. Drop old global unique constraint on slug
    # The column may have a unique constraint created inline or via index
    # Try dropping the index first (SQLAlchemy creates ix_agents_slug for unique=True)
    try:
        op.drop_index("ix_agents_slug", table_name="agents")
    except Exception:
        pass
    # Also try the constraint name pattern
    try:
        op.drop_constraint("agents_slug_key", "agents", type_="unique")
    except Exception:
        pass

    # 6. Widen slug column from 100 to 255
    op.alter_column("agents", "slug", type_=sa.String(255), existing_type=sa.String(100))

    # 7. Add per-workspace unique constraint on (workspace_id, slug)
    op.create_unique_constraint("uq_agent_workspace_slug", "agents", ["workspace_id", "slug"])


def downgrade() -> None:
    # Remove per-workspace slug constraint
    op.drop_constraint("uq_agent_workspace_slug", "agents", type_="unique")

    # Restore slug to String(100)
    op.alter_column("agents", "slug", type_=sa.String(100), existing_type=sa.String(255))

    # Restore global unique on slug
    op.create_index("ix_agents_slug", "agents", ["slug"], unique=True)

    # Remove public_id
    op.drop_index("ix_agents_public_id", table_name="agents")
    op.drop_column("agents", "public_id")
