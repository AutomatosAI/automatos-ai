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
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Add public_id column if not exists
    conn.execute(sa.text(
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS public_id UUID"
    ))

    # 2. Backfill existing rows with random UUIDs
    conn.execute(sa.text(
        "UPDATE agents SET public_id = gen_random_uuid() WHERE public_id IS NULL"
    ))

    # 3. Add unique index on public_id (if not exists)
    conn.execute(sa.text(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_agents_public_id ON agents (public_id)"
    ))

    # 4. Drop old global unique constraint on slug (try both naming patterns)
    conn.execute(sa.text(
        "DROP INDEX IF EXISTS ix_agents_slug"
    ))
    try:
        conn.execute(sa.text(
            "ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_slug_key"
        ))
    except Exception:
        pass

    # 5. Widen slug column to 255
    conn.execute(sa.text(
        "ALTER TABLE agents ALTER COLUMN slug TYPE VARCHAR(255)"
    ))

    # 6. Add per-workspace unique constraint on (workspace_id, slug) if not exists
    conn.execute(sa.text("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_agent_workspace_slug') THEN
                ALTER TABLE agents ADD CONSTRAINT uq_agent_workspace_slug UNIQUE (workspace_id, slug);
            END IF;
        END $$;
    """))

    print("agent_public_id_and_slug_fix: migration complete")


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
