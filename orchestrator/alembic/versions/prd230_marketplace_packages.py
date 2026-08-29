"""PRD-230 US-003 — marketplace_packages: the packages data model

A package is a curated, per-vertical bundle of EXISTING marketplace artifacts
(agents, tools, skills, plugins, playbooks, LLMs) with matching metadata and a
setup manifest. Packages are DATA, not code (D4) — curating a vertical is content
work, so the definition lives in JSONB columns, not new per-type tables.

This is the WAVE'S ONLY schema change — exactly one revision, chained onto the
single head so ``alembic heads`` stays 1.

Purely additive: ``CREATE TABLE IF NOT EXISTS`` for a brand-new table, so there
are NO assumptions about existing constraints (the ``workspaces_plan_check``
prod-drift incident is the cautionary tale — this migration touches no existing
table). Idempotent, so it is a no-op on fresh clones, CI, and any
partially-migrated environment.
"""

from alembic import op


revision = "prd230_marketplace_packages"
down_revision = "prd225_s1_asks_on_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS marketplace_packages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            slug VARCHAR(120) NOT NULL UNIQUE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            vertical_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            matching JSONB NOT NULL DEFAULT '{}'::jsonb,
            members JSONB NOT NULL DEFAULT '[]'::jsonb,
            setup_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
            showcase BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            updated_at TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )
    # Showcased packages surface first in the Packages tab (US-007).
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_marketplace_packages_showcase "
        "ON marketplace_packages (showcase);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_marketplace_packages_showcase;")
    op.execute("DROP TABLE IF EXISTS marketplace_packages;")
