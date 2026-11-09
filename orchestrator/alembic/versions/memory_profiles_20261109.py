"""Memory profiles: what distillation keeps as a fact, by domain (9 Nov 2026).

PRD draft "Automatos for products", Wave 4 (§5.6). One new table:

* ``memory_profiles``: a profile's slug, name, description, focus, what to keep and skip, and
  worked examples. A seeded profile has no workspace (``core/seeds/seed_memory_profiles.py``
  writes them, insert or update on the slug); a workspace's own has its workspace, ON DELETE
  CASCADE. Two partial unique indexes keep one seeded profile per slug and one per slug in a
  workspace.

``core/models/memory_profiles.py`` (``MemoryProfile``) declares the same shape;
``tests/test_afp_w4a_memory_profiles.py`` holds the two together.

Create_all-first safe: the table is ``CREATE TABLE IF NOT EXISTS`` and each index ``CREATE
UNIQUE INDEX IF NOT EXISTS``, so a database ``create_all`` built is left as it is, and running
it twice changes nothing. It alters no table and changes no row. The downgrade drops the table.

Revision ID: memory_profiles_20261109
Revises: auto_soul_one_source_20261107
Create Date: 2026-11-09
"""
from alembic import op

revision = "memory_profiles_20261109"
down_revision = "auto_soul_one_source_20261107"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS memory_profiles (
    id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    slug VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    focus TEXT NOT NULL DEFAULT '',
    keep JSONB NOT NULL DEFAULT '[]'::jsonb,
    skip JSONB NOT NULL DEFAULT '[]'::jsonb,
    examples JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""

# As core/models/memory_profiles.py declares them.
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_profiles_seeded_slug ON memory_profiles (slug) "
    "WHERE workspace_id IS NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_profiles_workspace_slug ON memory_profiles (workspace_id, slug) "
    "WHERE workspace_id IS NOT NULL;",
)


def upgrade() -> None:
    op.execute(_TABLE)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_profiles;")
