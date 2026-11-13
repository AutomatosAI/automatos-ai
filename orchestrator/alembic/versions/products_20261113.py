"""Products (13 Nov 2026).

PRD-274, "Automatos for products", Wave 5a (§5.1, §8-3). One new table, ``products``: a product's slug and
name, its owner workspace (``owner_workspace_id``, ON DELETE RESTRICT: the purge refuses an owner first,
and this is the backstop), the PBKDF2 hash of its key and the key's first characters, the roster and key
each customer's workspace starts with, and the default retention of an end user's memories.

``core/models/products.py`` (``Product``) declares the same shape. Create_all-first safe:
``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS``; it changes no row. The downgrade drops the
table.

Revision ID: products_20261113
Revises: job_schedules_20261112
Create Date: 2026-11-13
"""
from alembic import op

revision = "products_20261113"
down_revision = "job_schedules_20261112"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY,
    slug VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    owner_workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
    key_hash VARCHAR(128) NOT NULL,
    key_prefix VARCHAR(32) NOT NULL,
    agent_slugs JSONB NOT NULL DEFAULT '[]'::jsonb,
    ops_manager_slug VARCHAR(100),
    key_type VARCHAR(16) NOT NULL DEFAULT 'server',
    key_scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
    default_retention_days INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT products_slug_key UNIQUE (slug),
    CONSTRAINT products_key_hash_key UNIQUE (key_hash),
    CONSTRAINT ck_products_key_type CHECK (key_type IN ('public', 'server')),
    CONSTRAINT ck_products_default_retention_days CHECK (default_retention_days IS NULL OR default_retention_days >= 1)
);
"""


def upgrade() -> None:
    op.execute(_TABLE)
    op.execute("CREATE INDEX IF NOT EXISTS ix_products_owner_workspace_id ON products (owner_workspace_id);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS products;")
