"""PRD-259 Wave 1 (S-002): skins in Studio — the ONE migration of the wave (14 Oct 2026).

* ``marketplace_skins``: one row per skin version — a Marketplace skin (reviewed:
  ``status``) or one workspace's private import (``owner_type`` 'workspace', its
  ``workspace_id`` required by a CHECK). ``skin`` holds the validated skin.json plus
  the asset manifest; ``asset_prefix`` is where its files live in the marketplace
  bucket. A private slug is unique per workspace and a Marketplace slug per author
  and version (partial unique indexes).
* ``skin_installations``: a workspace's installs, by reference; at most one active
  row (``uninstalled_at`` IS NULL) per workspace and skin. Deleting a skin deletes
  its installations.

Every column Wave 2's routes need is here: Wave 2 adds no migration. No existing
table is altered.

``core/models/marketplace_skins.py`` declares the same shape, and
``tests/test_prd259_skins_models.py`` holds the two together.

Create_all-first safe (the 89d89c250 rule): CREATE TABLE IF NOT EXISTS and
CREATE [UNIQUE] INDEX IF NOT EXISTS, so it is a no-op on a database that
``create_all`` built from the models, and running it twice changes nothing.

Revision ID: prd259_skins
Revises: prd258_voice_timings
Create Date: 2026-10-14
"""

from alembic import op


revision = "prd259_skins"
down_revision = "prd258_voice_timings"
branch_labels = None
depends_on = None

_CREATE_SKINS = """
CREATE TABLE IF NOT EXISTS marketplace_skins (
    id UUID PRIMARY KEY,
    slug VARCHAR(64) NOT NULL,
    name VARCHAR(40) NOT NULL,
    version VARCHAR(32) NOT NULL,
    author_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    skin JSONB NOT NULL,
    asset_prefix VARCHAR NOT NULL,
    preview_path VARCHAR,
    owner_type VARCHAR(16) NOT NULL,
    workspace_id UUID,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    rejection_reason TEXT,
    featured BOOLEAN NOT NULL DEFAULT false,
    install_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_marketplace_skins_owner_type CHECK (owner_type IN ('marketplace', 'workspace')),
    CONSTRAINT ck_marketplace_skins_workspace_owner CHECK (owner_type <> 'workspace' OR workspace_id IS NOT NULL),
    CONSTRAINT ck_marketplace_skins_status CHECK (status IN ('pending', 'approved', 'rejected'))
);
"""

_CREATE_INSTALLATIONS = """
CREATE TABLE IF NOT EXISTS skin_installations (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL,
    skin_id UUID NOT NULL REFERENCES marketplace_skins(id) ON DELETE CASCADE,
    version VARCHAR(32) NOT NULL,
    installed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    installed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    uninstalled_at TIMESTAMP WITH TIME ZONE
);
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_marketplace_skins_workspace_id ON marketplace_skins (workspace_id);",
    "CREATE INDEX IF NOT EXISTS ix_marketplace_skins_status ON marketplace_skins (status);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_skins_workspace_slug "
    "ON marketplace_skins (workspace_id, slug) WHERE owner_type = 'workspace';",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_marketplace_skins_author_slug_version "
    "ON marketplace_skins (author_user_id, slug, version) WHERE owner_type = 'marketplace';",
    "CREATE INDEX IF NOT EXISTS ix_skin_installations_workspace_id ON skin_installations (workspace_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_skin_installations_active "
    "ON skin_installations (workspace_id, skin_id) WHERE uninstalled_at IS NULL;",
)


def upgrade() -> None:
    op.execute(_CREATE_SKINS)
    op.execute(_CREATE_INSTALLATIONS)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # The indexes go with their tables; installations first (they reference skins).
    op.execute("DROP TABLE IF EXISTS skin_installations;")
    op.execute("DROP TABLE IF EXISTS marketplace_skins;")
