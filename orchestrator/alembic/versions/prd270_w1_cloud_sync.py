"""PRD-270 W1: where a Marketplace theme row came from, and the Sync's one status row (30 Oct 2026).

The ONE migration of wave W1 (the open Marketplace); every later story in the wave adds none.

* ``theme_packages.source``: NULL for a row this install made (an upload, a submission, a
  publish), ``'cloud'`` for a Marketplace row a Sync with Automatos Cloud wrote. Nullable,
  with no default, so every existing row reads NULL (this install's own) and no row is
  rewritten. A CHECK allows ``'cloud'`` only on a Marketplace row (``owner_type =
  'marketplace'``) and refuses any other value.
* ``marketplace_cloud_sync``: one row (``CHECK (id = 1)``) the local edition's Sync writes:
  when it last tried, when it last succeeded, the catalogue's ``generatedAt``, the last
  error and the last Sync's counts (new, updated, unchanged, refused). It mirrors
  ``theme_revocation_sync`` (``prd261_revocation_feed``).

``core/models/themes.py`` (``ThemePackage.source``) and ``core/models/marketplace_cloud.py``
(``MarketplaceCloudSync``) declare the same shape; ``tests/test_prd261_theme_models.py`` and
``tests/test_prd270w1_migration.py`` hold the two together.

Create_all-first safe: the column and the CHECK are added only when the table lacks them
(a DO block that asks ``pg_attribute`` and ``pg_constraint`` for the table the unqualified
name resolves to, so a database ``create_all`` built is not even locked), the table is
``CREATE TABLE IF NOT EXISTS``, and running it twice changes nothing. A nullable column
with no default rewrites no row.

The downgrade drops the sync table, then the CHECK and the column. It deletes no
``theme_packages`` row: a row a Sync wrote stays, as a row of this install (deleting one
would delete its installs: ``theme_installations.package_id`` is ON DELETE CASCADE).

Revision ID: prd270_w1_cloud_sync
Revises: prd268_w1_proactive
Create Date: 2026-10-30
"""

from alembic import op


revision = "prd270_w1_cloud_sync"
down_revision = "prd268_w1_proactive"
branch_labels = None
depends_on = None

# theme_packages.source and its CHECK, each added only when the table lacks it.
_ADD_SOURCE = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'theme_packages'::regclass AND attname = 'source' AND NOT attisdropped
    ) THEN
        ALTER TABLE theme_packages ADD COLUMN source VARCHAR(16);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'theme_packages'::regclass AND conname = 'ck_theme_packages_source'
    ) THEN
        ALTER TABLE theme_packages ADD CONSTRAINT ck_theme_packages_source CHECK (source IS NULL OR (source = 'cloud' AND owner_type = 'marketplace'));
    END IF;
END $$;
"""

# (table, CREATE TABLE), as in prd261_revocation_feed.
_TABLES = (
    ("marketplace_cloud_sync", """
CREATE TABLE IF NOT EXISTS marketplace_cloud_sync (
    id INTEGER PRIMARY KEY,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    last_success_at TIMESTAMP WITH TIME ZONE,
    feed_generated_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    new_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    unchanged_count INTEGER NOT NULL DEFAULT 0,
    refused_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_marketplace_cloud_sync_single_row CHECK (id = 1)
);
"""),
)

# The upgrade's steps undone, in reverse order. No theme_packages row is deleted.
_DOWNGRADE = (
    "DROP TABLE IF EXISTS marketplace_cloud_sync;",
    "ALTER TABLE theme_packages DROP CONSTRAINT IF EXISTS ck_theme_packages_source;",
    "ALTER TABLE theme_packages DROP COLUMN IF EXISTS source;",
)


def upgrade() -> None:
    op.execute(_ADD_SOURCE)
    for _table, statement in _TABLES:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
