"""PRD-261 W-C3 (C-05f): where a revocation came from, and the pull's one sync row (21 Oct 2026).

* ``theme_revocations.origin``: ``'local'`` (written in this install: an admin's
  revocation, a review deactivation) or ``'feed'`` (pulled from the hosted revocation
  feed, keyed by the hosted row's id). ``NOT NULL DEFAULT 'local'``, so every existing
  row reads ``'local'``; a CHECK keeps it to those two values.
* ``uq_theme_revocations_live_local``: at most one live (``lifted_at IS NULL``) ``'local'``
  row per target and level, so two identical creates in the same instant can't both write
  (``revocation_admin.create_revocation`` answers the row that won). The target columns are
  read through ``COALESCE(<column>, '')``: a unique index counts NULLs as distinct, so on the bare
  columns a digest row (theme id and range NULL) would never collide. A ``'feed'`` row and a
  lifted row are outside it. Built after ``origin``, which its predicate names; a database
  that already holds two such twins (create and deactivate check for one before writing, so
  only a lost race leaves them) refuses it, and the upgrade stops there.
* ``theme_revocation_sync``: one row (``CHECK (id = 1)``) the local edition's pull
  writes: when it last tried, when it last succeeded, the feed's ``generatedAt``, the
  last error, the feed's ``revokedKeys`` and when the row last changed.

``core/models/themes.py`` declares the same shape (``ThemeRevocation.origin``,
``ThemeRevocationSync``) and ``tests/test_prd261_theme_models.py`` holds the two together.

Create_all-first safe: the column and the CHECK are added only when the table lacks
them (a DO block that asks ``pg_attribute`` and ``pg_constraint`` for the table the
unqualified name resolves to, so a database ``create_all`` built is not even locked),
the table is ``CREATE TABLE IF NOT EXISTS``, the index ``CREATE UNIQUE INDEX IF NOT
EXISTS``, and running it twice changes nothing.
Adding a column with a constant default rewrites no row on Postgres 11+.

The downgrade drops the index and the sync table, then the CHECK and the column; it deletes no
revocation row (a pulled row stays, as a row of this install).

Revision ID: prd261_revocation_feed
Revises: clerk_user_relinks_20261019
Create Date: 2026-10-21
"""

from alembic import op


revision = "prd261_revocation_feed"
down_revision = "clerk_user_relinks_20261019"
branch_labels = None
depends_on = None

# theme_revocations.origin and its CHECK, each added only when the table lacks it.
_ADD_ORIGIN = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'theme_revocations'::regclass AND attname = 'origin' AND NOT attisdropped
    ) THEN
        ALTER TABLE theme_revocations ADD COLUMN origin VARCHAR(16) NOT NULL DEFAULT 'local';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'theme_revocations'::regclass AND conname = 'ck_theme_revocations_origin'
    ) THEN
        ALTER TABLE theme_revocations ADD CONSTRAINT ck_theme_revocations_origin CHECK (origin IN ('local', 'feed'));
    END IF;
END $$;
"""

# (table, CREATE TABLE), as in prd261_themes.
_TABLES = (
    ("theme_revocation_sync", """
CREATE TABLE IF NOT EXISTS theme_revocation_sync (
    id INTEGER PRIMARY KEY,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    last_success_at TIMESTAMP WITH TIME ZONE,
    feed_generated_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    revoked_key_ids JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_theme_revocation_sync_single_row CHECK (id = 1)
);
"""),
)

# After _ADD_ORIGIN (the predicate names origin), as core/models/themes.py declares it.
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_revocations_live_local ON theme_revocations "
    "(COALESCE(digest, ''), COALESCE(theme_id, ''), COALESCE(version_range, ''), level) "
    "WHERE lifted_at IS NULL AND origin = 'local';",
)

# The upgrade's steps undone, in reverse order.
_DOWNGRADE = (
    "DROP INDEX IF EXISTS uq_theme_revocations_live_local;",
    "DROP TABLE IF EXISTS theme_revocation_sync;",
    "ALTER TABLE theme_revocations DROP CONSTRAINT IF EXISTS ck_theme_revocations_origin;",
    "ALTER TABLE theme_revocations DROP COLUMN IF EXISTS origin;",
)


def upgrade() -> None:
    op.execute(_ADD_ORIGIN)
    for _table, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
