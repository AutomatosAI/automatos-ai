"""People without Clerk: an identity link and a stored install-wide role (20 Nov 2026).

PRD-277, enterprise sign-in, Wave 2 (§5.4). The ONE migration of the wave's US-201 and US-203:

* ``user_identities`` (US-201, a new table): one row per ``(issuer, subject)`` a person signs in
  with, linked to their ``users`` row. ON DELETE CASCADE with the person. ``users.clerk_user_id``
  is untouched.
* ``uq_user_identities_issuer_subject`` on ``(issuer, subject)``: one pair names one person, and
  of two first sign-ins racing for the same pair, one link is written. ``ix_user_identities_user_id``
  serves a person's own links and the cascade.
* ``users.system_role`` (US-203, a new nullable column) and ``ck_users_system_role``: the
  install-wide role (``super_admin``, ``admin`` or ``user``), NULL on every row today, so no row
  changes. It is read only when the edition signs people in with OIDC.

``core/models/user_identities.py`` declares the same names. Create_all-first safe: ``CREATE TABLE
IF NOT EXISTS`` and ``CREATE [UNIQUE] INDEX IF NOT EXISTS``, and the column and the CHECK are added
only when ``users`` lacks them (a DO block that asks ``pg_attribute`` and ``pg_constraint``, as
``prd270_w1_cloud_sync`` does), so a database ``create_all`` built ends exactly as it was, with no
lock taken on ``users``, and running it twice changes nothing. No row is written or changed.

The locks, in the order they are taken, and why none scans or rewrites ``users``:

* ``SET LOCAL lock_timeout`` (``_LOCK_TIMEOUT``) first: a statement below that waits longer than
  that for its lock fails, and the migration run fails with it, instead of waiting in the lock
  queue with every later ``users`` query waiting behind it. ``alembic/env.py`` runs the pending revisions in one
  transaction, so the setting applies to this revision's statements; the value in force before
  it is kept in ``_SAVED_LOCK_TIMEOUT`` and put back as the upgrade's last statement, so a later
  revision in the same run waits exactly as it did before.
* ``ALTER TABLE users ADD COLUMN system_role`` (only when missing): ACCESS EXCLUSIVE, held for
  a catalog change only. A nullable column with no default rewrites no row. It runs before the
  table, so the strongest lock on ``users`` is the first one taken: no statement after it asks
  for a stronger lock on ``users`` than the run already holds.
* ``ALTER TABLE users ADD CONSTRAINT ck_users_system_role ... NOT VALID`` (only when missing):
  ACCESS EXCLUSIVE, held for a catalog change only. ``NOT VALID`` skips the scan of existing
  rows, every one of which is NULL at that point; each row written from then on is checked.
* ``CREATE TABLE user_identities`` (only when it is missing): its foreign key needs SHARE ROW
  EXCLUSIVE on ``users`` to add the key's triggers; when the column was added in this run, the
  ACCESS EXCLUSIVE already held covers it. Nothing in ``users`` is read, as the new table is
  empty. Skipped, with no lock on ``users``, when the table exists.
* The two ``CREATE [UNIQUE] INDEX`` statements: at most a SHARE lock on ``user_identities``, a
  new or ``create_all``-built table; never a lock on ``users``.

The downgrade drops the table and the column (its CHECK goes with it), and only them.

Revision ID: prd277_w2_identities_20261120
Revises: merge_heads_20261117
Create Date: 2026-11-20
"""
from alembic import op

revision = "prd277_w2_identities_20261120"
down_revision = "merge_heads_20261117"
branch_labels = None
depends_on = None

# How long one statement here may wait for a lock before the migration fails.
_LOCK_TIMEOUT = "5s"
# Where the lock_timeout in force before this revision is kept until the upgrade ends.
_SAVED_LOCK_TIMEOUT = "automatos.lock_timeout_before_prd277_w2"
_BOUND_LOCK_WAIT = (
    f"SELECT set_config('{_SAVED_LOCK_TIMEOUT}', current_setting('lock_timeout'), true);",
    f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}';",
)
_RESTORE_LOCK_WAIT = f"SELECT set_config('lock_timeout', current_setting('{_SAVED_LOCK_TIMEOUT}'), true);"

_TABLE = """
CREATE TABLE IF NOT EXISTS user_identities (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    issuer VARCHAR(512) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""

# As core/models/user_identities.py declares them (IDENTITY_PAIR_INDEX, IDENTITY_USER_INDEX).
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_identities_issuer_subject ON user_identities (issuer, subject);",
    "CREATE INDEX IF NOT EXISTS ix_user_identities_user_id ON user_identities (user_id);",
)

# As core/models/user_identities.py declares them (SYSTEM_ROLE_MAX, SYSTEM_ROLE_CHECK). Each is
# added only when users lacks it; the CHECK NOT VALID, so no existing row is scanned.
_SYSTEM_ROLE = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'users'::regclass AND attname = 'system_role' AND NOT attisdropped
    ) THEN
        ALTER TABLE users ADD COLUMN system_role VARCHAR(32);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'users'::regclass AND conname = 'ck_users_system_role'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT ck_users_system_role CHECK (system_role IS NULL OR system_role IN ('super_admin', 'admin', 'user')) NOT VALID;
    END IF;
END $$;
"""

_DOWNGRADE = (
    "DROP TABLE IF EXISTS user_identities;",
    "ALTER TABLE users DROP COLUMN IF EXISTS system_role;",
)


def upgrade() -> None:
    for statement in (*_BOUND_LOCK_WAIT, _SYSTEM_ROLE, _TABLE, *_INDEXES, _RESTORE_LOCK_WAIT):
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
