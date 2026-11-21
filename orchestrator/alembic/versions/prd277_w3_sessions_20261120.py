"""The sessions Automatos issues after a person signs in (20 Nov 2026).

PRD-277, enterprise sign-in, Wave 3 (§5.3, US-301). The ONE migration of the wave:

* ``auth_sessions`` (a new table): one row per session, linked to its person, ON DELETE CASCADE
  with them. It holds the SHA-256 of the session's token, never the token; the two clocks (idle
  and absolute expiry), when the person last proved who they are (``auth_time``), the identity
  provider's ``sid``, when and why the session was revoked, and the browser's address and agent.
  ``ck_auth_sessions_revoked_with_reason`` keeps ``revoked_at`` and ``revoked_reason`` together.
* ``uq_auth_sessions_token_hash`` on ``token_hash``, unique: the resolver's one lookup, and one
  hash names at most one session. ``ix_auth_sessions_user_id`` on ``user_id``: revoking every
  session of a person, listing them, and the cascade.

``core/models/auth_sessions.py`` declares the same names. Create_all-first safe: ``CREATE TABLE IF
NOT EXISTS`` and ``CREATE [UNIQUE] INDEX IF NOT EXISTS``, so a database ``create_all`` built ends
exactly as it was, and running it twice changes nothing. No row is written or changed.

The locks, and why none scans or rewrites ``users``:

* ``CREATE TABLE auth_sessions`` (only when it is missing): its foreign key takes SHARE ROW
  EXCLUSIVE on ``users`` to add the key's triggers, held for a catalog change only; nothing in
  ``users`` is read, as the new table is empty. Skipped, with no lock on ``users``, when the table
  exists.
* The two ``CREATE [UNIQUE] INDEX`` statements: a SHARE lock on ``auth_sessions`` while the
  table is still empty (it was created just above); a no-op when ``create_all`` already built
  both indexes. Never a lock on ``users``.

The downgrade drops the table (its indexes and CHECK go with it), and only it. Every session it
held ends: each person signs in again.

Revision ID: prd277_w3_sessions_20261120
Revises: prd277_w2_identities_20261120
Create Date: 2026-11-20
"""
from alembic import op

revision = "prd277_w3_sessions_20261120"
down_revision = "prd277_w2_identities_20261120"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMP WITH TIME ZONE,
    idle_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    absolute_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    auth_time TIMESTAMP WITH TIME ZONE NOT NULL,
    idp_session_id VARCHAR(255),
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason VARCHAR(32),
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    CONSTRAINT ck_auth_sessions_revoked_with_reason CHECK ((revoked_at IS NULL) = (revoked_reason IS NULL))
);
"""

# As core/models/auth_sessions.py declares them (TOKEN_HASH_INDEX, USER_INDEX).
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_auth_sessions_token_hash ON auth_sessions (token_hash);",
    "CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_id ON auth_sessions (user_id);",
)

_DOWNGRADE = ("DROP TABLE IF EXISTS auth_sessions;",)


def upgrade() -> None:
    for statement in (_TABLE, *_INDEXES):
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
