"""Sign-ins that have started and not finished yet (20 Nov 2026).

PRD-277, enterprise sign-in, Wave 4 (US-402, §5.2). The ONE migration of the wave's first part:

* ``oidc_pending_sign_ins`` (a new table): one row per sign-in sent to the organisation's identity
  provider and not back yet, kept at most ten minutes and used once. It holds the SHA-256 of the
  ``state``, of the ``nonce`` and of the browser binding (the secret in the browser's sign-in cookie),
  never any of them; the SHA-256 of the client address that started it, never the address; the
  PKCE verifier, which the code exchange must present; and where the person returns afterwards. It names no person and no workspace, so it has
  no foreign key.
* ``uq_oidc_pending_sign_ins_state_hash`` on ``state_hash``, unique: the callback's one lookup, and
  one state names at most one sign-in. ``ix_oidc_pending_sign_ins_expires_at`` on ``expires_at``:
  clearing sign-ins that were started and never finished, oldest first.
  ``ix_oidc_pending_sign_ins_client_hash_expires_at`` on ``(client_hash, expires_at)``: one
  address's sign-ins, oldest first, for the per-address bound.

``core/models/oidc_pending_sign_ins.py`` declares the same names. Create_all-first safe: ``CREATE
TABLE IF NOT EXISTS`` and ``CREATE [UNIQUE] INDEX IF NOT EXISTS``, so a database ``create_all`` built
ends exactly as it was, and running it twice changes nothing. No row is written or changed.

Locks: ``CREATE TABLE`` takes none on another table (no foreign key); the three indexes take a SHARE
lock on the new, empty table, or are no-ops when ``create_all`` built them.

The downgrade drops the table (its indexes go with it), and only it. A sign-in in flight then fails
at its callback, and the person starts again.

Revision ID: prd277_w4_oidc_sign_ins_20261120
Revises: prd277_w3_sessions_20261120
Create Date: 2026-11-20
"""
from alembic import op

revision = "prd277_w4_oidc_sign_ins_20261120"
down_revision = "prd277_w3_sessions_20261120"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS oidc_pending_sign_ins (
    id UUID PRIMARY KEY,
    state_hash VARCHAR(64) NOT NULL,
    nonce_hash VARCHAR(64) NOT NULL,
    binding_hash VARCHAR(64) NOT NULL,
    client_hash VARCHAR(64) NOT NULL,
    code_verifier VARCHAR(128) NOT NULL,
    return_to VARCHAR(1024) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);
"""

# As core/models/oidc_pending_sign_ins.py declares them (STATE_HASH_INDEX, EXPIRES_INDEX, CLIENT_INDEX).
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_oidc_pending_sign_ins_state_hash ON oidc_pending_sign_ins (state_hash);",
    "CREATE INDEX IF NOT EXISTS ix_oidc_pending_sign_ins_expires_at ON oidc_pending_sign_ins (expires_at);",
    "CREATE INDEX IF NOT EXISTS ix_oidc_pending_sign_ins_client_hash_expires_at "
    "ON oidc_pending_sign_ins (client_hash, expires_at);",
)

_DOWNGRADE = ("DROP TABLE IF EXISTS oidc_pending_sign_ins;",)


def upgrade() -> None:
    for statement in (_TABLE, *_INDEXES):
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
