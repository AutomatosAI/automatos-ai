"""mobile_devices — one row per phone or paired browser (PRD-266 §4.2, 20 Oct 2026).

Automatos Go pairs a phone with a workspace. Two kinds share the table: ``paired``
(the local edition: the phone, or a browser, paired with a one-time code and holds a
device token) and ``clerk`` (SaaS: the phone signs in with Clerk and registers for push
only, so it never holds a token hash: ``ck_mobile_devices_clerk_holds_no_token``).
A pending row is a code waiting for a phone, so it has no ``install_id`` yet; a phone's
``install_id`` is unique per workspace and person. Codes and tokens are stored only as
SHA-256 hashes, each unique: a code is found with no workspace, so it may lead to one
pending row only. After a rotation the previous hash stays valid until
``prev_token_valid_until``. ``expo_push_token`` and ``push_key`` serve push (W2);
``events`` keeps the device's recent audit trail. The names match
``core.models.mobile_devices.MobileDevice``.

Why not ``cli_hosts``: it registers CLI runtimes (capabilities, heartbeats, terminals,
sessions, local only); a phone carries a person, push credentials and digest
preferences, and mixing the two lifecycles would make every query branch on kind.

A new table only: no existing table or row changes. ``IF NOT EXISTS`` keeps the
upgrade a no-op on a database that ``create_all`` built from the models, which then
gains only a missing index.

Revision ID: prd266_mobile_devices
Revises: clerk_user_relinks_20261019
Create Date: 2026-10-20
"""
from alembic import op

revision = "prd266_mobile_devices"
down_revision = "prd265_credit_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS mobile_devices (
            id UUID PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind VARCHAR(16) NOT NULL,
            install_id VARCHAR(128),
            name VARCHAR(120),
            platform VARCHAR(16),
            app_version VARCHAR(32),
            locale VARCHAR(64),
            timezone VARCHAR(64),
            digest_time VARCHAR(5),
            status VARCHAR(16) NOT NULL DEFAULT 'pending',
            pairing_code_hash VARCHAR(64),
            pairing_expires_at TIMESTAMPTZ,
            token_hash VARCHAR(64),
            token_rotated_at TIMESTAMPTZ,
            prev_token_hash VARCHAR(64),
            prev_token_valid_until TIMESTAMPTZ,
            expo_push_token VARCHAR(255),
            push_enabled BOOLEAN NOT NULL DEFAULT false,
            push_disabled_reason VARCHAR(64),
            push_key VARCHAR(128),
            last_seen_at TIMESTAMPTZ,
            paired_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ,
            revoked_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            events JSONB NOT NULL DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_mobile_devices_workspace_user_install UNIQUE (workspace_id, user_id, install_id),
            CONSTRAINT uq_mobile_devices_token_hash UNIQUE (token_hash),
            CONSTRAINT uq_mobile_devices_expo_push_token UNIQUE (expo_push_token),
            CONSTRAINT uq_mobile_devices_pairing_code_hash UNIQUE (pairing_code_hash),
            CONSTRAINT ck_mobile_devices_kind CHECK (kind IN ('paired', 'clerk')),
            CONSTRAINT ck_mobile_devices_status CHECK (status IN ('pending', 'active', 'revoked')),
            CONSTRAINT ck_mobile_devices_platform CHECK (platform IS NULL OR platform IN ('ios', 'android', 'web')),
            CONSTRAINT ck_mobile_devices_clerk_holds_no_token CHECK (kind = 'paired' OR (token_hash IS NULL AND prev_token_hash IS NULL AND pairing_code_hash IS NULL))
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_mobile_devices_workspace_status "
        "ON mobile_devices (workspace_id, status);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_mobile_devices_prev_token_hash "
        "ON mobile_devices (prev_token_hash);"
    )


def downgrade() -> None:
    # The table's indexes and constraints go with it.
    op.execute("DROP TABLE IF EXISTS mobile_devices;")
