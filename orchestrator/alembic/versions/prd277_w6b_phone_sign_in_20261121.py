"""A phone can sign in at the install's own OIDC start address (21 Nov 2026).

PRD-277, enterprise sign-in, Wave 6 part 2 (US-603, §5.7, D-8). The ONE migration of the part:

* ``mobile_devices.ck_mobile_devices_kind`` allows ``oidc`` beside ``paired`` and ``clerk``: a device
  whose person signed in on the phone. ``ck_mobile_devices_clerk_holds_no_token`` lets an ``oidc``
  row hold a code and a token hash as a ``paired`` row does; a ``clerk`` row still holds neither.
  Each is dropped when it exists and added back ``NOT VALID``, then validated, so a database
  ``create_all`` built (which already has the wider checks) ends exactly as it was. Every existing
  row is ``paired`` or ``clerk``, so validating the wider checks finds no row that breaks them.
* ``mobile_devices.sign_in_challenge``, ``sign_in_install_id``, ``sign_in_app_link`` and
  ``sign_in_confirm_hash`` (new nullable columns): on a pending ``oidc`` row only, the app's PKCE
  challenge, install id and app link its one-time code is bound to, and, while the sign-in awaits
  the person's confirmation, the SHA-256 of the token in their browser's cookie.
* ``oidc_pending_sign_ins.phone_app_link``, ``phone_code_challenge``, ``phone_install_id`` and
  ``phone_platform`` (new nullable columns): what a phone's start asked for, kept for its callback.

``core/models/mobile_devices.py`` and ``core/models/oidc_pending_sign_ins.py`` declare the same names.
Create_all-first safe: ``DROP CONSTRAINT IF EXISTS`` then ``ADD``, and ``ADD COLUMN IF NOT EXISTS``;
running it twice changes nothing. No row is written or changed.

Locks: each ``ALTER TABLE`` takes an ACCESS EXCLUSIVE lock, held until the upgrade's one transaction
commits (``alembic/env.py`` runs every pending revision in one). ``VALIDATE CONSTRAINT`` reads ``mobile_devices`` once (one
row per phone); a column with no default rewrites nothing. No ``lock_timeout`` is set: no revision
here sets one (``prd268_w2_delivery`` says the same).

The downgrade drops the eight columns and puts the narrower checks back as ``NOT VALID``: they hold for
every row written after it, and an ``oidc`` row already there is kept as it is (the code of the
revision before never reads one: its device lane looks up ``paired`` rows only).

Revision ID: prd277_w6b_phone_sign_in_20261121
Revises: prd277_w4b_step_up_20261120
Create Date: 2026-11-21
"""
from alembic import op

revision = "prd277_w6b_phone_sign_in_20261121"
down_revision = "prd277_w4b_step_up_20261120"
branch_labels = None
depends_on = None

_TOKEN_FREE = "(token_hash IS NULL AND prev_token_hash IS NULL AND pairing_code_hash IS NULL)"
_UPGRADE = (
    "ALTER TABLE mobile_devices DROP CONSTRAINT IF EXISTS ck_mobile_devices_kind;",
    "ALTER TABLE mobile_devices ADD CONSTRAINT ck_mobile_devices_kind "
    "CHECK (kind IN ('paired', 'clerk', 'oidc')) NOT VALID;",
    "ALTER TABLE mobile_devices VALIDATE CONSTRAINT ck_mobile_devices_kind;",
    "ALTER TABLE mobile_devices DROP CONSTRAINT IF EXISTS ck_mobile_devices_clerk_holds_no_token;",
    "ALTER TABLE mobile_devices ADD CONSTRAINT ck_mobile_devices_clerk_holds_no_token "
    f"CHECK (kind IN ('paired', 'oidc') OR {_TOKEN_FREE}) NOT VALID;",
    "ALTER TABLE mobile_devices VALIDATE CONSTRAINT ck_mobile_devices_clerk_holds_no_token;",
    "ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS sign_in_challenge VARCHAR(43);",
    "ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS sign_in_install_id VARCHAR(128);",
    "ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS sign_in_app_link VARCHAR(16);",
    "ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS sign_in_confirm_hash VARCHAR(64);",
    "ALTER TABLE oidc_pending_sign_ins ADD COLUMN IF NOT EXISTS phone_app_link VARCHAR(16);",
    "ALTER TABLE oidc_pending_sign_ins ADD COLUMN IF NOT EXISTS phone_code_challenge VARCHAR(43);",
    "ALTER TABLE oidc_pending_sign_ins ADD COLUMN IF NOT EXISTS phone_install_id VARCHAR(128);",
    "ALTER TABLE oidc_pending_sign_ins ADD COLUMN IF NOT EXISTS phone_platform VARCHAR(16);",
)
_DOWNGRADE = (
    "ALTER TABLE oidc_pending_sign_ins DROP COLUMN IF EXISTS phone_platform;",
    "ALTER TABLE oidc_pending_sign_ins DROP COLUMN IF EXISTS phone_install_id;",
    "ALTER TABLE oidc_pending_sign_ins DROP COLUMN IF EXISTS phone_code_challenge;",
    "ALTER TABLE oidc_pending_sign_ins DROP COLUMN IF EXISTS phone_app_link;",
    "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS sign_in_confirm_hash;",
    "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS sign_in_app_link;",
    "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS sign_in_install_id;",
    "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS sign_in_challenge;",
    "ALTER TABLE mobile_devices DROP CONSTRAINT IF EXISTS ck_mobile_devices_clerk_holds_no_token;",
    "ALTER TABLE mobile_devices ADD CONSTRAINT ck_mobile_devices_clerk_holds_no_token "
    f"CHECK (kind = 'paired' OR {_TOKEN_FREE}) NOT VALID;",
    "ALTER TABLE mobile_devices DROP CONSTRAINT IF EXISTS ck_mobile_devices_kind;",
    "ALTER TABLE mobile_devices ADD CONSTRAINT ck_mobile_devices_kind "
    "CHECK (kind IN ('paired', 'clerk')) NOT VALID;",
)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
