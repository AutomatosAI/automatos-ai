"""A started sign-in remembers the session that asked for a step-up (20 Nov 2026).

PRD-277, enterprise sign-in, Wave 4 part 2 (US-406, §5.5). The ONE migration of the part:

* ``oidc_pending_sign_ins.step_up_session_id`` (a new nullable ``UUID`` column): set only on a
  sign-in started by ``GET /api/v1/auth/oidc/start?step_up=1``, to the id of the live session that
  asked. The callback then refreshes that session's ``auth_time`` instead of issuing a new session,
  and only when the person who signed in again is that session's person. NULL on every other
  sign-in, so every row written before this revision means what it meant.

No foreign key: the callback looks the session up, locked, and refuses a step-up whose session has
ended or is gone, so a key would add a lock on ``auth_sessions`` and nothing else.

``core/models/oidc_pending_sign_ins.py`` declares the same column. Create_all-first safe: ``ADD
COLUMN IF NOT EXISTS``, so a database ``create_all`` built ends exactly as it was, and running it
twice changes nothing. No row is written or changed.

Locks: ``ALTER TABLE … ADD COLUMN`` with no default takes a brief ACCESS EXCLUSIVE lock on
``oidc_pending_sign_ins`` (rows that live at most ten minutes) and rewrites nothing.

The downgrade drops the column, and only it; the code of the revision before never reads it.

Revision ID: prd277_w4b_step_up_20261120
Revises: prd277_w4_oidc_sign_ins_20261120
Create Date: 2026-11-20
"""
from alembic import op

revision = "prd277_w4b_step_up_20261120"
down_revision = "prd277_w4_oidc_sign_ins_20261120"
branch_labels = None
depends_on = None

_UPGRADE = ("ALTER TABLE oidc_pending_sign_ins ADD COLUMN IF NOT EXISTS step_up_session_id UUID;",)
_DOWNGRADE = ("ALTER TABLE oidc_pending_sign_ins DROP COLUMN IF EXISTS step_up_session_id;",)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
