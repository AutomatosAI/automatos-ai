"""PRD-222 W2·S1 (US-023) — workspaces.plan default 'starter' → 'basic' + backfill.

The ONE migration for PRD-222 Wave 2b. The tier contract (``config.PLAN_TIERS``)
renames the entry tier's name to ``basic``; this migration brings live databases
in line:

  * sets the ``workspaces.plan`` server default to ``'basic'`` (future inserts),
  * backfills every existing ``plan = 'starter'`` row to ``'basic'``.

Both statements are idempotent and set-based: re-running SET DEFAULT is a no-op,
and the backfill's ``WHERE plan = 'starter'`` matches nothing on a second run.
A row already on another tier (``pro`` / ``business`` / …) is untouched — the
predicate is exactly ``plan = 'starter'``. No ``plan_limits`` are rewritten here:
tier limits are applied only on explicit plan assignment (US-025, via
``services/plan_tiers.assign_plan``); this migration touches the plan STRING
only. Nothing else — Wave 2b's single schema change.

Chains single-parent on the current head ``prd222_w2s5_drop_onboarding_agents``.

Revision ID: prd222_w2s1_plan_default_basic
Revises: prd222_w2s5_drop_onboarding_agents
Create Date: 2026-08-28
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "prd222_w2s1_plan_default_basic"
down_revision = "prd222_w2s5_drop_onboarding_agents"
branch_labels = None
depends_on = None

_OLD_DEFAULT = "starter"
_NEW_DEFAULT = "basic"
# The exact idempotent backfill the @integration test replays.
_BACKFILL_SQL = f"UPDATE workspaces SET plan = '{_NEW_DEFAULT}' WHERE plan = '{_OLD_DEFAULT}'"


def upgrade() -> None:
    # Future inserts default to the renamed entry tier.
    op.alter_column(
        "workspaces",
        "plan",
        existing_type=sa.String(length=50),
        server_default=_NEW_DEFAULT,
    )
    # HOTFIX (2026-08-28 deploy crash-loop): production carries a LEGACY
    # value-list check constraint `workspaces_plan_check` that predates the tier
    # rename and exists in NO migration (prod-only schema drift — CI schemas are
    # built from the chain and never had it, so every test lane was green while
    # the deploy loop failed). It rejects 'basic'; drop it before the backfill.
    # Deliberately NOT recreated: plan validity is enforced in code against
    # config.PLAN_TIERS (the US-025 assignable set) — a DB value-list would
    # re-drift on the next config tier change. IF EXISTS keeps this a no-op on
    # databases that never had the constraint (CI, fresh installs).
    op.execute("ALTER TABLE workspaces DROP CONSTRAINT IF EXISTS workspaces_plan_check")

    # Bring existing rows onto the renamed tier (idempotent; only 'starter').
    op.execute(_BACKFILL_SQL)


def downgrade() -> None:
    # Restore the pre-rename server default only. The data backfill is
    # deliberately NOT reversed: after the rename, a 'basic' row that was
    # explicitly ASSIGNED (US-025) is indistinguishable from one migrated here,
    # so a blanket 'basic'→'starter' would clobber real assignments. Reverting
    # the default is the reversible part of a rename.
    op.alter_column(
        "workspaces",
        "plan",
        existing_type=sa.String(length=50),
        server_default=_OLD_DEFAULT,
    )
