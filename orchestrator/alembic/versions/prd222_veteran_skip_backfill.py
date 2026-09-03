"""PRD-222 test-round fix — veteran workspaces read as brand-new (backfill 'skipped').

Found by Gerard's first live test (2026-08-28): the onboarding column arrived
with PRD-222 and NO backfill marked pre-existing workspaces, while
``get_onboarding`` defaults a missing/stage-less doc to ``not_started`` — so
every veteran workspace (including a year-old super-admin account) rendered the
new-user opener and carried the onboarding spine in Auto's section.

Fix: every workspace WITHOUT a stage at migration time is a veteran (new
signups race-window excepted — they can re-enter via the dev reset). Mark them
``skipped`` (the honest terminal state: they never onboarded), preserving any
other keys in the doc (e.g. ``trial``), stamping ``stages.skipped`` in the same
shape ``advance_onboarding_stage`` uses, plus a ``veteran: true`` marker so the
downgrade touches ONLY rows this migration wrote.

Idempotent: the WHERE matches only stage-less docs; a second run matches none.

Revision ID: prd222_veteran_skip_backfill
Revises: prd222_w2s1_plan_default_basic
Create Date: 2026-08-28
"""
from __future__ import annotations

from alembic import op

revision = "prd222_veteran_skip_backfill"
down_revision = "prd222_w2s1_plan_default_basic"
branch_labels = None
depends_on = None

_UPGRADE_SQL = """
UPDATE workspaces
SET onboarding = COALESCE(onboarding, '{}'::jsonb)
    || jsonb_build_object(
         'stage', 'skipped',
         'veteran', true,
         'stages',
         COALESCE(onboarding->'stages', '{}'::jsonb)
           || jsonb_build_object(
                'skipped',
                to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              )
       )
WHERE (onboarding IS NULL OR onboarding->>'stage' IS NULL)
  -- A veteran predates PRD-222. Anything created after it shipped (2026-08-28)
  -- is a NEW workspace that has simply not started — never stamp it. Found
  -- 2026-09-03: on a fresh local install the orphan-root chains re-run at the
  -- container's `alembic upgrade heads` AFTER the operator workspace exists.
  AND created_at < TIMESTAMP '2026-08-29'
"""

_DOWNGRADE_SQL = """
UPDATE workspaces
SET onboarding = (onboarding - 'stage' - 'veteran')
WHERE onboarding->>'veteran' = 'true'
"""


def upgrade() -> None:
    op.execute(_UPGRADE_SQL)


def downgrade() -> None:
    # Strips ONLY rows this migration marked; their stage becomes unset again.
    op.execute(_DOWNGRADE_SQL)
