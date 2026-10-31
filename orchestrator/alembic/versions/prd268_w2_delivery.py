"""The suggestion push type and mobile_devices.push_categories — Proactive Auto's delivery (PRD-268 §6.5.3, §6.12.5, 28 Oct 2026).

The ONE migration of wave W2 (Delivery); every later story in the wave adds none.

* ``ck_mobile_push_tickets_push_type`` widens to four push types: ``suggestion`` joins
  ``needs_answer``, ``review`` and ``auto_message`` (PRD-266 D22, revised by PRD-268 K9).
  The push itself ships off (config ``SUGGESTION_PUSH_ENABLED``).
* ``mobile_devices.push_categories``: the optional push categories the phone's app
  registered (``["suggestion"]`` once automatos-go ships that category). A suggestion
  push goes only to a device whose list holds ``suggestion``, so an app that cannot show
  one is never sent one. Every existing phone registered none: ``DEFAULT '[]'``.

Why this is safe on a live database:

* Re-adding the CHECK holds an ACCESS EXCLUSIVE lock on ``mobile_push_tickets`` while it
  scans the table. The table holds a day of Expo tickets at most (the receipts job reads
  and prunes them), so the scan is short, and every row in it already holds one of the
  three old types (the old CHECK enforced them), which the wider CHECK admits. The drop
  and the add are one ``ALTER TABLE`` statement: the lock is taken once and no moment
  passes without a CHECK.
* ``ADD COLUMN … NOT NULL DEFAULT '[]'`` with a constant default is metadata-only on
  Postgres 11+: no table rewrite, and no row is written.
* Both locks are held until the migration's transaction commits (``alembic/env.py`` runs
  the pending revisions in one transaction), and each statement is brief once its lock is
  granted. No ``lock_timeout`` is set: no revision here sets one, and a ``SET LOCAL``
  would bind every later revision in the same run.

No existing row changes. ``DROP CONSTRAINT IF EXISTS`` before the ``ADD`` and ``ADD
COLUMN IF NOT EXISTS`` keep the upgrade safe on a database that ``create_all`` built from
the models (``core.models.mobile_push`` and ``core.models.mobile_devices`` declare the
same names), and running it twice changes nothing.
``tests/test_prd268w2_delivery_migration.py`` holds the two sides together.

The downgrade deletes the ``suggestion`` tickets first (transient, a day at most, and the
three-type CHECK would refuse them), puts back the three-type CHECK and drops the column
(the categories phones registered go with it; they register them again). Run it with the
suggestion push off, its default: a suggestion ticket written between the DELETE and the
ADD fails the downgrade, which then rolls back whole.

Revision ID: prd268_w2_delivery
Revises: prd268_w1_proactive
Create Date: 2026-10-28
"""
from alembic import op

revision = "prd268_w2_delivery"
down_revision = "prd268_w1_proactive"
branch_labels = None
depends_on = None

_UPGRADE = (
    "ALTER TABLE mobile_push_tickets DROP CONSTRAINT IF EXISTS ck_mobile_push_tickets_push_type, "
    "ADD CONSTRAINT ck_mobile_push_tickets_push_type "
    "CHECK (push_type IN ('needs_answer', 'review', 'auto_message', 'suggestion'));",
    "ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS push_categories JSONB NOT NULL DEFAULT '[]';",
)

_DOWNGRADE = (
    "DELETE FROM mobile_push_tickets WHERE push_type = 'suggestion';",
    "ALTER TABLE mobile_push_tickets DROP CONSTRAINT IF EXISTS ck_mobile_push_tickets_push_type, "
    "ADD CONSTRAINT ck_mobile_push_tickets_push_type "
    "CHECK (push_type IN ('needs_answer', 'review', 'auto_message'));",
    "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS push_categories;",
)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
