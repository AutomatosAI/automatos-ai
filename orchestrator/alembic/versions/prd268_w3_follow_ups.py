"""Follow-up rows, the posted outcome and proactive_calibrations — Proactive Auto's follow-ups (PRD-268 §6.3, §6.11.5, 29 Oct 2026).

The ONE migration of wave W3 (Follow-ups); every later story in the wave adds none.
It does four things (K6) and nothing else:

* ``ck_scheduled_task_deliver_as`` widens to ``follow_up``: a follow-up is one
  ``agent_scheduled_tasks`` row that Auto schedules in a person's own chat turn, fired
  once in the check-in lane. ``deliver_as`` and ``origin_chat_id`` already exist
  (``calendar_scheduled_board_tasks``, ``prd205_auto_speaks``).
* ``agent_scheduled_tasks.for_user_id``: the person whose chat turn scheduled the
  follow-up. Nullable: every existing row is nobody's. The partial index
  ``ix_scheduled_tasks_follow_ups`` (workspace, person, created_at), over follow-up rows
  only, serves the caps (pending per person, per person per day, per workspace per day)
  and the person's own list.
* ``ck_proactive_runs_outcome`` gains ``posted``: a follow-up that said something.
* ``proactive_calibrations``: one row per person and workspace (the unique is the
  offered-once claim): the status (declined, open, answered), what Auto said, the lines
  the person rejected and accepted, and the message it was posted as. ``user_id``,
  ``chat_id`` and ``message_id`` are plain values, not foreign keys: the record outlives
  the chat it was posted in. The workspace CASCADEs, and the unique leads with it, so
  deleting a workspace never scans the table.

Why this is safe on a live database:

* Re-adding a CHECK holds an ACCESS EXCLUSIVE lock on its table while it scans it.
  ``agent_scheduled_tasks`` holds the workspaces' schedules and ``proactive_runs`` the
  check-in audit; both are small, so each scan is short, and every row in them already
  holds a value the wider CHECK admits (the old CHECK enforced it). Each drop and add is
  one ``ALTER TABLE`` statement: the lock is taken once and no moment passes without a
  CHECK.
* ``ADD COLUMN`` of a nullable column with no default is metadata-only: no table
  rewrite, and no row is written. The partial index is built with a plain
  ``CREATE INDEX`` (``alembic/env.py`` runs the revisions in one transaction, where
  ``CONCURRENTLY`` cannot run); it covers no existing row, since none is a follow-up.
* Every lock is held until the migration's transaction commits. No ``lock_timeout`` is
  set: no revision here sets one, and a ``SET LOCAL`` would bind every later revision in
  the same run.

No existing row changes. ``DROP CONSTRAINT IF EXISTS`` before each ``ADD``, ``ADD COLUMN
IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS`` and ``CREATE TABLE IF NOT EXISTS`` keep the
upgrade safe on a database that ``create_all`` built from the models
(``core.models.proactive`` declares the same names; ``agent_scheduled_tasks`` has no
model and lives in the revisions only), and running it twice changes nothing.
``tests/test_prd268w3_migration.py`` holds the two sides together.

The downgrade deletes the follow-up rows and the posted runs first (transient: a
follow-up fires within days, and the narrower CHECKs would refuse both), puts back both
CHECKs, and drops the index, the column and the table (the calibrations go with it).
Run it with no follow-up pending: one scheduled between the DELETE and the ADD fails the
downgrade, which then rolls back whole.

Revision ID: prd268_w3_follow_ups
Revises: prd268_w2_delivery
Create Date: 2026-10-29
"""
from alembic import op

revision = "prd268_w3_follow_ups"
down_revision = "prd268_w2_delivery"
branch_labels = None
depends_on = None

_UPGRADE = (
    "ALTER TABLE agent_scheduled_tasks DROP CONSTRAINT IF EXISTS ck_scheduled_task_deliver_as, "
    "ADD CONSTRAINT ck_scheduled_task_deliver_as "
    "CHECK (deliver_as IN ('chat', 'board_task', 'follow_up'));",
    "ALTER TABLE agent_scheduled_tasks ADD COLUMN IF NOT EXISTS for_user_id INTEGER;",
    "CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_follow_ups "
    "ON agent_scheduled_tasks (workspace_id, for_user_id, created_at) WHERE deliver_as = 'follow_up';",
    "ALTER TABLE proactive_runs DROP CONSTRAINT IF EXISTS ck_proactive_runs_outcome, "
    "ADD CONSTRAINT ck_proactive_runs_outcome "
    "CHECK (outcome IN ('silent', 'suggested', 'error', 'skipped_quiet', 'skipped_cap', "
    "'skipped_credits', 'skipped_paused', 'skipped_off', 'posted'));",
    """
CREATE TABLE IF NOT EXISTS proactive_calibrations (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'open',
    calibration JSONB NOT NULL DEFAULT '{}',
    rejected_lines JSONB NOT NULL DEFAULT '[]',
    accepted_lines JSONB NOT NULL DEFAULT '[]',
    chat_id UUID,
    message_id UUID,
    consented_at TIMESTAMP WITH TIME ZONE,
    decided_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_proactive_calibrations_status CHECK (status IN ('declined', 'open', 'answered')),
    CONSTRAINT uq_proactive_calibrations_person UNIQUE (workspace_id, user_id)
);
""",
)

_DOWNGRADE = (
    "DELETE FROM agent_scheduled_tasks WHERE deliver_as = 'follow_up';",
    "DELETE FROM proactive_runs WHERE outcome = 'posted';",
    "ALTER TABLE agent_scheduled_tasks DROP CONSTRAINT IF EXISTS ck_scheduled_task_deliver_as, "
    "ADD CONSTRAINT ck_scheduled_task_deliver_as "
    "CHECK (deliver_as IN ('chat', 'board_task'));",
    "ALTER TABLE proactive_runs DROP CONSTRAINT IF EXISTS ck_proactive_runs_outcome, "
    "ADD CONSTRAINT ck_proactive_runs_outcome "
    "CHECK (outcome IN ('silent', 'suggested', 'error', 'skipped_quiet', 'skipped_cap', "
    "'skipped_credits', 'skipped_paused', 'skipped_off'));",
    "DROP INDEX IF EXISTS ix_scheduled_tasks_follow_ups;",
    "ALTER TABLE agent_scheduled_tasks DROP COLUMN IF EXISTS for_user_id;",
    "DROP TABLE IF EXISTS proactive_calibrations;",
)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
