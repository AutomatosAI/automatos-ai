"""harness_task_ledger becomes the governed apply rail's ledger (PRD-263 US-051, 19 Oct 2026).

F156 kept two states here: ``applied`` (a change self-management made) and ``held``
(a [HARNESS] card waiting for /approve). PRD-263 deleted both paths (US-016, US-047).
From the governed apply (US-052) each row is one change proposal: its prescription
(``rx_id``, unique per workspace), run, approval grant (``grant_id``, an integer like
``approval_grants.id``), kind and target agent, the value before and the value
proposed, when it was applied and over what value, and the frozen reference the
post-change check reads (US-056). ``board_task_id`` becomes nullable. The state
CHECK allows the rail's states, and every ``held`` row becomes ``expired``: its
card can no longer be approved. The names match ``core.models.harness.HarnessTaskLedger``.

This alters existing rows (maintainer issue #1151, US-005). The CHECK is dropped
before the ``held`` rewrite and added back after it. ``IF NOT EXISTS`` / ``IF EXISTS``
keep the upgrade safe on a database ``create_all`` built from the models, where the
columns, the CHECK and the index already exist. A legacy row has a NULL ``rx_id``,
which the unique index allows any number of times.

The downgrade deletes no row: it drops the index and the new columns, maps every
state the F156 CHECK does not know to ``held`` (the old not-applied state; DIAGNOSE
reads only ``applied``) and restores that CHECK. ``board_task_id`` stays nullable,
because a row raised without a card has none.

Revision ID: prd263_harness_ledger_grants
Revises: prd262_boot_columns
Create Date: 2026-10-19
"""
from alembic import op

revision = "prd263_harness_ledger_grants"
down_revision = "prd262_boot_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE harness_task_ledger
            ADD COLUMN IF NOT EXISTS rx_id VARCHAR(64),
            ADD COLUMN IF NOT EXISTS run_id INTEGER,
            ADD COLUMN IF NOT EXISTS grant_id INTEGER,
            ADD COLUMN IF NOT EXISTS kind VARCHAR(40),
            ADD COLUMN IF NOT EXISTS target_agent_id INTEGER,
            ADD COLUMN IF NOT EXISTS prior_value JSONB,
            ADD COLUMN IF NOT EXISTS proposed_value JSONB,
            ADD COLUMN IF NOT EXISTS applied_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS value_at_apply JSONB,
            ADD COLUMN IF NOT EXISTS frozen_ref_start TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS frozen_ref_end TIMESTAMPTZ
    """)
    op.execute("ALTER TABLE harness_task_ledger ALTER COLUMN board_task_id DROP NOT NULL")
    op.execute("ALTER TABLE harness_task_ledger DROP CONSTRAINT IF EXISTS ck_harness_task_ledger_state")
    op.execute("UPDATE harness_task_ledger SET state = 'expired' WHERE state = 'held'")
    op.execute("""
        ALTER TABLE harness_task_ledger ADD CONSTRAINT ck_harness_task_ledger_state
            CHECK (state IN ('pending', 'applying', 'applied', 'failed', 'denied', 'expired',
                'drifted', 'withdrawn', 'raise_failed', 'rolled_back'))
    """)
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_harness_task_ledger_rx "
        "ON harness_task_ledger (workspace_id, rx_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_harness_task_ledger_rx")
    op.execute("ALTER TABLE harness_task_ledger DROP CONSTRAINT IF EXISTS ck_harness_task_ledger_state")
    op.execute("UPDATE harness_task_ledger SET state = 'held' WHERE state <> 'applied'")
    op.execute("""
        ALTER TABLE harness_task_ledger ADD CONSTRAINT ck_harness_task_ledger_state
            CHECK (state IN ('applied', 'held'))
    """)
    op.execute("""
        ALTER TABLE harness_task_ledger
            DROP COLUMN IF EXISTS rx_id,
            DROP COLUMN IF EXISTS run_id,
            DROP COLUMN IF EXISTS grant_id,
            DROP COLUMN IF EXISTS kind,
            DROP COLUMN IF EXISTS target_agent_id,
            DROP COLUMN IF EXISTS prior_value,
            DROP COLUMN IF EXISTS proposed_value,
            DROP COLUMN IF EXISTS applied_at,
            DROP COLUMN IF EXISTS value_at_apply,
            DROP COLUMN IF EXISTS frozen_ref_start,
            DROP COLUMN IF EXISTS frozen_ref_end
    """)
