"""Wave 1.B — Heartbeat completion semantics

A heartbeat row today says "ran" with status + findings + actions. It does
not say "did the agent actually do the thing it was supposed to do?". Adds
two cheap fields:

  - ``objective_met``  bool    — explicit yes/no/null outcome
  - ``evidence_ref``   text    — pointer to the artefact that proves it
                                  (workspace file path, report id, task id, etc.)

The column is nullable so old rows stay valid; new ticks populate it where
the heartbeat has a clear objective (e.g. "submit a daily report" → check
report exists; "review failed tasks" → check actions_taken non-empty).

Idempotent.
"""

from alembic import op


revision = "wave1b_heartbeat_completion"
down_revision = "wave1a_agent_responsibilities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE heartbeat_results
        ADD COLUMN IF NOT EXISTS objective_met BOOLEAN;
        """
    )
    op.execute(
        """
        ALTER TABLE heartbeat_results
        ADD COLUMN IF NOT EXISTS evidence_ref TEXT;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE heartbeat_results DROP COLUMN IF EXISTS evidence_ref;")
    op.execute("ALTER TABLE heartbeat_results DROP COLUMN IF EXISTS objective_met;")
