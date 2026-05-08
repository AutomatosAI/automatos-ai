"""Wave 1.D — Mission lifecycle fields

OrchestrationRun has goal/state/plan/output_summary already. Auto needs four
more fields to manage missions like a real CTO:

  - ``linked_prd``         text   — PRD or issue reference, e.g. "PRD-121"
  - ``completion_evidence`` jsonb  — proof of done (artefacts, links)
  - ``deadline``            tstz   — target completion time
  - ``risks``               jsonb  — recorded risk register

Acceptance criteria already lives in ``OrchestrationTask.verification_criteria``;
not duplicating it on the run.

Idempotent.
"""

from alembic import op


revision = "wave1d_mission_lifecycle"
down_revision = "wave1c_report_signals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE orchestration_runs
        ADD COLUMN IF NOT EXISTS linked_prd          TEXT,
        ADD COLUMN IF NOT EXISTS completion_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS deadline            TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS risks               JSONB NOT NULL DEFAULT '[]'::jsonb;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_orchestration_runs_deadline "
        "ON orchestration_runs (workspace_id, deadline) "
        "WHERE deadline IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_orchestration_runs_deadline;")
    op.execute(
        """
        ALTER TABLE orchestration_runs
        DROP COLUMN IF EXISTS risks,
        DROP COLUMN IF EXISTS deadline,
        DROP COLUMN IF EXISTS completion_evidence,
        DROP COLUMN IF EXISTS linked_prd;
        """
    )
