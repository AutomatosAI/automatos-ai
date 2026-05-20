"""Observability Decision Layer — trace_id + evidence on agent_reports

Two columns turn reports into reproducible artifacts:

  - ``trace_id``   text   — correlation ID for all log lines + LLM calls
                            within the same logical operation. Lets you
                            answer "show me everything that ran inside
                            this report's work."
  - ``evidence``    jsonb  — list of {tool, query, window, sample_count,
                            top_signature} records that backed the
                            report's conclusions. Lets a reviewer (or
                            HARNESS PRESCRIBE) re-run the queries.

Both nullable / default-empty — existing rows stay valid. Idempotent.
"""

from alembic import op


revision = "observability_evidence"
down_revision = "prd140_team_lead_enabled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_reports
        ADD COLUMN IF NOT EXISTS trace_id TEXT,
        ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '[]'::jsonb;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_reports_trace_id "
        "ON agent_reports(trace_id) WHERE trace_id IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_reports_trace_id;")
    op.execute(
        "ALTER TABLE agent_reports "
        "DROP COLUMN IF EXISTS evidence, "
        "DROP COLUMN IF EXISTS trace_id;"
    )
