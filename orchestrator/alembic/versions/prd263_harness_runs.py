"""harness_runs — one row per HARNESS run, in the database (PRD-263 US-009, 18 Oct 2026).

HARNESS kept its run state in JSON files under <WORKSPACE_VOLUME_PATH>/<ws>/harness
(last_run.json, traces) and guarded a run with an in-memory dict that only one
uvicorn worker could see. Each run is now one ``harness_runs`` row: the ISO week it
judges (``cur_window``), when it started and finished, its status (a CHECK
constraint) and iteration, coverage, counts, verdicts, prescriptions and the audit
report with its outcome. The names match ``core.models.harness.HarnessRun``.

A new table only: no existing table or row changes. ``IF NOT EXISTS`` keeps the
upgrade a no-op on a database that ``create_all`` built from the models, which
then gains only a missing index.

Revision ID: prd263_harness_runs
Revises: prd258_voice_timings
Create Date: 2026-10-18
"""
from alembic import op

revision = "prd263_harness_runs"
down_revision = "prd258_voice_timings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS harness_runs (
            id SERIAL PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            cur_window VARCHAR(10) NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            status VARCHAR(40) NOT NULL,
            iteration INTEGER NOT NULL DEFAULT 0,
            coverage DOUBLE PRECISION,
            counts JSONB,
            verdicts JSONB,
            prescriptions JSONB,
            report_md TEXT,
            report_status TEXT,
            CONSTRAINT ck_harness_runs_status CHECK (status IN ('running', 'completed', 'failed',
                'dormant_insufficient_agents', 'dormant_insufficient_data'))
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_harness_runs_ws_window "
        "ON harness_runs (workspace_id, cur_window);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_harness_runs_ws_window;")
    op.execute("DROP TABLE IF EXISTS harness_runs;")
