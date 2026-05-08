"""Wave 3 — escalation_level column on board_tasks, agent_reports, orchestration_runs

Single L0-L4 ladder so Auto can triage by severity across all three operating
surfaces (tasks, reports, missions) with one query. Nullable; existing rows
untouched. Backfill happens lazily as new events fire.

Idempotent.
"""

from alembic import op


revision = "wave3_escalation_level"
down_revision = "wave1d_mission_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("board_tasks", "agent_reports", "orchestration_runs"):
        op.execute(
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS escalation_level SMALLINT;"
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_board_tasks_escalation "
        "ON board_tasks (workspace_id, escalation_level) "
        "WHERE escalation_level IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_reports_escalation "
        "ON agent_reports (workspace_id, escalation_level) "
        "WHERE escalation_level IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_orchestration_runs_escalation "
        "ON orchestration_runs (workspace_id, escalation_level) "
        "WHERE escalation_level IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_orchestration_runs_escalation;")
    op.execute("DROP INDEX IF EXISTS ix_agent_reports_escalation;")
    op.execute("DROP INDEX IF EXISTS ix_board_tasks_escalation;")
    for table in ("board_tasks", "agent_reports", "orchestration_runs"):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS escalation_level;")
