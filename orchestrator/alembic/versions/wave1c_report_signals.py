"""Wave 1.C — Reports as operating signals

Reports today are write-once narratives. To make them an operating layer
Auto can act on, add the structured fields that turn a report into asks:

  - ``recommendations``   jsonb   — list of recommended actions/changes
  - ``action_items``       jsonb   — list of concrete next steps with owners
  - ``linked_task_ids``    jsonb   — board task IDs this report references
  - ``requires_approval``  bool    — does this need a Gerard call?
  - ``acknowledged_by``    int FK  — user who acknowledged it
  - ``acknowledged_at``    tstz    — when it was acknowledged

All nullable / default-empty — existing rows stay valid.

Idempotent.
"""

from alembic import op


revision = "wave1c_report_signals"
down_revision = "wave1b_heartbeat_completion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_reports
        ADD COLUMN IF NOT EXISTS recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS action_items    JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS linked_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS requires_approval BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS acknowledged_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
        ADD COLUMN IF NOT EXISTS acknowledged_at   TIMESTAMPTZ;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_reports_requires_approval "
        "ON agent_reports (workspace_id, requires_approval) "
        "WHERE requires_approval = TRUE;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_reports_unacknowledged "
        "ON agent_reports (workspace_id, acknowledged_at) "
        "WHERE acknowledged_at IS NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_reports_unacknowledged;")
    op.execute("DROP INDEX IF EXISTS ix_agent_reports_requires_approval;")
    op.execute(
        """
        ALTER TABLE agent_reports
        DROP COLUMN IF EXISTS acknowledged_at,
        DROP COLUMN IF EXISTS acknowledged_by,
        DROP COLUMN IF EXISTS requires_approval,
        DROP COLUMN IF EXISTS linked_task_ids,
        DROP COLUMN IF EXISTS action_items,
        DROP COLUMN IF EXISTS recommendations;
        """
    )
