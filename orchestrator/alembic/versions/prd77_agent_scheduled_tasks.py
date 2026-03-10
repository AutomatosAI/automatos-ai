"""PRD-77: Agent Scheduled Tasks

Create agent_scheduled_tasks table for agent-initiated one-shot and recurring tasks.
Agents call platform_schedule_task to create entries; UnifiedScheduler picks them up.
"""

from alembic import op
import sqlalchemy as sa

revision = "prd77_agent_scheduled_tasks"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_scheduled_tasks (
            id              SERIAL PRIMARY KEY,
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            created_by_agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            target_agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,

            -- Task definition
            task_type       VARCHAR(20) NOT NULL DEFAULT 'one_shot',
            description     TEXT NOT NULL,
            schedule        VARCHAR(100) NOT NULL,
            max_runs        INTEGER,
            run_count       INTEGER NOT NULL DEFAULT 0,

            -- State
            status          VARCHAR(20) NOT NULL DEFAULT 'active',
            last_run_at     TIMESTAMPTZ,
            next_run_at     TIMESTAMPTZ,
            last_error      TEXT,

            -- Timestamps
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_workspace
            ON agent_scheduled_tasks(workspace_id);
        CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_status
            ON agent_scheduled_tasks(workspace_id, status);
        CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_target
            ON agent_scheduled_tasks(target_agent_id);
        CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_next_run
            ON agent_scheduled_tasks(next_run_at)
            WHERE status = 'active';

        -- Constraint: task_type must be valid
        ALTER TABLE agent_scheduled_tasks
            ADD CONSTRAINT ck_scheduled_task_type
            CHECK (task_type IN ('one_shot', 'recurring'));

        -- Constraint: status must be valid
        ALTER TABLE agent_scheduled_tasks
            ADD CONSTRAINT ck_scheduled_task_status
            CHECK (status IN ('active', 'paused', 'completed', 'cancelled', 'failed'));
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_scheduled_tasks;")
