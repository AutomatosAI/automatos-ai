"""PRD-72: Board Tasks

Create board_tasks table for the lightweight task board system.
Tasks follow a Kanban lifecycle: inbox -> assigned -> in_progress -> review -> done.
Agents pick up assigned tasks during heartbeats.
"""

from alembic import op
import sqlalchemy as sa

revision = "prd72_board_tasks"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS board_tasks (
            id              SERIAL PRIMARY KEY,
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

            -- Task content
            title           VARCHAR(500) NOT NULL,
            description     TEXT,
            raw_prompt      TEXT,

            -- Status lifecycle: inbox -> assigned -> in_progress -> review -> done
            status          VARCHAR(20) NOT NULL DEFAULT 'inbox',
            priority        VARCHAR(10) NOT NULL DEFAULT 'medium',
            review_mode     VARCHAR(10) NOT NULL DEFAULT 'auto',

            -- Assignment
            assigned_agent_id INTEGER REFERENCES agents(id) ON DELETE SET NULL,

            -- Lineage
            created_by_type VARCHAR(20) NOT NULL DEFAULT 'user',
            created_by_id   VARCHAR(255),
            parent_task_id  INTEGER REFERENCES board_tasks(id) ON DELETE SET NULL,

            -- Tags
            tags            JSONB DEFAULT '[]',

            -- Execution
            result          TEXT,
            error_message   TEXT,
            started_at      TIMESTAMPTZ,
            completed_at    TIMESTAMPTZ,

            -- Planning
            planning_data   JSONB,

            -- Timestamps
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS ix_board_tasks_workspace ON board_tasks(workspace_id);
        CREATE INDEX IF NOT EXISTS ix_board_tasks_status ON board_tasks(workspace_id, status);
        CREATE INDEX IF NOT EXISTS ix_board_tasks_agent ON board_tasks(assigned_agent_id, status);
        CREATE INDEX IF NOT EXISTS ix_board_tasks_parent ON board_tasks(parent_task_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS board_tasks;")
