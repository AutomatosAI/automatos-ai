"""PRD-76: Agent Reports & Workspace

Create agent_reports table for structured report metadata.
Content lives in workspace filesystem; this table is for discovery, filtering, and grading.

Revision ID: prd76_agent_reports
Revises: None (standalone — safe to run anytime)
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa

revision = "prd76_agent_reports"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_reports (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id        UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            agent_id            INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            heartbeat_result_id INTEGER REFERENCES heartbeat_results(id) ON DELETE SET NULL,

            -- Report metadata
            report_type         VARCHAR(30) NOT NULL DEFAULT 'standup',
            title               VARCHAR(255) NOT NULL,
            summary             VARCHAR(500),
            status              VARCHAR(20) NOT NULL DEFAULT 'ok',

            -- File reference
            file_path           VARCHAR(1024) NOT NULL,
            file_type           VARCHAR(20) NOT NULL DEFAULT 'markdown',
            file_size_bytes     INTEGER,

            -- Structured metrics
            metrics             JSONB DEFAULT '{}',

            -- Linked attachments
            attachments         JSONB DEFAULT '[]',

            -- User grading
            grade               SMALLINT CHECK (grade >= 1 AND grade <= 5),
            grade_notes         TEXT,
            graded_by           INTEGER REFERENCES users(id) ON DELETE SET NULL,
            graded_at           TIMESTAMPTZ,

            -- Timestamps
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indices for common queries
        CREATE INDEX IF NOT EXISTS ix_agent_reports_workspace
            ON agent_reports(workspace_id);
        CREATE INDEX IF NOT EXISTS ix_agent_reports_agent
            ON agent_reports(agent_id);
        CREATE INDEX IF NOT EXISTS ix_agent_reports_type
            ON agent_reports(workspace_id, report_type);
        CREATE INDEX IF NOT EXISTS ix_agent_reports_status
            ON agent_reports(workspace_id, status);
        CREATE INDEX IF NOT EXISTS ix_agent_reports_created
            ON agent_reports(created_at DESC);
        CREATE INDEX IF NOT EXISTS ix_agent_reports_heartbeat
            ON agent_reports(heartbeat_result_id)
            WHERE heartbeat_result_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS ix_agent_reports_ungraded
            ON agent_reports(workspace_id, created_at DESC)
            WHERE grade IS NULL;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_reports;")
