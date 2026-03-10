"""PRD-72: Add memory_access_log table for hit-rate analytics

Revision ID: prd72_memory_access_log
Revises: None
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa

revision = "prd72_memory_access_log"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS memory_access_log (
            id          BIGSERIAL PRIMARY KEY,
            workspace_id UUID NOT NULL,
            had_results  BOOLEAN NOT NULL DEFAULT FALSE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_memory_access_log_ws_created
            ON memory_access_log (workspace_id, created_at DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_access_log;")
