"""PRD-82B US-009: Add orchestration_archive table for archival of terminal runs

Creates:
  - orchestration_archive table with full JSONB snapshot
  - Index on workspace_id for scoped queries
  - Unique constraint on original_run_id

Revision ID: prd82b_us009_archive
Revises: prd82b_us005_replan
Create Date: 2026-03-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "prd82b_us009_archive"
down_revision = "prd82b_us005_replan"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_archive (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            original_run_id UUID NOT NULL UNIQUE,
            goal            TEXT NOT NULL,
            state           VARCHAR(30) NOT NULL,
            workspace_id    UUID NOT NULL,
            created_by      VARCHAR(255) NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL,
            completed_at    TIMESTAMPTZ,
            archive_data    JSONB NOT NULL,
            archived_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_orchestration_archive_workspace
        ON orchestration_archive (workspace_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS orchestration_archive")
