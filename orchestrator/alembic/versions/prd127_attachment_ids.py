"""PRD-127: Attachment IDs column for board_tasks

Add attachment_ids JSONB column to board_tasks.
Stores inline metadata for ephemeral attachments (no separate attachments table).
Mission tables (mission_runs, mission_tasks) will be handled when PRD-82A creates them.
"""

from alembic import op
import sqlalchemy as sa

revision = "prd127_attachment_ids"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE board_tasks
        ADD COLUMN IF NOT EXISTS attachment_ids JSONB NOT NULL DEFAULT '[]'::jsonb;
    """)
    op.execute("""
        COMMENT ON COLUMN board_tasks.attachment_ids IS
        'Ephemeral attachment refs: [{attachment_id, filename, mime, media_type}]. Blobs in S3 with 7-day TTL.';
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE board_tasks DROP COLUMN IF EXISTS attachment_ids;")
