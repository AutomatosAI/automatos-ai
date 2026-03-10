"""PRD-72: Add last_accessed and rag_query_count to documents table

Revision ID: prd72_doc_access
Revises: None
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa

revision = "prd72_doc_access"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS rag_query_count INTEGER DEFAULT 0;
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE documents DROP COLUMN IF EXISTS last_accessed;
        ALTER TABLE documents DROP COLUMN IF EXISTS rag_query_count;
    """)
