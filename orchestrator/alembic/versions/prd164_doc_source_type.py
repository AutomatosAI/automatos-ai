"""PRD-164 S3 (Q58): documents.source_type — the agent-output flywheel scope.

Adds a nullable provenance column to ``documents`` so flywheel-ingested agent
outputs (mission syntheses, generated documents, submitted reports) are
tagged ``source_type='agent_output'`` and become a filterable, team-like
scope on every documents surface (list API, knowledge UI counts, retrieval
diagnostics). NULL means a regular upload — existing rows need no backfill.

Index covers the two read shapes: the scope filter on the documents list and
the agent-outputs count on /team-counts.

Revision ID: prd164_doc_source_type
Revises: 20260612_template_blocks (current head at branch time)
Create Date: 2026-06-13
"""

from alembic import op

revision = "prd164_doc_source_type"
down_revision = "20260612_template_blocks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS source_type VARCHAR(50);

        CREATE INDEX IF NOT EXISTS ix_documents_workspace_source_type
            ON documents (workspace_id, source_type)
            WHERE source_type IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ix_documents_workspace_source_type;
        ALTER TABLE documents DROP COLUMN IF EXISTS source_type;
        """
    )
