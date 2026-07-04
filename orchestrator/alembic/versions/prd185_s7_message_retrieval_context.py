"""PRD-185 S7: messages.retrieval_context — per-turn RAG provenance.

Adds a nullable JSONB column recording the ``{document_ids, chunk_ids, query}``
a chat turn retrieved. Read at vote time (``PATCH /api/chat/vote``) to write a
complete ``rag_feedback`` row, which the PRD-179 live ranker consumes via
``UNNEST(document_ids)``. NULL means the turn retrieved nothing — existing rows
need no backfill. Kept off ``parts`` so retrieval provenance never reaches the
AI-SDK render contract.

Revision ID: prd185_s7_msg_retrieval_ctx
Revises: e773c09189a9 (single head at branch time)
Create Date: 2026-07-04
"""

from alembic import op

revision = "prd185_s7_msg_retrieval_ctx"
down_revision = "e773c09189a9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS retrieval_context JSONB;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE messages DROP COLUMN IF EXISTS retrieval_context;
        """
    )
