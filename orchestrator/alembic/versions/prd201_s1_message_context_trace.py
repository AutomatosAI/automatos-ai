"""PRD-201 S1: messages.context_trace — per-turn context-assembly trace.

Adds a nullable JSONB column recording what ``ContextService.build_context``
assembled for the turn: mode, per-section ``{name, priority, token_estimate,
rendered_nonempty, trimmed}``, the driving model, the resolved budget ceiling,
injected memory ids and prep_ms. The assembler already computed these fields on
every build and threw them away; persisting them makes "what did Auto know when
it said that?" a query. Written regardless of ``TRACING_ENABLED`` (the Langfuse
span is the live mirror; this column is the durable, offline-answerable record).

NULL means the turn was built before this shipped (no backfill needed). Kept off
``parts`` so it never reaches the AI-SDK render contract — same discipline as the
PRD-185 S7 ``retrieval_context`` column this mirrors.

Revision ID: prd201_s1_msg_context_trace
Revises: prd196_audit_logs_ws_created_idx (latest head at branch time)
Create Date: 2026-07-14
"""

from alembic import op

revision = "prd201_s1_msg_context_trace"
down_revision = "prd196_audit_logs_ws_created_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS context_trace JSONB;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE messages DROP COLUMN IF EXISTS context_trace;
        """
    )
