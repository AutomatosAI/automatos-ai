"""PRD-206 S2: chats.summary — the thread checkpoint.

Additive only: one nullable JSONB column on chats holding
{topic, decisions[], open_questions[], last_summary, next_step,
updated_at, checkpointed_at, trigger}. Written by the checkpoint distill
(idle sweep + platform_checkpoint_thread); read by the S3 resume payload.
NULL until a thread has been checkpointed.

Chains single-parent on prd221_digest_feedback (the current single head).

Revision ID: prd206_chat_summary
Revises: prd221_digest_feedback
Create Date: 2026-07-17
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd206_chat_summary"
down_revision = "prd221_digest_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chats", sa.Column("summary", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("chats", "summary")
