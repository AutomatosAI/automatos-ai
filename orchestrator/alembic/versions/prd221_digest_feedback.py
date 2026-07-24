"""PRD-221 S10: digest_feedback — thumbs up/down on Auto's Read.

Additive only: one table keyed by (workspace_id, state_hash) so feedback
attaches to the workspace state a digest described (the digest itself is
cache-only, never persisted). rating is constrained to {-1, 1}.

Chains single-parent on prd205_auto_speaks (the current single head — the
never-author-a-second-join-of-the-same-parents rule). PRD-221's only schema
change; background→chat delivery already shipped in prd205_auto_speaks, so
no agent_scheduled_tasks column is added here.

Revision ID: prd221_digest_feedback
Revises: prd205_auto_speaks
Create Date: 2026-07-17
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "prd221_digest_feedback"
down_revision = "prd205_auto_speaks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "digest_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_digest_feedback_rating"),
    )
    op.create_index(
        "ix_digest_feedback_workspace", "digest_feedback", ["workspace_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_digest_feedback_workspace", table_name="digest_feedback")
    op.drop_table("digest_feedback")
