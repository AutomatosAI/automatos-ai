"""PRD-205 S3: Auto Speaks — background→chat delivery columns.

Additive only:
- ``messages.source`` JSONB — background-author provenance
  ({origin, label, link_type, link_id}); NULL on every in-turn message.
- ``chats.kind`` ('user'|'auto') — the per-user per-workspace Auto thread,
  one per (workspace, user) via a partial unique index.
- ``watches.origin_chat_id`` UUID — the conversation a watched launch came
  from, so verdicts post back into it.
- ``agent_scheduled_tasks.origin_chat_id`` UUID — same capture for scheduled
  tasks, whose rows are agent-created (no user to fall back to).

Chains single-parent on prd199_drop_fake_stats (the current single head —
the §8-Q7 rule: never author a second join of the same parents).

Revision ID: prd205_auto_speaks
Revises: prd199_drop_fake_stats
Create Date: 2026-07-17
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "prd205_auto_speaks"
down_revision = "prd199_drop_fake_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("source", JSONB(), nullable=True))

    op.add_column(
        "chats",
        sa.Column(
            "kind", sa.String(length=20), nullable=False, server_default="user"
        ),
    )
    op.create_check_constraint(
        "check_chat_kind", "chats", "kind IN ('user', 'auto')"
    )
    op.create_index(
        "uq_chats_auto_thread",
        "chats",
        ["workspace_id", "user_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'auto'"),
    )

    op.add_column(
        "watches", sa.Column("origin_chat_id", UUID(as_uuid=True), nullable=True)
    )

    # S6: scheduled tasks are agent-created (created_by_agent_id — no human,
    # no chat on the row), so without a captured origin the delivered output
    # has no target. Same capture pattern as watches.
    op.add_column(
        "agent_scheduled_tasks",
        sa.Column("origin_chat_id", UUID(as_uuid=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_scheduled_tasks", "origin_chat_id")
    op.drop_column("watches", "origin_chat_id")
    op.drop_index("uq_chats_auto_thread", table_name="chats")
    op.drop_constraint("check_chat_kind", "chats", type_="check")
    op.drop_column("chats", "kind")
    op.drop_column("messages", "source")
