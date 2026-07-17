"""PRD-205 S3: Auto Speaks -- background->chat delivery columns

Additive only, four surfaces:

- messages.source (JSONB, nullable): persisted author/source signal for
  background-authored assistant messages ({"origin": "watcher"|
  "scheduled_task", "label": "Auto <middle-dot> background", "link_type",
  "link_id"}). NULL for every in-turn message, old and new -- the UI badge
  reads this, so it survives reload (the frontend-only message.metadata
  never persisted).
- chats.kind (String(20), NOT NULL, default 'user') + CHECK ('user'|'auto')
  + partial UNIQUE index on (workspace_id, user_id) WHERE kind='auto':
  at most ONE per-user "Auto" thread per workspace -- the fallback target
  where Auto speaks unprompted. Race-safe find-or-create leans on the index.
- watches.origin_chat_id (UUID, nullable): the conversation a watch was
  created from (captured by the S4 executor injection). SOFT reference by
  design -- NO foreign key to chats: deleting a chat must never break or
  cascade into the watch registry; a dangling origin simply falls back to
  the creator's Auto thread at delivery time.
- agent_scheduled_tasks.origin_chat_id (UUID, nullable) +
  agent_scheduled_tasks.created_by (String(255), nullable): same capture for
  the PRD-77 scheduled-task fix (S6). The table's only creator column was
  created_by_agent_id (an agents.id -- an AGENT, not a user), so without
  these there is NO user to deliver output to. created_by is the driving
  Clerk user id string (same convention as watches.created_by); both are
  soft references for the same deletion-safety reason as above.

Chaining (PRD-205 Section 8 Q7): PR #551's join (prd204_w3_join_heads)
MERGED mid-build, restoring a single lineage whose head then advanced
through prd197_substrate_metrics to prd199_drop_fake_stats -- so this
chains single-parent on that CURRENT head, keeping the graph at exactly
one head. (The first cut of this revision chained on
prd204_watch_registry while #551 was still open; landing that after the
join would have re-forked the graph -- the #545 x #548 lesson.)

Revision ID: prd205_auto_speaks
Revises: prd199_drop_fake_stats
Create Date: 2026-07-17
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

revision = "prd205_auto_speaks"
down_revision = "prd199_drop_fake_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- messages.source ---------------------------------------------------
    op.add_column("messages", sa.Column("source", JSONB, nullable=True))

    # -- chats.kind + one-auto-thread-per-(workspace,user) -----------------
    op.add_column(
        "chats",
        sa.Column("kind", sa.String(20), nullable=False, server_default="user"),
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

    # -- watches.origin_chat_id (soft reference -- no FK, see docstring) ---
    op.add_column(
        "watches", sa.Column("origin_chat_id", PGUUID(as_uuid=True), nullable=True)
    )

    # -- agent_scheduled_tasks origin capture (S6) --------------------------
    op.add_column(
        "agent_scheduled_tasks",
        sa.Column("origin_chat_id", PGUUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "agent_scheduled_tasks",
        sa.Column("created_by", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_scheduled_tasks", "created_by")
    op.drop_column("agent_scheduled_tasks", "origin_chat_id")
    op.drop_column("watches", "origin_chat_id")
    op.drop_index("uq_chats_auto_thread", table_name="chats")
    op.drop_constraint("check_chat_kind", "chats", type_="check")
    op.drop_column("chats", "kind")
    op.drop_column("messages", "source")
