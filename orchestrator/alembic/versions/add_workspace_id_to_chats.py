"""Add workspace_id to chats table for multi-tenancy isolation.

CRITICAL FIX: Chats were not scoped to workspaces, causing cross-workspace
data leakage — users in workspace B could see workspace A's conversations.

This migration:
1. Adds workspace_id FK column to chats (nullable for backfill)
2. Backfills existing chats from their messages' workspace_id
3. Creates index for efficient workspace+user queries
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = "fix_chat_workspace_isolation"
down_revision = None  # standalone migration — will be applied manually
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add column (nullable initially for backfill)
    op.add_column(
        "chats",
        sa.Column("workspace_id", UUID(as_uuid=True), nullable=True),
    )

    # 2. Add FK constraint
    op.create_foreign_key(
        "fk_chats_workspace_id",
        "chats",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 3. Backfill from messages table (each chat's first message has workspace_id)
    op.execute("""
        UPDATE chats
        SET workspace_id = sub.ws
        FROM (
            SELECT DISTINCT ON (chat_id) chat_id, workspace_id AS ws
            FROM messages
            WHERE workspace_id IS NOT NULL
            ORDER BY chat_id, created_at ASC
        ) sub
        WHERE chats.id = sub.chat_id
          AND chats.workspace_id IS NULL
    """)

    # 4. Create composite index for workspace+user queries
    op.create_index(
        "ix_chats_workspace_user",
        "chats",
        ["workspace_id", "user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_chats_workspace_user", table_name="chats")
    op.drop_constraint("fk_chats_workspace_id", "chats", type_="foreignkey")
    op.drop_column("chats", "workspace_id")
