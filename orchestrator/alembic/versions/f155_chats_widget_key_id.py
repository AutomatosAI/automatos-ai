"""chats.widget_key_id — the widget key that started a conversation.

F155: the widget records the key that starts a conversation, and a key reads
and resumes only the conversations it started. NULL is every other chat (the
dashboard's, Auto's thread, and widget chats from before this revision), which
no key reaches. No foreign key: a deleted key's id stays on its chats, and no
other key matches it. Idempotent (IF NOT EXISTS) so a create_all-first boot is
safe.

Revision ID: f155_chats_widget_key_id
Revises: f049_prd251_merge_heads
Create Date: 2026-09-25
"""
from alembic import op

revision = "f155_chats_widget_key_id"
down_revision = "f049_prd251_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chats ADD COLUMN IF NOT EXISTS widget_key_id UUID;")


def downgrade() -> None:
    op.execute("ALTER TABLE chats DROP COLUMN IF EXISTS widget_key_id;")
