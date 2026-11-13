"""The end user a server key's conversation is for (11 Nov 2026).

PRD-274, "Automatos for products", Wave 3 (§5.5). One new nullable column and its index:

* ``chats.end_user_id``: the end user (``end_users.end_user_id``) a server key's chat turn named. Only
  that end user's turns continue the conversation, and erasing them deletes it. NULL on every other
  chat, so no row changes.
* ``ix_chats_workspace_end_user``: a partial index on ``(workspace_id, end_user_id)`` where it is set,
  for an end user's erasure.

``core/models/core.py`` declares the same. Create_all-first safe: ``ADD COLUMN IF NOT EXISTS`` and
``CREATE INDEX IF NOT EXISTS``. The downgrade drops the index and the column.

Revision ID: end_user_chats_20261111
Revises: end_users_20261110
Create Date: 2026-11-11
"""
from alembic import op

revision = "end_user_chats_20261111"
down_revision = "end_users_20261110"
branch_labels = None
depends_on = None

_INDEX = "ix_chats_workspace_end_user"


def upgrade() -> None:
    op.execute("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128);")
    op.execute(f"CREATE INDEX IF NOT EXISTS {_INDEX} ON chats (workspace_id, end_user_id) "
               "WHERE end_user_id IS NOT NULL;")


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX};")
    op.execute("ALTER TABLE chats DROP COLUMN IF EXISTS end_user_id;")
