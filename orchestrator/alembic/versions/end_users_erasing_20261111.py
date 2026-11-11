"""When an end user's erase started (11 Nov 2026).

PRD-274, "Automatos for products", Wave 3 (§5.5). One new nullable column:

* ``end_users.erasing_at``: set when ``DELETE /api/memory/end-users/{id}`` starts, before anything is
  deleted. From then on the end user's consent no longer counts: no new chat turn or store for them, and a
  turn still running writes nothing more to their memory. An erase that failed part-way leaves the record
  and its mark, and the next erase finishes it. NULL on every row today, so no row changes.

``core/models/end_users.py`` declares the same. Create_all-first safe: ``ADD COLUMN IF NOT EXISTS``. The
downgrade drops the column.

Revision ID: end_users_erasing_20261111
Revises: end_user_chats_20261111
Create Date: 2026-11-11
"""
from alembic import op

revision = "end_users_erasing_20261111"
down_revision = "end_user_chats_20261111"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE end_users ADD COLUMN IF NOT EXISTS erasing_at TIMESTAMP WITH TIME ZONE;")


def downgrade() -> None:
    op.execute("ALTER TABLE end_users DROP COLUMN IF EXISTS erasing_at;")
