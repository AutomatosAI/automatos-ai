"""A product's end users, with the consent recorded for them (10 Nov 2026).

PRD-274, "Automatos for products", Wave 3 (§5.5, §8-17). One new table:

* ``end_users``: one row per ``(workspace_id, end_user_id)`` (a unique index), the id a product's
  server key asserts; when its consent was recorded, by which key (``ON DELETE SET NULL``), and how
  many days its memories are kept (NULL: until erased). ON DELETE CASCADE with the workspace.

``core/models/end_users.py`` declares the same shape. Create_all-first safe: ``CREATE TABLE IF NOT
EXISTS`` and ``CREATE UNIQUE INDEX IF NOT EXISTS``; it changes no row. The downgrade drops the table.

Revision ID: end_users_20261110
Revises: server_key_chat_20261110
Create Date: 2026-11-10
"""
from alembic import op

revision = "end_users_20261110"
down_revision = "server_key_chat_20261110"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS end_users (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    end_user_id VARCHAR(128) NOT NULL,
    api_key_id UUID REFERENCES sdk_api_keys(id) ON DELETE SET NULL,
    consent_recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    retention_days INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""

_INDEX = ("CREATE UNIQUE INDEX IF NOT EXISTS uq_end_users_workspace_end_user "
          "ON end_users (workspace_id, end_user_id);")


def upgrade() -> None:
    op.execute(_TABLE)
    op.execute(_INDEX)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS end_users;")
