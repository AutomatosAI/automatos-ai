"""A product's server key's chat turn under an Idempotency-Key (16 Nov 2026).

PRD-274, "Automatos for products", Wave 1 (§5.2). ``POST /api/chat`` from a product's server key
honours ``Idempotency-Key``, as ``POST /api/agents/{id}/execute`` does. One new table:

* ``chat_turn_keys``: one row per (key, Idempotency-Key): the request's hash, the turn's state (running,
  done, failed), the chat it answered in and its saved answer's message id; no words of the turn. ON
  DELETE CASCADE with the workspace and with the key. ``chat_id`` and ``answer_message_id`` have no
  foreign key, so deleting a chat is never held up by a row.
* ``uq_chat_turn_keys_key`` on ``(api_key_id, idempotency_key)``: of two racing requests, one runs.

``core/models/chat_turn_keys.py`` declares the same shape. Create_all-first safe: ``CREATE TABLE IF NOT
EXISTS`` and ``CREATE UNIQUE INDEX IF NOT EXISTS``, so a database ``create_all`` built is left as it is,
and running it twice changes nothing. It alters no table and changes no row. The downgrade drops the
table.

Revision ID: chat_turn_keys_20261116
Revises: credit_ledger_sub_tenant_20261113
Create Date: 2026-11-16
"""
from alembic import op

revision = "chat_turn_keys_20261116"
down_revision = "credit_ledger_sub_tenant_20261113"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS chat_turn_keys (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    api_key_id UUID NOT NULL REFERENCES sdk_api_keys(id) ON DELETE CASCADE,
    idempotency_key VARCHAR(255) NOT NULL,
    request_hash VARCHAR(64) NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'running',
    chat_id UUID,
    answer_message_id UUID,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""

# As core/models/chat_turn_keys.py declares it (UNIQUE_INDEX).
_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_turn_keys_key ON chat_turn_keys (api_key_id, idempotency_key);"
)


def upgrade() -> None:
    op.execute(_TABLE)
    op.execute(_INDEX)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chat_turn_keys;")
