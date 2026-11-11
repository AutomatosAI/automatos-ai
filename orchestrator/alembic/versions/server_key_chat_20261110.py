"""A server key's model allow-list and the service user its chat turns run as (10 Nov 2026).

PRD-274, "Automatos for products", Wave 1 (§5.2, §8-5). Two new nullable columns on
``sdk_api_keys``:

* ``model_allow_list`` (JSONB): the models a request through the key may ask a lean agent for,
  ``[{"provider", "model_id"}]``; a model must be on the agent's own list too.
* ``service_user_id`` (INTEGER, REFERENCES users ON DELETE SET NULL): the non-login ``users`` row
  the key's headless chat turns run as, made on first use (``api/chat_server_key.py``).

``core/models/sdk_api_keys.py`` declares the same columns. Create_all-first safe: ``ADD COLUMN IF
NOT EXISTS``; no row changes, both start NULL. The downgrade drops them.

Revision ID: server_key_chat_20261110
Revises: execute_idempotency_20261109
Create Date: 2026-11-10
"""
from alembic import op

revision = "server_key_chat_20261110"
down_revision = "execute_idempotency_20261109"
branch_labels = None
depends_on = None

_COLUMNS = (
    "ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS model_allow_list JSONB;",
    "ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS service_user_id INTEGER "
    "REFERENCES users(id) ON DELETE SET NULL;",
)


def upgrade() -> None:
    for statement in _COLUMNS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("ALTER TABLE sdk_api_keys DROP COLUMN IF EXISTS service_user_id;")
    op.execute("ALTER TABLE sdk_api_keys DROP COLUMN IF EXISTS model_allow_list;")
