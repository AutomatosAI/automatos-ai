"""user_api_keys.provider_workspace_id — the workspace an Anthropic key bills (9 Oct 2026).

An Anthropic key created at organization level is not scoped to a workspace: every
request must name one in the ``anthropic-workspace-id`` header, or the API answers
400. The Add API Key dialog had nowhere to put it, so such a key failed validation
and could never be used. The workspace ID (``wrkspc_…``) is now saved with the key
(``core.llm.anthropic_workspace``), beside the key's fingerprint (PBKDF2)
(``key_fingerprint``, indexed): the client finds a key's workspace by it, so no
stored key is decrypted to find one.

Nullable with no default: every existing key, and every key scoped to its workspace,
keeps NULL and is sent exactly as before. ``IF NOT EXISTS`` keeps the upgrade a
no-op on a database that ``create_all`` built from the updated model.

Revision ID: user_api_keys_provider_workspace_id
Revises: user_api_keys_base_url
Create Date: 2026-10-09
"""

from alembic import op


revision = "user_api_keys_provider_workspace_id"
down_revision = "user_api_keys_base_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE user_api_keys ADD COLUMN IF NOT EXISTS provider_workspace_id VARCHAR(64);")
    op.execute("ALTER TABLE user_api_keys ADD COLUMN IF NOT EXISTS key_fingerprint VARCHAR(64);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_user_api_keys_key_fingerprint ON user_api_keys (key_fingerprint);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_user_api_keys_key_fingerprint;")
    op.execute("ALTER TABLE user_api_keys DROP COLUMN IF EXISTS key_fingerprint;")
    op.execute("ALTER TABLE user_api_keys DROP COLUMN IF EXISTS provider_workspace_id;")
