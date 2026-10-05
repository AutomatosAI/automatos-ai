"""user_api_keys.base_url — a workspace key's own endpoint (#873).

Azure OpenAI (Microsoft Foundry) has no fixed address: each resource has its own
(``https://<resource>.openai.azure.com``). The Add API Key dialog took only the key,
so a workspace's Azure key could only reach the API service's
``AZURE_OPENAI_ENDPOINT``. That works for a self-hosted operator and never for a
hosted workspace. The endpoint is now saved with the key and travels with it to the
client (``core.llm.byok_endpoint``).

Nullable with no default: every existing key, and every provider with a fixed
address, keeps NULL and resolves exactly as before. ``IF NOT EXISTS`` keeps the
upgrade a no-op on a database that ``create_all`` built from the updated model.

Revision ID: user_api_keys_base_url
Revises: workflows_tags_jsonb
Create Date: 2026-10-05
"""

from alembic import op


revision = "user_api_keys_base_url"
down_revision = "workflows_tags_jsonb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE user_api_keys ADD COLUMN IF NOT EXISTS base_url TEXT;")


def downgrade() -> None:
    op.execute("ALTER TABLE user_api_keys DROP COLUMN IF EXISTS base_url;")
