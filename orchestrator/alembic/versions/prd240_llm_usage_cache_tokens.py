"""Analytics — prompt-cache tokens on every usage row.

The Analytics page promises "where the cache is paying for itself", and nothing
recorded a cache token: the Anthropic API reports cache reads/writes beside
``input_tokens``, OpenAI-compatible APIs (OpenRouter, OpenAI) report
``prompt_tokens_details.cached_tokens``, and a Claude Code session on the
user's subscription is almost entirely cache reads (117k cached vs 6 fresh
input tokens on a typical ticket). Two nullable counters on ``llm_usage``:

- ``cache_read_tokens``   prompt tokens served from the provider's cache
- ``cache_write_tokens``  prompt tokens written into the cache this call

``input_tokens`` stays the FULL prompt size (fresh + cached + written) on
every row, whatever the provider's own accounting; the two counters are the
breakdown. Idempotent (IF NOT EXISTS) so a create_all-first boot is safe.
"""

from alembic import op


revision = "prd240_llm_usage_cache_tokens"
down_revision = "prd240_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS cache_read_tokens INTEGER NOT NULL DEFAULT 0;")
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS cache_write_tokens INTEGER NOT NULL DEFAULT 0;")


def downgrade() -> None:
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS cache_write_tokens;")
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS cache_read_tokens;")
