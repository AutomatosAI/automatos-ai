"""PRD-236 Wave 1 — the catalogue knows who serves what.

``llm_models`` gains ``serving_provider`` (the registry slug of the API that
serves the row); the UNIQUE moves from ``model_id`` to
``(serving_provider, model_id)`` so one vendor id can exist once per route —
"moonshotai/kimi-k3" on OpenRouter AND on NVIDIA, each with its own price;
``tier`` is renamed ``sourcing`` (PRD-223 §8 Q1, owner: yes) and set from the
registry kind of the serving provider. ``external_id`` (already present) holds
the provider-native id.

Data rule: a row with ``tier='direct'`` and a registered direct provider keeps
its ``provider`` as the serving provider. Everything else — aggregator rows
(whose ``provider`` is the VENDOR), legacy 'aiml'/'together' seeds, byok_only —
is served by OpenRouter, which is exactly where the factory's string-shape
routing sent them at runtime. ``workspace_models`` is untouched: it already
keys on the integer row, so an install IS the route tag.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "prd236_w1_serving_provider"
down_revision = "prd234_s1a_cli_hosts_runtime_ref"
branch_labels = None
depends_on = None

# Snapshot of the registry's direct chat providers at migration time
# (core/llm/providers.py) — a migration must not import application code.
_DIRECT_PROVIDERS = (
    "openai", "anthropic", "google", "azure", "azure_openai",
    "bedrock", "aws_bedrock", "grok", "huggingface", "deepseek",
)


def upgrade():
    op.add_column(
        "llm_models",
        sa.Column("serving_provider", sa.String(length=50), nullable=False, server_default="openrouter"),
    )
    direct_list = ", ".join(f"'{p}'" for p in _DIRECT_PROVIDERS)
    op.execute(
        f"UPDATE llm_models SET serving_provider = provider "
        f"WHERE tier = 'direct' AND provider IN ({direct_list})"
    )
    # alias normalisation — the registry slug is 'bedrock' / 'azure'
    op.execute("UPDATE llm_models SET serving_provider = 'bedrock' WHERE serving_provider = 'aws_bedrock'")
    op.execute("UPDATE llm_models SET serving_provider = 'azure' WHERE serving_provider = 'azure_openai'")
    op.execute("UPDATE llm_models SET external_id = model_id WHERE external_id IS NULL")

    # One row per route: UNIQUE (serving_provider, model_id) replaces UNIQUE (model_id).
    # Tolerant of both writers: the alembic-created constraint and a create_all()
    # unique index carry different names.
    op.execute("ALTER TABLE llm_models DROP CONSTRAINT IF EXISTS llm_models_model_id_key")
    op.execute("DROP INDEX IF EXISTS ix_llm_models_model_id")
    op.create_unique_constraint(
        "uq_llm_models_provider_model", "llm_models", ["serving_provider", "model_id"]
    )
    op.create_index("ix_llm_models_model_id", "llm_models", ["model_id"], unique=False)
    op.create_index("idx_llm_models_serving_provider", "llm_models", ["serving_provider"], unique=False)

    # PRD-223 Q1: the sourcing vocabulary gets its real name; values follow the
    # registry kind of the serving provider.
    op.alter_column("llm_models", "tier", new_column_name="sourcing")
    op.execute(
        "UPDATE llm_models SET sourcing = CASE "
        "WHEN serving_provider = 'openrouter' THEN 'aggregator' "
        "WHEN serving_provider = 'nvidia' THEN 'hosted_open' "
        "ELSE 'direct' END"
    )


def downgrade():
    op.alter_column("llm_models", "sourcing", new_column_name="tier")
    op.drop_index("idx_llm_models_serving_provider", table_name="llm_models")
    op.drop_index("ix_llm_models_model_id", table_name="llm_models")
    op.drop_constraint("uq_llm_models_provider_model", "llm_models", type_="unique")
    # Routes served by other providers cannot coexist under the old key.
    op.execute("DELETE FROM llm_models WHERE serving_provider <> 'openrouter' AND tier <> 'direct'")
    op.create_unique_constraint("llm_models_model_id_key", "llm_models", ["model_id"])
    op.drop_column("llm_models", "serving_provider")
