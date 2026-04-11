"""Drop hardcoded gpt-4 default on agents.model_config

The `agents.model_config` column was originally created via
`core/database/add_missing_agent_columns.sql` with a hardcoded default:

    DEFAULT '{"provider": "openai", "model_id": "gpt-4", "temperature": 0.7}'::jsonb

That default silently shackled any agent created without an explicit
model_config to the original 8K-context `gpt-4` model. Symptom: Mission
Zero's seeded researcher/writer/analyst/strategist blew up at execution
time with "Your model (gpt-4) does not have enough context for this
conversation" because the mission prompt exceeded 8192 tokens.

This migration drops the default so agents without an explicit
model_config resolve through `AgentFactory._get_default_llm_config_from_settings`,
which reads `system_settings.orchestrator_llm` → `config.LLM_MODEL` env
var → fails loud if still unset. That chain has one knob to tune
(`LLM_MODEL=gpt-4o` etc), not a rotting DEFAULT baked into DDL.

Companion fix in `orchestrator/api/wizard.py:_ensure_mission_zero_team`
sets `model_config` explicitly from `config.LLM_MODEL` so fresh Mission
Zero runs never rely on the default either way.

Revision ID: drop_agents_model_config_default
Revises: prd127_attachment_ids
Create Date: 2026-04-11
"""
from alembic import op

revision = "drop_agents_model_config_default"
down_revision = "prd127_attachment_ids"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agents ALTER COLUMN model_config DROP DEFAULT")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE agents ALTER COLUMN model_config SET DEFAULT "
        "'{\"provider\": \"openai\", \"model_id\": \"gpt-4\", \"temperature\": 0.7}'::jsonb"
    )
