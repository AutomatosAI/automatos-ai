"""llm_usage.agent_name — a usage row keeps the name of the agent that spent it.

F049 (night 1): deleting an agent left its spend behind with nothing to name it.
``llm_usage.agent_id`` carries no foreign key, so the rows survive with an id
that joins to no agent: Analytics rendered them as "Agent #273", and the KPI
card's top spenders dropped them altogether (an inner join). Agent 273 was
CELLAR. The delete path now stamps the agent's name on its usage rows before
the agent row goes; readers take the live name first, then this one.

No backfill: nothing in the database kept the names of agents already deleted
(checked 2026-09-22: 111 rows across 4 agents, no name in agent_reports or
deliverables). Idempotent (IF NOT EXISTS) so a create_all-first boot is safe.

Revision ID: llm_usage_agent_name
Revises: kb_multimodal_tables
Create Date: 2026-09-22
"""
from alembic import op

revision = "llm_usage_agent_name"
down_revision = "kb_multimodal_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS agent_name VARCHAR(255);")


def downgrade() -> None:
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS agent_name;")
