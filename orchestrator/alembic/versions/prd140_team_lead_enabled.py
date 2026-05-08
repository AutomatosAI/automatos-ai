"""PRD-140 Phase 1 — agents.team_lead_enabled

Adds the explicit activation flag for team-lead feedback loops. ``is_team_lead``
remains a *derived* property (an agent has direct reports), exposed via API
as a computed field. The loop only runs for agents that are derived team
leads AND have ``team_lead_enabled=true`` — operators have to opt in
deliberately.

Default is FALSE so existing agents are unaffected. Opt-in is per-agent
through the existing agent settings UI / platform_update_agent.

Idempotent.
"""

from alembic import op


revision = "prd140_team_lead_enabled"
down_revision = "wave3_escalation_level"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS team_lead_enabled BOOLEAN
        NOT NULL DEFAULT FALSE;
        """
    )
    # Cheap partial index — most workspaces have ≤ 4 team leads.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agents_team_lead_enabled "
        "ON agents (workspace_id) WHERE team_lead_enabled = TRUE;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agents_team_lead_enabled;")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS team_lead_enabled;")
