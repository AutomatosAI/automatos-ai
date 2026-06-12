
"""prd158 teams table (collapse heads)

Revision ID: prd158_teams
Revises: 20260226_cto_agent, prd141_backfill_workspace_vertical, add_content_hash_to_skills, add_job_title_to_agents, add_ws_admin_lifecycle, fix_chat_workspace_isolation, agent_public_id_and_slug_fix, agents_public_id_default, blog_content_to_workspace, board_blocked_sla, drop_agents_model_config_default, prd123_cost_tracking, prd128_notifications, prd129_deliverables, prd130_workspace_graphs, prd133b_outputs_view, prd135_drop_bucket_1, prd135_drop_bucket_2, prd135_drop_bucket_3, prd135_drop_bucket_4, prd135_drop_bucket_5, prd135_drop_bucket_6, prd136_collapse_llm_tiers, prd139_tool_routing_graph, prd139_tool_routing_telemetry, prd140_permission_bypass_log, prd142_wave0_error_events, prd142_wave4_harness_store, prd142_wave5_drop_dead_tables, prd71_unified_skills, prd72_board_tasks, prd72_doc_access, prd72_fix_composio_trigger, prd72_memory_access_log, prd72_recipe_board_tasks, prd72_task_reconciliation, prd73_infra_alerts, prd74_voice_profiles, prd76_agent_reports, prd76_nullable_agent, prd77_agent_scheduled_tasks, prd79_memory_short_term, seed_auto_agents_existing_workspaces, unify_marketplace_categories
Create Date: 2026-06-12 10:43:28.905226

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'prd158_teams'
down_revision = ('20260226_cto_agent', 'prd141_backfill_workspace_vertical', 'add_content_hash_to_skills', 'add_job_title_to_agents', 'add_ws_admin_lifecycle', 'fix_chat_workspace_isolation', 'agent_public_id_and_slug_fix', 'agents_public_id_default', 'blog_content_to_workspace', 'board_blocked_sla', 'drop_agents_model_config_default', 'prd123_cost_tracking', 'prd128_notifications', 'prd129_deliverables', 'prd130_workspace_graphs', 'prd133b_outputs_view', 'prd135_drop_bucket_1', 'prd135_drop_bucket_2', 'prd135_drop_bucket_3', 'prd135_drop_bucket_4', 'prd135_drop_bucket_5', 'prd135_drop_bucket_6', 'prd136_collapse_llm_tiers', 'prd139_tool_routing_graph', 'prd139_tool_routing_telemetry', 'prd140_permission_bypass_log', 'prd142_wave0_error_events', 'prd142_wave4_harness_store', 'prd142_wave5_drop_dead_tables', 'prd71_unified_skills', 'prd72_board_tasks', 'prd72_doc_access', 'prd72_fix_composio_trigger', 'prd72_memory_access_log', 'prd72_recipe_board_tasks', 'prd72_task_reconciliation', 'prd73_infra_alerts', 'prd74_voice_profiles', 'prd76_agent_reports', 'prd76_nullable_agent', 'prd77_agent_scheduled_tasks', 'prd79_memory_short_term', 'seed_auto_agents_existing_workspaces', 'unify_marketplace_categories')
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PRD-158 S1: real Teams entity. This revision also COLLAPSES the 44 pre-existing
    # alembic heads (see the down_revision tuple) into a single head, so the repo is
    # single-head again — the same migrations still run under `alembic upgrade heads`.
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("normalized_name", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()"), nullable=False),
        sa.UniqueConstraint("workspace_id", "normalized_name", name="uq_teams_workspace_normalized"),
    )
    op.create_index("ix_teams_workspace", "teams", ["workspace_id"])

    # Backfill DISTINCT teams from agents.team + documents.team_access (unnested).
    # normalized_name is the lowercased/trimmed canonical form (one team per
    # workspace+normalized); MIN(name) picks a deterministic display name, so
    # 'Support'/'support' collapse to a single 'Support' team.
    op.execute(
        """
        INSERT INTO teams (workspace_id, name, normalized_name)
        SELECT workspace_id, MIN(name) AS name, normalized_name
        FROM (
            SELECT workspace_id, TRIM(team) AS name, LOWER(TRIM(team)) AS normalized_name
            FROM agents
            WHERE team IS NOT NULL AND TRIM(team) <> ''
            UNION ALL
            SELECT workspace_id, TRIM(t) AS name, LOWER(TRIM(t)) AS normalized_name
            FROM documents, unnest(team_access) AS t
            WHERE team_access IS NOT NULL AND TRIM(t) <> ''
        ) src
        WHERE normalized_name <> ''
        GROUP BY workspace_id, normalized_name
        ON CONFLICT (workspace_id, normalized_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("ix_teams_workspace", table_name="teams")
    op.drop_table("teams")
