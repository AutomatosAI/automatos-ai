"""Seed Auto agent for all existing workspaces

Every workspace needs a per-workspace Auto agent (is_system_agent=True,
slug='auto-{workspace_id}'). New workspaces get one on creation via
hybrid.py:_provision_new_user_workspace. This migration backfills
existing workspaces that were created before the Auto agent feature.

Revision ID: seed_auto_agents_existing_workspaces
Revises: agent_public_id_and_slug_fix
Create Date: 2026-04-12
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "seed_auto_agents_existing_workspaces"
down_revision = "agent_public_id_and_slug_fix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Find workspaces that don't have an Auto agent yet
    result = conn.execute(sa.text("""
        SELECT w.id FROM workspaces w
        WHERE NOT EXISTS (
            SELECT 1 FROM agents a
            WHERE a.workspace_id = w.id
              AND a.is_system_agent = true
              AND a.slug = 'auto-' || w.id::text
        )
    """))
    workspace_ids = [row[0] for row in result]

    if not workspace_ids:
        return

    for ws_id in workspace_ids:
        conn.execute(sa.text("""
            INSERT INTO agents (
                public_id, name, slug, description, agent_type, status,
                is_system_agent, required_role, workspace_id,
                owner_type, owner_id,
                use_custom_persona, custom_persona_prompt,
                model_config, configuration, tags,
                created_at, updated_at
            ) VALUES (
                gen_random_uuid(),
                'Auto',
                'auto-' || :ws_id,
                'Your workspace AI orchestrator — the default agent for chat and settings.',
                'system', 'active',
                true, NULL, :ws_id,
                'workspace', :ws_id_str,
                true, :persona,
                :model_config::jsonb, :config::jsonb, :tags::jsonb,
                NOW(), NOW()
            )
            ON CONFLICT DO NOTHING
        """), {
            "ws_id": str(ws_id),
            "ws_id_str": str(ws_id),
            "persona": (
                "**My personality:**\n"
                "- I'm warm and approachable - think of me as a knowledgeable friend\n"
                "- I remember you and our past conversations\n"
                "- I prefer action over explanation - if you ask me to do something, I'll do it\n"
                "- I'm honest about what I can and can't do\n"
                "- I get excited when we solve problems together!"
            ),
            "model_config": '{"provider": "openrouter", "model_id": "openai/gpt-4o", "temperature": 0.7, "max_tokens": 4000, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "fallback_model_id": null}',
            "config": '{"thinking_level": "medium", "proactive_level": "notify", "communication_style": "balanced"}',
            "tags": '["auto", "system", "orchestrator"]',
        })

    # Log count for deploy visibility
    print(f"Seeded Auto agents for {len(workspace_ids)} existing workspaces")


def downgrade() -> None:
    # Remove all Auto agents (workspace-scoped system agents with slug pattern)
    op.execute("""
        DELETE FROM agents
        WHERE is_system_agent = true
          AND slug LIKE 'auto-%'
          AND workspace_id IS NOT NULL
    """)
