"""
Seed per-workspace Auto Agent
==============================

Every workspace gets exactly one Auto agent — the default chat agent for that
workspace. It is the single source of truth for the workspace's orchestrator
LLM config, persona, and configuration. The Settings > Orchestrator page
reads from and writes to this agent row.

Idempotent: upserts on slug='auto-{workspace_id}'.

Unlike the CTO agent (global, admin-only), Auto agents are:
  - workspace-scoped (workspace_id is set)
  - visible to all workspace members (required_role=None)
  - hidden from the Roster UI (is_system_agent=True + workspace_id != None)
"""

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import Agent

logger = logging.getLogger(__name__)

# Default persona — matches personality.py _FRIENDLY_PERSONALITY preset
_DEFAULT_PERSONA = """\
**My personality:**
- I'm warm and approachable - think of me as a knowledgeable friend
- I remember you and our past conversations
- I prefer action over explanation - if you ask me to do something, I'll do it
- I'm honest about what I can and can't do
- I get excited when we solve problems together!"""


def _get_default_model_config() -> dict:
    """Build default model_config from system_settings / env.

    Reads from the global system_settings table (orchestrator_llm category)
    which serves as the deployment-level default for new workspaces.
    Falls back to env vars (LLM_PROVIDER, LLM_MODEL) if DB is unavailable.
    """
    try:
        from config import config
        provider = config.LLM_PROVIDER or "openrouter"
        model_id = config.LLM_MODEL or "openai/gpt-4o"
    except Exception:
        provider = "openrouter"
        model_id = "openai/gpt-4o"

    return {
        "provider": provider,
        "model_id": model_id,
        "temperature": 0.7,
        "max_tokens": 4000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": None,
    }


def seed_auto_agent(db: Session, workspace_id: UUID) -> Agent:
    """Create or return the Auto agent for a workspace.

    Safe to call multiple times — returns existing agent if already seeded.
    """
    slug = f"auto-{workspace_id}"

    existing = (
        db.query(Agent)
        .filter(Agent.slug == slug, Agent.workspace_id == workspace_id)
        .first()
    )
    if existing:
        return existing

    agent = Agent(
        name="Auto",
        slug=slug,
        description="Your workspace AI orchestrator — the default agent for chat and settings.",
        agent_type="system",
        status="active",
        is_system_agent=True,
        required_role=None,
        workspace_id=workspace_id,
        owner_type="workspace",
        owner_id=str(workspace_id),
        use_custom_persona=True,
        custom_persona_prompt=_DEFAULT_PERSONA,
        model_config=_get_default_model_config(),
        configuration={
            "thinking_level": "medium",
            "proactive_level": "notify",
            "communication_style": "balanced",
            "personality_mode": "friendly",
        },
        tags=["auto", "system", "orchestrator"],
    )
    db.add(agent)
    db.flush()
    logger.info("Seeded Auto agent for workspace %s (agent.id=%s)", workspace_id, agent.id)
    return agent
