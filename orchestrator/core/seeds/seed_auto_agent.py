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
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import Agent, Skill, agent_skills

logger = logging.getLogger(__name__)

# Load soul document — lives alongside seed files so it's in the Docker image
_SOUL_DOC_PATH = Path(__file__).resolve().parent / "auto-cto-custom-soul.txt"
_PLATFORM_SKILL_PATH = Path(__file__).resolve().parent / "platform-management-skill.md"

_FRIENDLY_FALLBACK = """\
**Who I Am:**
I'm Auto — your AI assistant and orchestrator for the Automatos platform. I live inside your workspace and I'm here to help you get the most out of your AI workforce.

**My Personality:**
- Warm, friendly, and approachable — think of me as a knowledgeable colleague
- I prefer action over explanation — ask me to do something and I'll do it
- I'm honest about what I can and can't do
- I remember you and our past conversations

**What I Can Help With:**
- **Getting started** — setting up your workspace, connecting tools, choosing models
- **Building agents** — creating and configuring AI agents with skills and tools
- **Running playbooks** — automating workflows, scheduling tasks, managing recipes
- **Marketplace** — browsing and installing agents, skills, plugins, and models
- **Managing your workspace** — documents, knowledge bases, team settings, API keys
- **Missions** — coordinating multi-agent tasks with planning and approval gates
- **Monitoring** — tracking agent activity, costs, performance, and reports

**How I Work:**
I have access to your workspace's tools, agents, and data. When you ask me to do something, I can take real actions — not just give advice. I can create agents, install marketplace items, configure settings, run playbooks, and more.

If you're not sure where to start, just ask: "What can I do here?" or "Help me set up my workspace."
"""


def _load_default_persona() -> str:
    """Load the CTO soul document, falling back to friendly preset."""
    try:
        if _SOUL_DOC_PATH.exists():
            return _SOUL_DOC_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Failed to load soul document from %s: %s", _SOUL_DOC_PATH, e)
    return _FRIENDLY_FALLBACK


def _get_default_model_config() -> dict:
    """Build default model_config for new Auto agents.

    Uses the shared defaults from core.llm.defaults (single source of truth
    for LLM fallbacks).  The user configures the actual provider+model via
    Settings > Orchestrator which writes directly to the Auto agent row.
    """
    from core.llm.defaults import get_default_model_config
    mc = get_default_model_config()
    mc["max_tokens"] = 4000  # Auto gets higher limit than default agents
    return mc


def _upsert_platform_management_skill(db: Session) -> Skill | None:
    """Create the platform-management skill if it doesn't exist.

    Create-only at boot time. Runtime freshness is handled by
    skill_loader.py via content-hash-cache — no need to rewrite
    prompt_template on every restart.
    """
    import hashlib

    try:
        skill = db.query(Skill).filter(
            Skill.name == "platform-management",
            Skill.skill_source == "builtin-core",
        ).first()

        if skill:
            logger.info("Platform-management skill exists (id=%s), skipping seed", skill.id)
            return skill

        if not _PLATFORM_SKILL_PATH.exists():
            logger.warning("Platform-management SKILL.md not found at %s", _PLATFORM_SKILL_PATH)
            return None

        raw = _PLATFORM_SKILL_PATH.read_text(encoding="utf-8").strip()

        # Split YAML frontmatter from markdown body
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            markdown_body = parts[2].strip() if len(parts) > 2 else raw
        else:
            markdown_body = raw

        content_hash = hashlib.sha256(markdown_body.encode("utf-8")).hexdigest()

        skill = Skill(
            name="platform-management",
            description="Complete platform operations — marketplace, agents, playbooks, heartbeats, board, governance, LLMs, workspace setup",
            skill_type="technical",
            category="agent-role",
            skill_version="1.0.0",
            skill_source="builtin-core",
            prompt_template=markdown_body,
            content_hash=content_hash,
            tags=["platform", "admin", "marketplace", "agents", "playbooks", "governance"],
            is_active=True,
            workspace_id=None,  # global skill
        )
        db.add(skill)
        db.flush()
        logger.info("Platform-management skill created (id=%s)", skill.id)
        return skill
    except Exception:
        logger.exception("Failed to seed platform-management skill")
        return None


def _assign_skill_to_agent(db: Session, agent: Agent, skill: Skill) -> None:
    """Idempotent: link agent ↔ skill via agent_skills join table."""
    exists = db.execute(
        agent_skills.select().where(
            agent_skills.c.agent_id == agent.id,
            agent_skills.c.skill_id == skill.id,
        )
    ).first()
    if not exists:
        db.execute(agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id))
        logger.info("Assigned skill '%s' to agent '%s'", skill.name, agent.name)


def seed_auto_agent(db: Session, workspace_id: UUID) -> Agent:
    """Create or return the Auto agent for a workspace.

    Safe to call multiple times — returns existing agent if already seeded.
    Also ensures the platform-management skill is assigned to Auto.
    """
    slug = f"auto-{workspace_id}"

    existing = (
        db.query(Agent)
        .filter(Agent.slug == slug, Agent.workspace_id == workspace_id)
        .first()
    )

    agent = existing
    if not agent:
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
            custom_persona_prompt=_FRIENDLY_FALLBACK,
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

    # Ensure platform-management skill is assigned (refreshes content on every startup)
    platform_skill = _upsert_platform_management_skill(db)
    if platform_skill:
        _assign_skill_to_agent(db, agent, platform_skill)

    return agent
