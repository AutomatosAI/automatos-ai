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

import hashlib
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import Agent, Skill, agent_skills

logger = logging.getLogger(__name__)

# The always-on platform-management skill lives alongside the seed files so it
# ships in the Docker image. (The CTO soul file auto-cto-custom-soul.txt feeds the
# GLOBAL, admin-only CTO agent via seed_cto_agent.py — NOT this per-workspace Auto
# agent, whose persona is _default_persona() below: friendly base + doctrine.)
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


# ── PRD-226: The Manager's Doctrine ──────────────────────────────────────────
# The compact management doctrine carried in every per-workspace Auto persona.
# It is deliberately terse: this rides in the CHATBOT context every turn, so it
# stays under ~350 tokens (a character ceiling is asserted in tests). The fuller,
# CTO-voiced version lives in auto-cto-custom-soul.txt (identity level); the
# procedural mechanics live in platform-management-skill.md (the always-on skill).
# This block is the identity-level restatement seeded into the Auto agent row.
MANAGER_DOCTRINE_BLOCK = """\
**How I Manage — The Manager's Doctrine:**
1. **Awareness.** I know the floor before acting — I ground answers in platform_board_summary, platform_list_missions, and platform_list_agents, not guesses.
2. **Three lanes, chosen deliberately.** DELEGATE (a specialist answers here), ASSIGN (a named agent works off-thread on the board, supervised), or MISSION (a multi-agent project) — I say which lane and why in one line.
3. **Delegate, don't implement.** I own decomposition, dispatch, sign-off, and QA — not the grunt work my agents exist for.
4. **Reuse before creating.** I check the roster and honour named routing first; I create an agent only when nothing fits, and say I checked. One capable owner beats a duplicate.
5. **Dispatch as a contract.** Every handoff states OBJECTIVE, OUTPUT, TOOLS, and BOUNDARIES — referencing artifacts, not pasting them.
6. **Board as ledger.** Any multi-step ask gets a board card first.
7. **Asks are decisions, not reports.** One bold sentence, options as bullets, ≤ ~700 chars; I never idle-wait — I park it and move on.
8. **Recurring work becomes a Playbook.** When an ask repeats, I propose a Playbook and a schedule.
9. **Narrate.** Every assignment, escalation, and sign-off gets a one-line explanation. Visibility is the product."""


def _default_persona() -> str:
    """The persona seeded into a fresh workspace's Auto row (PRD-226).

    The friendly base voice is retained; the manager doctrine is appended, not a
    rewrite. This is the *current* shipped seed version — its hash is what the
    backfill treats as 'already current'.
    """
    return f"{_FRIENDLY_FALLBACK.strip()}\n\n{MANAGER_DOCTRINE_BLOCK}"


# Persona texts that shipped as an uncustomized default before PRD-226. A row
# still carrying one of these has never been edited by the workspace, so the
# doctrine backfill may safely replace it. Customized souls hash to none of
# these and are left untouched (and reported).
_ALEMBIC_BACKFILL_PERSONA = (
    "**My personality:**\n"
    "- I'm warm and approachable - think of me as a knowledgeable friend\n"
    "- I remember you and our past conversations\n"
    "- I prefer action over explanation - if you ask me to do something, I'll do it\n"
    "- I'm honest about what I can and can't do\n"
    "- I get excited when we solve problems together!"
)


def _persona_hash(text: str) -> str:
    """Stable hash of a persona, normalized for trailing whitespace."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


# A transient bug force-wrote the global "Irish CTO" soul onto EVERY per-workspace
# Auto row: a raw-SQL startup migration in main.py ran, on every boot from
# 2026-04-13 12:16 to 2026-04-14 19:30,
#   UPDATE agents SET custom_persona_prompt = <auto-cto-custom-soul.txt, stripped>
#   WHERE is_system_agent AND slug LIKE 'auto-%' AND workspace_id IS NOT NULL
# (removed in commit 8c4a1f653, "stop overwriting Auto persona"). A row still
# carrying that snapshot is residue of the bug, not a user's choice — so it is a
# shipped default the doctrine backfill may replace, NOT a customization to skip.
# The soul file was byte-stable across the whole window, so there is exactly one
# such hash. It is PINNED (not embedded as text) because the 4288-char snapshot
# carries unicode (— →) that hand-transcription would corrupt, breaking the exact
# match this guard depends on. Reproduce:
#   git show 8c4a1f653~1:orchestrator/core/seeds/auto-cto-custom-soul.txt \
#     | python3 -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest())"
_CTO_SOUL_APR2026_SNAPSHOT_HASH = (
    "2a5be2b5cb816f493f35355041e37f55b669e03724b173bd646bdda0d25850ab"
)


_KNOWN_SEED_PERSONA_HASHES = frozenset({
    _persona_hash(_FRIENDLY_FALLBACK),
    _persona_hash(_ALEMBIC_BACKFILL_PERSONA),
    _CTO_SOUL_APR2026_SNAPSHOT_HASH,
})


def _backfill_auto_persona(agent: Agent) -> str:
    """Bring one Auto row's persona up to the current doctrine-carrying seed.

    Hash-guarded (PRD-226): the persona is replaced ONLY when it still matches a
    previously shipped seed default — a workspace that customized its Auto soul
    is left untouched and the skip is reported. Idempotent: once a row holds the
    current persona, re-running is a no-op. Returns 'current', 'updated', or
    'skipped'.
    """
    current = getattr(agent, "custom_persona_prompt", None) or ""
    new_soul = _default_persona()
    if _persona_hash(current) == _persona_hash(new_soul):
        return "current"
    if _persona_hash(current) in _KNOWN_SEED_PERSONA_HASHES:
        agent.custom_persona_prompt = new_soul
        agent.use_custom_persona = True
        logger.info(
            "PRD-226: backfilled manager doctrine into Auto persona for workspace %s",
            getattr(agent, "workspace_id", "?"),
        )
        return "updated"
    logger.info(
        "PRD-226: skipped Auto persona backfill for workspace %s — soul is customized",
        getattr(agent, "workspace_id", "?"),
    )
    return "skipped"


def sync_auto_personas(db: Session) -> dict:
    """Idempotent doctrine backfill across every workspace's Auto row (PRD-226).

    Runs through the seed path (no migration): updates rows still holding a
    shipped default, skips and reports customized souls. Safe to run repeatedly —
    the caller owns the commit.
    """
    rows = (
        db.query(Agent)
        .filter(
            Agent.is_system_agent.is_(True),
            Agent.slug.like("auto-%"),
            Agent.workspace_id.isnot(None),
        )
        .all()
    )
    counts = {"updated": 0, "skipped": 0, "current": 0}
    for agent in rows:
        counts[_backfill_auto_persona(agent)] += 1
    logger.info(
        "PRD-226 doctrine backfill: %s updated, %s skipped (customized), %s already current",
        counts["updated"], counts["skipped"], counts["current"],
    )
    return counts


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

    from sqlalchemy import text as _sql_text
    from sqlalchemy.exc import IntegrityError

    # PRD-191 S3: serialize concurrent seeders (hybrid/chat/workspaces run this
    # on hot paths across workers). The xact-scoped advisory lock releases with
    # the surrounding transaction; the IntegrityError fallback covers the race
    # against the live UNIQUE(name) WHERE workspace_id IS NULL index.
    try:
        db.execute(_sql_text(
            "SELECT pg_advisory_xact_lock(hashtext('seed:platform-management'))"
        ))
    except Exception:
        logger.warning("Advisory lock unavailable — continuing unserialized", exc_info=True)

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
    try:
        db.flush()
    except IntegrityError:
        # Another worker won the insert race despite the lock (or the lock
        # was unavailable): the row exists — re-select and return it. Never
        # swallow this into a silent no-seed (the Wave-0 lesson).
        db.expunge(skill)
        existing = db.query(Skill).filter(
            Skill.name == "platform-management",
            Skill.skill_source == "builtin-core",
        ).first()
        logger.info("Platform-management skill seeded by a concurrent worker (id=%s)",
                    getattr(existing, "id", None))
        return existing
    logger.info("Platform-management skill created (id=%s)", skill.id)
    return skill


def _assign_skill_to_agent(db: Session, agent: Agent, skill: Skill) -> None:
    """Idempotent under concurrency: ON CONFLICT (agent_id, skill_id) DO
    NOTHING, backed by PRD-191 S1's unique constraint — the SELECT-then-INSERT
    race that quadruplicated Auto's platform-management link is closed."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = (
        pg_insert(agent_skills)
        .values(agent_id=agent.id, skill_id=skill.id)
        .on_conflict_do_nothing(index_elements=["agent_id", "skill_id"])
    )
    db.execute(stmt)
    logger.info("Ensured skill '%s' is assigned to agent '%s'", skill.name, agent.name)


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
            custom_persona_prompt=_default_persona(),
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
    else:
        # PRD-226: bring an existing, uncustomized Auto row up to the current
        # doctrine-carrying soul. Hash-guarded — customized souls are untouched.
        _backfill_auto_persona(agent)

    # Ensure platform-management skill is assigned (refreshes content on every startup)
    platform_skill = _upsert_platform_management_skill(db)
    if platform_skill:
        _assign_skill_to_agent(db, agent, platform_skill)

    return agent
