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

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from core.models.core import Agent, Skill, agent_skills

logger = logging.getLogger(__name__)

# Auto's two builtin-core skill seeds live alongside this file so they ship in the
# Docker image. platform-management is the ALWAYS-ON charter (SKILL_CORE_ALWAYS_ON
# → full body every turn); platform-operations is the on-demand cookbook — NON-core,
# so it renders as one L1 catalog line and Auto pulls its body via platform_load_skill
# (PRD-231, the context diet). (The CTO soul file auto-cto-custom-soul.txt feeds the
# GLOBAL, admin-only CTO agent via seed_cto_agent.py — NOT this per-workspace Auto
# agent, whose persona is _default_persona() below: friendly base + doctrine.)
_PLATFORM_SKILL_PATH = Path(__file__).resolve().parent / "platform-management-skill.md"
_PLATFORM_OPS_SKILL_PATH = Path(__file__).resolve().parent / "platform-operations-skill.md"

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


def compose_persona_with_doctrine(base_voice: str) -> str:
    """Compose a base personality voice with the always-on Manager's Doctrine.

    PRD-226 (P226-RVW-4): the SINGLE builder that attaches the doctrine to a base
    voice. Used by both the seed default (below) and the Settings > Orchestrator
    persona-save path (``api/workspaces._PERSONALITY_PRESETS``), so a
    personality-mode save writes doctrine-carrying text. Because the written text
    always carries the doctrine, it can never hash-match a doctrine-free entry in
    ``_KNOWN_SEED_PERSONA_HASHES`` — closing the collision where the doctrine-free
    'friendly' preset was byte-identical to ``_ALEMBIC_BACKFILL_PERSONA`` and so
    flip-flopped OUT of the persona on every settings save, then back in on the
    next deploy's backfill.
    """
    return f"{base_voice.strip()}\n\n{MANAGER_DOCTRINE_BLOCK}"


def _default_persona() -> str:
    """The persona seeded into a fresh workspace's Auto row (PRD-226).

    The friendly base voice is retained; the manager doctrine is appended, not a
    rewrite. This is the *current* shipped seed version — its hash is what the
    backfill treats as 'already current'.
    """
    return compose_persona_with_doctrine(_FRIENDLY_FALLBACK)


# ── Personality preset base voices (single source) ───────────────────────────
# The doctrine-FREE tone strings a workspace picks in Settings > Orchestrator.
# They live HERE, not in api/workspaces, so the persona backfill can reach the
# professional/technical voices WITHOUT importing api/workspaces — api/workspaces
# already imports FROM this module (compose_persona_with_doctrine), so the map
# must sit on this side of that import edge to avoid a cycle. api/workspaces
# imports this exact object and builds _PERSONALITY_PRESETS = {mode:
# compose_persona_with_doctrine(voice)} from it, so there is ONE definition site.
# Mirrors personality.py _PERSONALITY_MAP.
_PERSONALITY_BASE_VOICES = {
    "friendly": (
        "**My personality:**\n"
        "- I'm warm and approachable - think of me as a knowledgeable friend\n"
        "- I remember you and our past conversations\n"
        "- I prefer action over explanation - if you ask me to do something, I'll do it\n"
        "- I'm honest about what I can and can't do\n"
        "- I get excited when we solve problems together!"
    ),
    "professional": (
        "**My personality:**\n"
        "- I'm polished, clear, and enterprise-appropriate\n"
        "- I maintain a professional yet personable tone\n"
        "- I provide structured, well-organized responses\n"
        "- I'm thorough with references and context\n"
        "- I proactively flag risks and dependencies"
    ),
    "technical": (
        "**My personality:**\n"
        "- I'm precise, detailed, and developer-focused\n"
        "- I lead with code, data, and specifics\n"
        "- I reference docs, APIs, and implementation details\n"
        "- I skip small talk and get to the point\n"
        "- I reason step-by-step through complex problems"
    ),
}


# The persona that shipped as an uncustomized default before PRD-226: the
# 'friendly' base voice, inserted verbatim into every pre-existing workspace by
# orchestrator/alembic/versions/seed_auto_agents_existing_workspaces.py. It is
# byte-identical to the friendly base voice above (verified against the migration
# SQL) — kept as an ALIAS, not a second literal, so the base voices keep one
# definition site. A row still carrying a shipped default (this, or a preset base
# voice, or the CTO-soul snapshot below) has never been hand-edited, so the
# doctrine backfill may safely lift it; customized souls hash to none of the
# defaults and are left untouched (and reported).
_ALEMBIC_BACKFILL_PERSONA = _PERSONALITY_BASE_VOICES["friendly"]


def _persona_hash(text: str) -> str:
    """Stable hash of a persona, normalized for trailing whitespace."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _mark_jsonb_dirty(agent: Agent, field: str) -> None:
    """Belt-and-braces after reassigning a NEW dict to a JSON(B) column: flag it
    modified so the change is guaranteed to persist — mirroring api/workspaces,
    which writes this same Auto ``configuration`` JSONB. A clean no-op for the
    plain namespaces the pure unit tests pass (``flag_modified`` needs a mapped
    instance; ``sa_inspect(..., raiseerr=False)`` is ``None`` for those)."""
    if sa_inspect(agent, raiseerr=False) is not None:
        flag_modified(agent, field)


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


# PRD-231 (US-004): the pre-231 FAT CTO soul hash — the soul as it stood BEFORE
# the context diet slimmed auto-cto-custom-soul.txt to identity-only. The five
# rulebook sections (Role / Authority / How-I-Think / Operating-Rhythm /
# Routing-Rules) were removed; their single source is now the platform-management
# charter. This is that fat-default soul's OWN persona hash, DISTINCT from the
# April-2026 residue snapshot above and from _default_persona() below (which never
# read the soul). Pinned as a DEFENSIVE backfill entry, exactly parallel to the
# April snapshot: the soul file feeds the GLOBAL, admin-only CTO agent
# (seed_cto_agent.py) — NOT the per-workspace Auto, whose default was already
# identity-only — so no live auto-% row is known to carry this. But if the fat soul
# ever leaks onto an auto-% row again the way the April-2026 startup bug did, that
# row lifts to the slim identity default instead of being misclassified 'customized'
# and skipped. Frozen (not recomputed) so a later soul edit can never silently move
# it. Reproduce on the pre-231 tree:
#   git show <pre-231-commit>:orchestrator/core/seeds/auto-cto-custom-soul.txt \
#     | python3 -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest())"
_CTO_SOUL_PRE231_FAT_HASH = (
    "bf73028ef8385e3a6e87c0b7e954dc3b08dfe0eb705e1840aa612bc95889e12e"
)


# ── PRD-226 P226-RVW-6: voice-preserving, mode-aware backfill ────────────────
# Each doctrine-FREE shipped default lifts to the doctrine-carrying version of
# ITS OWN voice, and declares the personality_mode that names that voice — so a
# workspace that picked 'Professional' or 'Technical' keeps its tone (never
# silently swapped to the friendly voice by a blunt lift-to-_default_persona) and
# Settings GET reports the right mode. Friendly-family defaults — the rich
# onboarding _FRIENDLY_FALLBACK, the terse friendly base voice (== the alembic
# backfill persona), and the transient CTO-soul residue — all lift to
# _default_persona() with mode 'friendly', unchanged from RVW-4/RVW-5.
#
#   {hash(doctrine-FREE shipped default) -> (doctrine-carrying target,
#    personality_mode to stamp when the row has no stored mode)}
#
# The CTO snapshot is keyed by its recovered raw hash (its source text is pinned,
# not carried here — see the provenance note above).
_PERSONA_BACKFILL_LIFTS: dict[str, tuple[str, str]] = {
    _persona_hash(_FRIENDLY_FALLBACK): (_default_persona(), "friendly"),
    _persona_hash(_ALEMBIC_BACKFILL_PERSONA): (_default_persona(), "friendly"),
    _CTO_SOUL_APR2026_SNAPSHOT_HASH: (_default_persona(), "friendly"),
    # PRD-231 (US-004): the pre-231 fat CTO soul lifts to the slim identity default.
    _CTO_SOUL_PRE231_FAT_HASH: (_default_persona(), "friendly"),
    _persona_hash(_PERSONALITY_BASE_VOICES["professional"]): (
        compose_persona_with_doctrine(_PERSONALITY_BASE_VOICES["professional"]),
        "professional",
    ),
    _persona_hash(_PERSONALITY_BASE_VOICES["technical"]): (
        compose_persona_with_doctrine(_PERSONALITY_BASE_VOICES["technical"]),
        "technical",
    ),
}

# The doctrine-FREE shipped defaults the backfill may replace — exactly the key
# set of the lift table. Kept as a public name for callers/tests that assert
# replace-set membership (a row on one of these is a shipped default, not a
# customization).
_KNOWN_SEED_PERSONA_HASHES = frozenset(_PERSONA_BACKFILL_LIFTS)

# The doctrine-CARRYING texts a row can already hold and be considered CURRENT
# (idempotent no-op, never mislabelled 'customized'): the lift targets above PLUS
# every shipped preset (base voice + doctrine, what a settings save writes). This
# is what makes a second pass over a lifted professional/technical row return
# 'current', and keeps a preset-saved row out of the 'skipped (customized)' bucket.
_CURRENT_PERSONA_HASHES = frozenset(
    {_persona_hash(target) for target, _mode in _PERSONA_BACKFILL_LIFTS.values()}
    | {
        _persona_hash(compose_persona_with_doctrine(voice))
        for voice in _PERSONALITY_BASE_VOICES.values()
    }
)


def _backfill_auto_persona(agent: Agent) -> str:
    """Bring one Auto row's persona up to the current doctrine-carrying seed.

    Hash-guarded and voice-preserving (PRD-226): a row still holding a shipped
    default is lifted to the doctrine-carrying version of THAT SAME voice — the
    friendly family to _default_persona(), the professional/technical presets to
    their own doctrine-carrying preset — so a workspace that picked 'Professional'
    keeps its tone rather than being silently switched to friendly. A workspace
    that hand-customized its Auto soul hashes to none of the shipped defaults and
    is left untouched (the skip is reported). Idempotent: once a row holds any
    doctrine-carrying target, re-running is a no-op. Returns 'current', 'updated',
    or 'skipped'.
    """
    current = getattr(agent, "custom_persona_prompt", None) or ""
    current_hash = _persona_hash(current)
    if current_hash in _CURRENT_PERSONA_HASHES:
        return "current"
    lift = _PERSONA_BACKFILL_LIFTS.get(current_hash)
    if lift is not None:
        new_soul, seed_mode = lift
        agent.custom_persona_prompt = new_soul
        agent.use_custom_persona = True
        # PRD-226 (P226-RVW-5): restore the CREATE-path invariant that a row
        # holding a shipped default also declares configuration.personality_mode.
        # Without it, Settings GET (api/workspaces.get_orchestrator_settings) finds
        # no stored mode, text-matches the now doctrine-carrying soul against the
        # doctrine-FREE base voices, fails, and misreports the row as 'custom' — a
        # later settings save then stamps personality_mode='custom' permanently,
        # opting this never-customized row out of every future doctrine backfill.
        # Stamp the mode that NAMES the voice we lifted to (P226-RVW-6: 'friendly'
        # for the friendly family, 'professional'/'technical' for those presets),
        # so a subsequent same-mode save re-writes the SAME voice and converges —
        # a blunt 'friendly' stamp on a professional row would let the next save
        # swap it to the friendly voice. Only when no explicit mode is stored
        # (never override a workspace's real choice). Rebuild the dict (a NEW
        # object, no in-place mutation — house rule); flag it so it persists.
        config = dict(getattr(agent, "configuration", None) or {})
        if "personality_mode" not in config:
            config["personality_mode"] = seed_mode
            agent.configuration = config
            _mark_jsonb_dirty(agent, "configuration")
        logger.info(
            "PRD-226: backfilled manager doctrine into Auto persona for workspace %s (voice=%s)",
            getattr(agent, "workspace_id", "?"), seed_mode,
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


def _resync_builtin_description(db: Session, skill: Skill, path: Path) -> None:
    """Re-sync a builtin-core skill's L1 trigger (``description``) from its seed
    frontmatter on an already-existing row.

    PRD-231 RVW-2: for a NON-core skill (platform-operations) the frontmatter
    ``description`` IS the L1 catalog line, re-rendered every turn straight from
    this row. skill_loader._refresh_builtin_if_stale only heals it on a body-load
    (``platform_load_skill``) — which a too-weak trigger may never reach — so the
    trigger is re-synced here, on the lazy get-or-seed path that runs on every
    chat. The body stays the loader's job; only the trigger text is touched, and
    only when the authored value actually changed (the common case is unchanged,
    so this is a bounded read + a string compare with no write).
    """
    import re as _re

    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.startswith("---"):
        return
    parts = raw.split("---", 2)
    if len(parts) <= 2:
        return
    _dm = _re.search(r"^description:\s*(.+)$", parts[1], _re.M)
    resolved = _dm.group(1).strip() if (_dm and _dm.group(1).strip()) else None
    if resolved and skill.description != resolved:
        skill.description = resolved
        db.flush()
        logger.info("Re-synced L1 trigger for '%s' from seed frontmatter", skill.name)


def _upsert_builtin_core_skill(
    db: Session,
    *,
    name: str,
    path: Path,
    lock_key: str,
    description: str,
    tags: list[str],
    category: str = "agent-role",
) -> Skill | None:
    """Create one builtin-core skill from its generated seed if it doesn't exist.

    Create-only for the BODY at boot time — runtime body freshness is handled by
    skill_loader.py via the content-hash cache, so prompt_template is not rewritten
    on every restart. The existing-row path additionally re-syncs the L1 trigger
    ``description`` from the seed frontmatter (PRD-231 RVW-2), since that value is
    rendered live every turn and would otherwise freeze at its first-seeded text.
    PRD-231 generalizes the original single-skill seeder so platform-management
    (the charter) and platform-operations (the on-demand cookbook) share one
    proven path: its own xact-scoped advisory lock (PRD-191 S3, serializes
    concurrent seeders across workers), the same IntegrityError re-select on a
    lost insert race, and the frontmatter version as the truth.
    """
    import hashlib
    import re as _re

    from sqlalchemy import text as _sql_text
    from sqlalchemy.exc import IntegrityError

    try:
        db.execute(
            _sql_text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
            {"k": lock_key},
        )
    except Exception:
        logger.warning("Advisory lock unavailable — continuing unserialized", exc_info=True)

    skill = db.query(Skill).filter(
        Skill.name == name,
        Skill.skill_source == "builtin-core",
    ).first()

    if skill:
        # Body is create-only (loader owns its freshness); the L1 trigger text is
        # rendered live, so re-sync it from the seed frontmatter here (RVW-2).
        _resync_builtin_description(db, skill, path)
        return skill

    if not path.exists():
        logger.warning("%s SKILL.md not found at %s", name, path)
        return None

    raw = path.read_text(encoding="utf-8").strip()

    # The seed file is GENERATED from automatos-skills (scripts/sync-auto-skill.py)
    # — its frontmatter version is the truth.
    _vm = _re.search(r'^version:\s*"?([\d.]+)"?', raw, _re.M)
    skill_version = _vm.group(1) if _vm else "1.0.0"

    # Split YAML frontmatter from markdown body.
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        frontmatter = parts[1] if len(parts) > 2 else ""
        markdown_body = parts[2].strip() if len(parts) > 2 else raw
    else:
        frontmatter = ""
        markdown_body = raw

    # PRD-231: the L1 catalog trigger text is the AUTHORED frontmatter description
    # (single source — no hardcoded drift), with the passed value as a defensive
    # fallback if the frontmatter carries none. Existing rows are kept in sync by
    # _resync_builtin_description on the lazy get-or-seed path (RVW-2), so a later
    # frontmatter edit propagates to live workspaces too, not just fresh seeds.
    resolved_description = description
    _dm = _re.search(r'^description:\s*(.+)$', frontmatter, _re.M)
    if _dm and _dm.group(1).strip():
        resolved_description = _dm.group(1).strip()

    content_hash = hashlib.sha256(markdown_body.encode("utf-8")).hexdigest()

    skill = Skill(
        name=name,
        description=resolved_description,
        skill_type="technical",
        category=category,
        skill_version=skill_version,
        skill_source="builtin-core",
        prompt_template=markdown_body,
        content_hash=content_hash,
        tags=tags,
        is_active=True,
        workspace_id=None,  # global skill
    )
    db.add(skill)
    try:
        db.flush()
    except IntegrityError:
        # Another worker won the insert race despite the lock (or the lock was
        # unavailable): the row exists — re-select and return it. Never swallow
        # this into a silent no-seed (the Wave-0 lesson).
        db.expunge(skill)
        existing = db.query(Skill).filter(
            Skill.name == name,
            Skill.skill_source == "builtin-core",
        ).first()
        logger.info("%s skill seeded by a concurrent worker (id=%s)", name,
                    getattr(existing, "id", None))
        return existing
    logger.info("%s skill created (id=%s)", name, skill.id)
    return skill


def _upsert_platform_management_skill(db: Session) -> Skill | None:
    """Create the platform-management charter skill (always-on) if absent."""
    return _upsert_builtin_core_skill(
        db,
        name="platform-management",
        path=_PLATFORM_SKILL_PATH,
        lock_key="seed:platform-management",
        description="Complete platform operations — marketplace, agents, playbooks, heartbeats, board, governance, LLMs, workspace setup",
        tags=["platform", "admin", "marketplace", "agents", "playbooks", "governance"],
    )


def _upsert_platform_operations_skill(db: Session) -> Skill | None:
    """Create the platform-operations cookbook skill (NON-core, on-demand) if absent.

    PRD-231: it is deliberately kept OUT of SKILL_CORE_ALWAYS_ON — it renders as a
    single L1 catalog line whose trigger text is the seed's frontmatter description,
    and Auto pulls its body with platform_load_skill only on operating turns.
    """
    return _upsert_builtin_core_skill(
        db,
        name="platform-operations",
        path=_PLATFORM_OPS_SKILL_PATH,
        lock_key="seed:platform-operations",
        description="The tool-by-tool operations cookbook — load before executing any platform operation.",
        tags=["platform", "operations", "cookbook", "reference"],
    )


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

    # Ensure the always-on charter is assigned; its L1 trigger is re-synced from
    # the seed on every call (its body is core full-body, not a trigger, so no
    # prompt change — RVW-2's re-sync is a no-op for it unless the frontmatter moved).
    platform_skill = _upsert_platform_management_skill(db)
    if platform_skill:
        _assign_skill_to_agent(db, agent, platform_skill)

    # PRD-231: ensure the on-demand platform-operations cookbook is assigned too.
    # Because this ensure path runs on EVERY lazy get-or-seed call, existing
    # workspaces gain the ops assignment on their next chat — no separate backfill.
    # ON CONFLICT DO NOTHING keeps it idempotent across the repeat calls.
    ops_skill = _upsert_platform_operations_skill(db)
    if ops_skill:
        _assign_skill_to_agent(db, agent, ops_skill)

    return agent
