"""Agent assignment handlers for PlatformActionExecutor (PRD-71) — assign tool/skill/plugin, configure heartbeat."""

import logging
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# How many agents an error message names before it stops being helpful.
ROSTER_IN_ERROR = 20


def resolve_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]):
    """Resolve agent by ID or name within a workspace. Returns (agent, error_dict)."""
    from core.models import Agent

    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")

    if not agent_id and not agent_name:
        return None, {"success": False, "error": "Provide agent_id or agent_name"}

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)
    if agent_id:
        query = query.filter(Agent.id == agent_id)
    else:
        query = query.filter(Agent.name.ilike(f"%{agent_name}%"))

    agent = query.first()
    if not agent:
        # Name the roster. Night 1: Auto passed a TICKET number as an agent id
        # and got a bare "Agent not found" eight times in a row on T64 — with
        # nothing in the message to tell it what a valid agent looks like, it
        # simply tried again (F032).
        roster = [
            f"{a.id}:{a.name}"
            for a in db.query(Agent)
            .filter(Agent.workspace_id == workspace_id, Agent.status == "active")
            .order_by(Agent.id)
            .limit(ROSTER_IN_ERROR)
            .all()
        ]
        asked = f"agent_id={agent_id!r}" if agent_id else f"agent_name={agent_name!r}"
        hint = (
            f"No agent matches {asked} in this workspace. "
            "Agent ids and TICKET ids are different numbering — if that number came "
            "from a task, it is not an agent id. "
        )
        hint += f"This workspace's agents are: {', '.join(roster)}." if roster else (
            "This workspace has no active agents."
        )
        return None, {"success": False, "error": hint}

    return agent, None


def _connection(db: Session, workspace_id: UUID, app_name: str, said: str) -> Dict[str, Any]:
    """F188 (night 6): whether the assigned app is connected, and a message that
    says so. At 02:03:19 GMAIL was assigned with no app connected, the result
    said nothing of it, and Auto told the owner Gmail was ready."""
    from core.composio.entity_manager import EntityManager

    try:
        connected = app_name in {a.upper().strip() for a in EntityManager(db).get_connected_apps(workspace_id)}
    except Exception:  # noqa: BLE001 -- unknown is not connected: never "ready"
        logger.warning("[F188] could not read the connected apps for %s", workspace_id, exc_info=True)
        connected = False
    if connected:
        return {"connected": True, "message": f"{said}."}
    shown = app_name.replace("_", " ").title()
    return {"connected": False, "message": (
        f"{said}, but {app_name} is NOT connected for this workspace, so the agent cannot use it yet. Tell the "
        f"owner it is assigned but not connected — never that it is ready. They connect it on "
        f"Tools & Integrations → {shown} → Connect.")}


async def assign_tool_to_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a Composio tool/app to an agent."""
    from core.models.composio_cache import AgentAppAssignment

    agent, err = resolve_agent(db, workspace_id, params)
    if err:
        return err

    app_name = params.get("app_name")
    if not app_name:
        return {"success": False, "error": "Missing required parameter: app_name"}

    app_name = app_name.upper()

    # Idempotency: check existing assignment
    existing = (
        db.query(AgentAppAssignment)
        .filter(
            AgentAppAssignment.agent_id == agent.id,
            AgentAppAssignment.app_name == app_name,
        )
        .first()
    )

    if existing:
        if existing.is_active:
            return {
                "success": True,
                "already_assigned": True,
                "agent": {"id": agent.id, "name": agent.name},
                "app_name": app_name,
                **_connection(db, workspace_id, app_name, f"Tool '{app_name}' is already assigned to agent '{agent.name}'"),
            }
        # Re-activate
        existing.is_active = True
        db.flush()
        logger.info("[PlatformExecutor] Re-activated tool '%s' for agent %d", app_name, agent.id)
        return {
            "success": True,
            "reactivated": True,
            "agent": {"id": agent.id, "name": agent.name},
            "app_name": app_name,
            **_connection(db, workspace_id, app_name, f"Tool '{app_name}' re-activated for agent '{agent.name}'"),
        }

    # Create assignment
    assignment = AgentAppAssignment(
        agent_id=agent.id,
        app_name=app_name,
        app_type="EXTERNAL",
        is_active=True,
    )
    db.add(assignment)
    db.flush()

    logger.info("[PlatformExecutor] Assigned tool '%s' to agent '%s' (id=%d)", app_name, agent.name, agent.id)

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "app_name": app_name,
        **_connection(db, workspace_id, app_name, f"Tool '{app_name}' assigned to agent '{agent.name}'"),
    }


INSTALLED_SKILLS_NAMED = 12


def _installed_skills(db: Session, workspace_id: UUID) -> list:
    """The active skills this workspace may assign: its own and the marketplace
    skills enabled for it, by name."""
    from sqlalchemy import and_, or_

    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    enabled = db.query(WorkspaceEnabledSkill.skill_id).filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
    return (
        db.query(Skill)
        .filter(
            Skill.is_active.is_(True),
            or_(Skill.workspace_id == workspace_id,
                and_(Skill.workspace_id.is_(None), Skill.id.in_(enabled.subquery()))),
        )
        .order_by(Skill.name)
        .all()
    )


def _pick_installed_skill(installed: list, skill_id: Any, skill_name: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """(skill, refusal): by id, by whole name, or by the one installed name containing it."""
    if skill_id:
        try:
            wanted_id = int(skill_id)
        except (TypeError, ValueError):
            return None, {"success": False, "error": f"skill_id must be a number, got {skill_id!r}"}
        return next((s for s in installed if s.id == wanted_id), None), None
    wanted = str(skill_name).strip().lower()
    exact = next((s for s in installed if (s.name or "").strip().lower() == wanted), None)
    if exact is not None:
        return exact, None
    partial = [s for s in installed if wanted and wanted in (s.name or "").lower()]
    if len(partial) > 1:
        named = ", ".join(f"'{s.name}'" for s in partial[:INSTALLED_SKILLS_NAMED])
        return None, {"success": False, "error": f"{len(partial)} installed skills match '{skill_name}': {named}. Use one exactly."}
    return (partial[0] if partial else None), None


async def _skill_not_installed(db: Session, workspace_id: UUID, requested: str, installed: list) -> Dict[str, Any]:
    from modules.tools.discovery.handlers_marketplace import browse_marketplace_skills
    from modules.tools.discovery.not_found_candidates import find_candidates

    here = ", ".join(f"'{s.name}'" for s in installed[:INSTALLED_SKILLS_NAMED]) or "none yet"
    candidates = await find_candidates(browse_marketplace_skills, db, workspace_id, requested, list_key="skills")
    market = ", ".join(f"'{c.get('slug') or c.get('name')}'" for c in candidates)
    error = (
        f"No skill '{requested}' is installed in this workspace. Installed here: {here}."
        + (f" In the marketplace: {market} (install one with platform_install_skill, then assign it)." if market else "")
        + " Search with platform_browse_marketplace_skills — never guess a skill's name."
    )
    return {"success": False, "error": error, "requested": requested,
            "installed": [s.name for s in installed], "candidates": candidates}


async def assign_skill_to_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a skill to an agent via the agent_skills M2M table."""
    from core.models.core import agent_skills

    agent, err = resolve_agent(db, workspace_id, params)
    if err:
        return err

    skill_id = params.get("skill_id")
    skill_name = params.get("skill_name")

    if not skill_id and not skill_name:
        return {"success": False, "error": "Provide skill_id or skill_name"}

    # F184 (night 6): asked for 'data-analysis', Auto got a bare "Skill not found"
    # and offered to install a name it made up while 'spreadsheet-qa' was
    # installed. A skill is assigned from this workspace's own: its skills and the
    # marketplace skills enabled for it (the lookup read every workspace's). A miss
    # names what is installed and what the marketplace has.
    installed = _installed_skills(db, workspace_id)
    skill, refusal = _pick_installed_skill(installed, skill_id, skill_name)
    if refusal:
        return refusal
    if not skill:
        return await _skill_not_installed(db, workspace_id, str(skill_name or skill_id), installed)

    # Idempotency: check if already assigned
    from sqlalchemy import select as sa_select
    existing = db.execute(
        sa_select(agent_skills).where(
            agent_skills.c.agent_id == agent.id,
            agent_skills.c.skill_id == skill.id,
        )
    ).first()

    if existing:
        return {
            "success": True,
            "already_assigned": True,
            "agent": {"id": agent.id, "name": agent.name},
            "skill": {"id": skill.id, "name": skill.name},
            "message": f"Skill '{skill.name}' is already assigned to agent '{agent.name}'.",
        }

    # Insert into M2M table
    db.execute(
        agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id)
    )
    db.flush()

    logger.info("[PlatformExecutor] Assigned skill '%s' (id=%d) to agent '%s' (id=%d)",
                 skill.name, skill.id, agent.name, agent.id)

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "skill": {"id": skill.id, "name": skill.name},
        "message": f"Skill '{skill.name}' assigned to agent '{agent.name}'.",
    }


async def assign_plugin_to_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a marketplace plugin to an agent."""
    from core.models.marketplace_plugins import (
        MarketplacePlugin, WorkspaceEnabledPlugin, AgentAssignedPlugin,
    )

    agent, err = resolve_agent(db, workspace_id, params)
    if err:
        return err

    plugin_id = params.get("plugin_id")
    plugin_slug = params.get("plugin_slug")

    if not plugin_id and not plugin_slug:
        return {"success": False, "error": "Provide plugin_id or plugin_slug"}

    # Resolve plugin
    query = db.query(MarketplacePlugin)
    if plugin_id:
        from uuid import UUID as _UUID
        query = query.filter(MarketplacePlugin.id == _UUID(str(plugin_id)))
    else:
        query = query.filter(MarketplacePlugin.slug == plugin_slug)

    plugin = query.first()
    if not plugin:
        return {"success": False, "error": "Plugin not found"}

    # Verify plugin is enabled for this workspace
    ws_enabled = (
        db.query(WorkspaceEnabledPlugin)
        .filter(
            WorkspaceEnabledPlugin.workspace_id == workspace_id,
            WorkspaceEnabledPlugin.plugin_id == plugin.id,
        )
        .first()
    )
    if not ws_enabled:
        return {
            "success": False,
            "error": f"Plugin '{plugin.name}' is not enabled for this workspace. Install it first with platform_install_plugin.",
        }

    # Idempotency check
    existing = (
        db.query(AgentAssignedPlugin)
        .filter(
            AgentAssignedPlugin.agent_id == agent.id,
            AgentAssignedPlugin.plugin_id == plugin.id,
        )
        .first()
    )
    if existing:
        return {
            "success": True,
            "already_assigned": True,
            "agent": {"id": agent.id, "name": agent.name},
            "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
            "message": f"Plugin '{plugin.name}' is already assigned to agent '{agent.name}'.",
        }

    # Create assignment
    assignment = AgentAssignedPlugin(
        agent_id=agent.id,
        plugin_id=plugin.id,
    )
    db.add(assignment)
    db.flush()

    logger.info("[PlatformExecutor] Assigned plugin '%s' to agent '%s' (id=%d)",
                 plugin.name, agent.name, agent.id)

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
        "message": f"Plugin '{plugin.name}' assigned to agent '{agent.name}'.",
    }


async def configure_agent_heartbeat(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Configure or update the heartbeat schedule for an agent."""
    from sqlalchemy.orm.attributes import flag_modified

    agent, err = resolve_agent(db, workspace_id, params)
    if err:
        return err

    # Read current configuration (immutable pattern -- build new dict)
    config = dict(agent.configuration or {})
    hb = dict(config.get("heartbeat", {}))

    changes = []

    # Apply each provided field
    if "enabled" in params:
        hb["enabled"] = bool(params["enabled"])
        changes.append(f"enabled -> {hb['enabled']}")

    if "interval_minutes" in params:
        minutes = max(5, min(1440, int(params["interval_minutes"])))
        hb["interval_minutes"] = minutes
        changes.append(f"interval -> {minutes}m")

    if "prompt" in params:
        hb["prompt"] = str(params["prompt"])[:2000]
        changes.append("prompt updated")

    if "auto_act" in params:
        hb["auto_act"] = bool(params["auto_act"])
        changes.append(f"auto_act -> {hb['auto_act']}")

    if "active_hours_start" in params:
        hb["active_hours_start"] = str(params["active_hours_start"])
        changes.append(f"active_hours_start -> {hb['active_hours_start']}")

    if "active_hours_end" in params:
        hb["active_hours_end"] = str(params["active_hours_end"])
        changes.append(f"active_hours_end -> {hb['active_hours_end']}")

    if "proactive_level" in params:
        level = str(params["proactive_level"])
        if level in ("silent", "notify", "act_notify", "autonomous"):
            hb["proactive_level"] = level
            changes.append(f"proactive_level -> {level}")

    if "notification_channel" in params:
        hb["notification_channel"] = str(params["notification_channel"])
        changes.append(f"notification_channel -> {hb['notification_channel']}")

    if "checklist" in params:
        hb["checklist"] = str(params["checklist"])[:5000]
        changes.append("checklist updated")

    if not changes:
        from modules.tools.discovery.action_registry import nothing_changed

        return {
            "success": False,
            "error": nothing_changed("platform_configure_agent_heartbeat", "agent_id", "agent_name"),
            "current_heartbeat": hb,
            "agent_id": agent.id,
        }

    # Write back (immutable: new dict, not mutation)
    config["heartbeat"] = hb
    agent.configuration = config
    flag_modified(agent, "configuration")
    db.flush()

    logger.info(
        "[PlatformExecutor] Configured heartbeat for agent '%s' (id=%d): %s",
        agent.name, agent.id, ", ".join(changes),
    )

    # Note: heartbeat schedule will be picked up on next service reload.
    # Live rescheduling requires the HeartbeatService singleton (future enhancement).

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "heartbeat": hb,
        "changes": changes,
        "message": f"Heartbeat for agent '{agent.name}' configured: {', '.join(changes)}",
    }
