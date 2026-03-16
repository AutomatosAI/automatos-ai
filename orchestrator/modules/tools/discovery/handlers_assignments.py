"""Agent assignment handlers for PlatformActionExecutor (PRD-71) — assign tool/skill/plugin, configure heartbeat."""

import logging
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


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
        return None, {"success": False, "error": "Agent not found in this workspace"}

    return agent, None


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
                "message": f"Tool '{app_name}' is already assigned to agent '{agent.name}'.",
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
            "message": f"Tool '{app_name}' re-activated for agent '{agent.name}'.",
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
        "message": f"Tool '{app_name}' assigned to agent '{agent.name}'.",
    }


async def assign_skill_to_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a skill to an agent via the agent_skills M2M table."""
    from core.models.core import Skill, agent_skills

    agent, err = resolve_agent(db, workspace_id, params)
    if err:
        return err

    skill_id = params.get("skill_id")
    skill_name = params.get("skill_name")

    if not skill_id and not skill_name:
        return {"success": False, "error": "Provide skill_id or skill_name"}

    # Resolve skill
    query = db.query(Skill)
    if skill_id:
        query = query.filter(Skill.id == skill_id)
    else:
        query = query.filter(Skill.name.ilike(f"%{skill_name}%"))

    skill = query.first()
    if not skill:
        return {"success": False, "error": "Skill not found"}

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
        return {
            "success": True,
            "message": "No changes specified",
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
