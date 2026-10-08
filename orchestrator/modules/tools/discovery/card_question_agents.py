"""PRD-256 FX-010: what an approval card says for an agent-setting change.

Night 12 changed GREEN BUYER's heartbeat, deleted MARKET-MANAGER on one word and gave
agents eleven skills, with no card. Those calls are owner-only now (Decision D1, amended
8 Oct), and their card names the agent and the change: the heartbeat's fields 'from → to'
(read from the agent's own row), the agent's skills or plugins before and after, or that
the agent is deleted for good. The agent is looked up in the caller's workspace only; an
agent that is not there gives no lines, and the handler refuses the call itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_text import change_line, value_line

AGENT_LINE = "agent: '{name}' (agent #{id})"
DELETED_FOR_GOOD = "agent: '{name}' (agent #{id}) is deleted for good: this cannot be undone"
HEARTBEAT = "heartbeat"
# platform_configure_agent_heartbeat: param → what the owner calls it (the heartbeat holds it under the param).
HEARTBEAT_FIELDS = (
    ("enabled", "heartbeat on"), ("interval_minutes", "every (minutes)"), ("prompt", "checks"),
    ("auto_act", "acts on its findings"), ("active_hours_start", "active from"),
    ("active_hours_end", "active until"), ("proactive_level", "proactive level"),
    ("notification_channel", "reports to"), ("checklist", "checklist"),
)
SKILLS_FIELD = "skills"
PLUGINS_FIELD = "plugins"
ASSIGNS = "platform_assign_"


def _agent(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[Any]:
    from modules.tools.discovery.handlers_assignments import resolve_agent

    return resolve_agent(db, workspace_id, params)[0]


def heartbeat_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The agent, then each heartbeat field the call changes, 'field: from → to'."""
    agent = _agent(db, workspace_id, params)
    if agent is None:
        return []
    beat = (agent.configuration or {}).get(HEARTBEAT) or {}
    changes = [change_line(label, beat.get(param), params[param])
               for param, label in HEARTBEAT_FIELDS if param in params]
    return [value_line(AGENT_LINE.format(name=agent.name, id=agent.id)), *changes]


def delete_agent_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """Which agent goes, and that it cannot be undone."""
    agent = _agent(db, workspace_id, params)
    return [value_line(DELETED_FOR_GOOD.format(name=agent.name, id=agent.id))] if agent is not None else []


def skill_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The agent's skills before and after the call: 'skills: menu-writer → menu-writer, sourcing'."""
    from core.models.core import Skill, agent_skills

    agent = _agent(db, workspace_id, params)
    named = _skill_named(db, workspace_id, params)
    if agent is None or named is None:
        return []
    rows = (db.query(Skill.name).join(agent_skills, agent_skills.c.skill_id == Skill.id)
            .filter(agent_skills.c.agent_id == agent.id).all())
    return [value_line(AGENT_LINE.format(name=agent.name, id=agent.id)),
            _before_and_after(SKILLS_FIELD, [row.name for row in rows], named, action)]


def _skill_named(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[str]:
    """The skill the call names: its name as said, or the name of the workspace's (or a
    marketplace) skill with that id."""
    from core.models.core import Skill

    if params.get("skill_name"):
        return str(params["skill_name"])
    try:
        skill_id = int(params.get("skill_id"))
    except (TypeError, ValueError):
        return None
    skill = db.query(Skill.name).filter(
        Skill.id == skill_id, (Skill.workspace_id.is_(None)) | (Skill.workspace_id == workspace_id)).first()
    return skill.name if skill is not None else None


def plugin_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The agent's plugins before and after the call."""
    from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin

    agent = _agent(db, workspace_id, params)
    named = params.get("plugin_slug") or params.get("plugin_id")
    if agent is None or not named:
        return []
    rows = (db.query(MarketplacePlugin.slug).join(AgentAssignedPlugin, AgentAssignedPlugin.plugin_id == MarketplacePlugin.id)
            .filter(AgentAssignedPlugin.agent_id == agent.id).all())
    return [value_line(AGENT_LINE.format(name=agent.name, id=agent.id)),
            _before_and_after(PLUGINS_FIELD, [row.slug for row in rows], str(named), action)]


def _before_and_after(field: str, now: List[str], named: str, action: str) -> str:
    """'skills: a → a, b' for an assign; 'skills: a, b → a' for an unassign."""
    now = sorted({str(name) for name in now if name})
    if action.startswith(ASSIGNS):
        after = sorted({*now, named})
    else:
        after = [name for name in now if name.lower() != named.lower()]
    return change_line(field, now, after)


__all__ = ["delete_agent_lines", "heartbeat_lines", "plugin_lines", "skill_lines"]
