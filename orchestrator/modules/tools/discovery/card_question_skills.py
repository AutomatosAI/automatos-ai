"""PRD-256 P256-FIX-RVW-14 (Decision D1 amended, FX-010): a workspace's plugin or skill
taken from every agent waits for the owner's click, and its card says from whom.

The fix-wave review: three tools changed every agent's settings with no card from the
owner's chat. ``platform_uninstall_plugin`` unassigns the plugin from every agent in the
workspace, ``platform_delete_workspace_skill`` drops the skill's agent_skills rows and
``platform_update_skill`` forks a marketplace skill and moves its agents onto the fork.
They are owner-only now, and their card names the plugin or skill by its id and the
agents it is taken from or moved for (this workspace's agents only).

Before the ask is raised the subject is bound to its id (``agent_binding``'s pattern):
the plugin's id as the database holds it, the skill's as a number, so the click runs on
the row the card showed. A plugin not enabled here, or a skill the tool would refuse
(another workspace's, a marketplace skill to delete, one not enabled to edit), is
refused before any grant: nothing is asked about what the call cannot act on (F091).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from modules.tools.discovery.card_question_text import said_line, value_line

PLUGIN_ID, SKILL_ID = "plugin_id", "skill_id"
UNINSTALL_PLUGIN = "platform_uninstall_plugin"
DELETE_SKILL = "platform_delete_workspace_skill"
UPDATE_SKILL = "platform_update_skill"
# The owner-only actions whose subject is a plugin or a skill, and the param that names it.
BINDS = {UNINSTALL_PLUGIN: PLUGIN_ID, DELETE_SKILL: SKILL_ID, UPDATE_SKILL: SKILL_ID}

PLUGIN_LINE = "plugin: '{name}' ({slug}, plugin {id}) is turned off for this workspace"
DELETED_FOR_GOOD = "skill: '{name}' (skill #{id}) is deleted for good"
FORKED = "skill: '{name}' (skill #{id}), a marketplace skill: your edit makes this workspace's own copy"
EDITED = "skill: '{name}' (skill #{id}), this workspace's own: edited in place"
TAKEN_FROM = "taken from"
MOVED_FOR = "moved onto the copy for"
USED_BY = "used by"
NO_AGENT = "no agent"
NEW_CONTENT = "new content"
OVERRIDES_THE_SCAN = "saved over the security scanner's high-severity findings (acknowledge_warnings)"
MAX_AGENTS_NAMED = 10
MORE_AGENTS = ", and {count} more"
NOT_HERE = ("No {noun} {said} the call can act on in this workspace: nothing was asked or done. "
            "Look it up first ({lookup}), then call {action} with its id.")
LOOKUPS = {PLUGIN_ID: "platform_list_workspace_plugins", SKILL_ID: "platform_list_workspace_skills"}
NOUNS = {PLUGIN_ID: "plugin", SKILL_ID: "skill"}


def bound_to_the_subject(db: Any, workspace_id: Any, action: str,
                         params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call with its plugin or skill id as the row's own, the refusal when it names
    none the tool would act on). Any other call is as it is; a skill or plugin given to
    (or taken from) one agent is bound by ``assigned_subjects`` (P256-FIX-RVW-17)."""
    from modules.tools.discovery.assigned_subjects import BINDS as ASSIGNED, bound_to_the_assigned

    if action in ASSIGNED:
        return bound_to_the_assigned(db, workspace_id, action, params)
    key = BINDS.get(action)
    if key is None:
        return params, None
    row = _plugin(db, workspace_id, params.get(key)) if key == PLUGIN_ID else _skill(db, workspace_id, action, params)
    if row is None:
        error = NOT_HERE.format(noun=NOUNS[key], said=params.get(key), lookup=LOOKUPS[key], action=action)
        return params, {"success": False, "error": error}
    return {**params, key: str(row.id) if key == PLUGIN_ID else row.id}, None


def uninstall_plugin_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The plugin turned off, and every agent of this workspace it is taken from."""
    from core.models.core import Agent
    from core.models.marketplace_plugins import AgentAssignedPlugin

    plugin = _plugin(db, workspace_id, params.get(PLUGIN_ID))
    if plugin is None:
        return []
    agents = (db.query(Agent.id, Agent.name).join(AgentAssignedPlugin, AgentAssignedPlugin.agent_id == Agent.id)
              .filter(AgentAssignedPlugin.plugin_id == plugin.id, Agent.workspace_id == workspace_id)
              .order_by(Agent.id).all())
    return [value_line(PLUGIN_LINE.format(name=plugin.name, slug=plugin.slug, id=plugin.id)),
            _agents_line(TAKEN_FROM, agents)]


def delete_skill_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The skill deleted for good, and every agent of this workspace it is taken from."""
    skill = _skill(db, workspace_id, action, params)
    if skill is None:
        return []
    return [value_line(DELETED_FOR_GOOD.format(name=skill.name, id=skill.id)),
            _agents_line(TAKEN_FROM, _skill_agents(db, workspace_id, skill.id))]


def update_skill_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The skill edited (a marketplace one forked), the agents moved onto the fork or
    using it, the new content's start, and when the call overrides the scanner's warnings."""
    skill = _skill(db, workspace_id, action, params)
    if skill is None:
        return []
    forks = skill.workspace_id is None
    head = (FORKED if forks else EDITED).format(name=skill.name, id=skill.id)
    agents = _skill_agents(db, workspace_id, skill.id)
    lines = [value_line(head), _agents_line(MOVED_FOR if forks else USED_BY, agents),
             said_line(NEW_CONTENT, params.get("content"))]
    return [*lines, value_line(OVERRIDES_THE_SCAN)] if params.get("acknowledge_warnings") else lines


def _plugin(db: Any, workspace_id: Any, said: Any) -> Optional[Any]:
    """The plugin with this id enabled for this workspace, or None."""
    from core.models.marketplace_plugins import MarketplacePlugin, WorkspaceEnabledPlugin

    try:
        plugin_id = said if isinstance(said, UUID) else UUID(str(said))
    except (TypeError, ValueError, AttributeError):
        return None
    return (db.query(MarketplacePlugin).join(WorkspaceEnabledPlugin, WorkspaceEnabledPlugin.plugin_id == MarketplacePlugin.id)
            .filter(WorkspaceEnabledPlugin.workspace_id == workspace_id, MarketplacePlugin.id == plugin_id).first())


def _skill(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> Optional[Any]:
    """The skill the tool would act on: this workspace's own for a delete; for an edit,
    this workspace's own or a marketplace skill enabled here (handlers_skills' rules)."""
    from core.models.core import Skill

    try:
        skill_id = int(params.get(SKILL_ID))
    except (TypeError, ValueError):
        return None
    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if skill is None or (action == UPDATE_SKILL and not skill.is_active):
        return None
    if skill.workspace_id is not None:
        return skill if str(skill.workspace_id) == str(workspace_id) else None
    return skill if action == UPDATE_SKILL and _enabled_here(db, workspace_id, skill.id) else None


def _enabled_here(db: Any, workspace_id: Any, skill_id: int) -> bool:
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    return db.query(WorkspaceEnabledSkill).filter(WorkspaceEnabledSkill.workspace_id == workspace_id,
                                                  WorkspaceEnabledSkill.skill_id == skill_id).first() is not None


def _skill_agents(db: Any, workspace_id: Any, skill_id: int) -> List[Any]:
    """This workspace's agents that hold the skill, by id."""
    from core.models.core import Agent, agent_skills

    return (db.query(Agent.id, Agent.name).join(agent_skills, agent_skills.c.agent_id == Agent.id)
            .filter(agent_skills.c.skill_id == skill_id, Agent.workspace_id == workspace_id)
            .order_by(Agent.id).all())


def _agents_line(label: str, agents: List[Any]) -> str:
    """"- taken from: 'GREEN BUYER' (agent #44), 'ROASTER' (agent #45)", or no agent."""
    named = ", ".join(f"'{agent.name}' (agent #{agent.id})" for agent in agents[:MAX_AGENTS_NAMED])
    extra = len(agents) - MAX_AGENTS_NAMED
    more = MORE_AGENTS.format(count=extra) if extra > 0 else ""
    return value_line(f"{label}: {named or NO_AGENT}{more}")


__all__ = ["BINDS", "bound_to_the_subject", "delete_skill_lines", "uninstall_plugin_lines", "update_skill_lines"]
