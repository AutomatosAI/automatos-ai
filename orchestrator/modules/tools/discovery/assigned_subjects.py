"""PRD-256 P256-FIX-RVW-17 (FX-008, FX-010): a skill or plugin given to an agent, or a skill
taken from one, is bound to one row before its card, so the card and the click name the same.

The second fix-wave review: the card read ``skill_name`` when the call gave one, while the
handler (``handlers_assignments._pick_installed_skill``) takes ``skill_id`` first, then an
installed skill's whole name, then the one installed name containing it. {skill_id: 5,
skill_name: 'menu-writer'} showed 'menu-writer' and gave skill 5; 'sourcing' showed
'sourcing' and gave 'Sourcing Advanced'. The plugin card read ``plugin_slug`` first, the
handler ``plugin_id``.

Now the ask resolves the subject the handler's way (``agent_binding``'s pattern, with the
handler's own picker): a skill given is one of this workspace's installed, enabled skills;
a skill taken is one the agent holds; a plugin is one enabled for this workspace. The call
the card asks about names it by its id alone, the card reads that row, and the click runs
on it. None, or a partial name several skills carry, is refused before any card (F091).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

SKILL_ID, SKILL_NAME = "skill_id", "skill_name"
PLUGIN_ID, PLUGIN_SLUG = "plugin_id", "plugin_slug"
ASSIGN_SKILL = "platform_assign_skill_to_agent"
UNASSIGN_SKILL = "platform_unassign_skill_from_agent"
ASSIGN_PLUGIN = "platform_assign_plugin_to_agent"
BINDS = frozenset({ASSIGN_SKILL, UNASSIGN_SKILL, ASSIGN_PLUGIN})
MAX_NAMED = 12
NONE_YET = "none"

NO_SKILL = ("No skill {said!r} is installed and enabled in this workspace, so nothing was asked or done. "
            "Installed here: {named}. Call {action} again with the skill_id of the one meant "
            "(platform_list_workspace_skills lists them).")
NOT_HELD = ("Agent '{agent}' holds no skill {said!r}, so nothing was asked or done. Its skills: {named}. "
            "Call {action} again with the skill_id of the one meant.")
NO_PLUGIN = ("No plugin {said!r} is enabled for this workspace, so nothing was asked or done. "
             "Call {action} again with the plugin_id of the one meant (platform_list_workspace_plugins lists them).")
NAME_THE_SKILL = "Provide skill_id or skill_name"
NAME_THE_PLUGIN = "Provide plugin_id or plugin_slug"


def bound_to_the_assigned(db: Any, workspace_id: Any, action: str,
                          params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call naming its skill or plugin by the row's own id alone, the refusal when it
    names none the handler would act on, or several). Any other call is as it is."""
    if action not in BINDS:
        return params, None
    if action == ASSIGN_PLUGIN:
        row, refusal = plugin_of(db, workspace_id, params)
        key, said_key = PLUGIN_ID, PLUGIN_SLUG
    else:
        row, refusal = skill_of(db, workspace_id, action, params)
        key, said_key = SKILL_ID, SKILL_NAME
    if row is None:
        return params, refusal
    rest = {name: value for name, value in params.items() if name != said_key}
    return {**rest, key: str(row.id) if key == PLUGIN_ID else row.id}, None


def skill_of(db: Any, workspace_id: Any, action: str,
             params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """(the skill the handler acts on, None), or (None, why not): picked by the handler's
    rule from the workspace's installed skills to give, from the agent's own to take."""
    from modules.tools.discovery.handlers_assignments import _installed_skills, _pick_installed_skill, resolve_agent

    said_id, said_name = params.get(SKILL_ID), params.get(SKILL_NAME)
    if said_id in (None, "") and said_name in (None, ""):
        return None, {"success": False, "error": NAME_THE_SKILL}
    agent = None
    if action == UNASSIGN_SKILL:
        agent, refused = resolve_agent(db, workspace_id, params)
        if agent is None:
            return None, refused
    pool = _held(db, agent) if agent is not None else _installed_skills(db, workspace_id)
    skill, refusal = _pick_installed_skill(pool, said_id, said_name)
    if skill is not None or refusal:
        return skill, refusal
    said = said_id if said_id not in (None, "") else said_name
    named = ", ".join(f"{row.id}:{row.name}" for row in pool[:MAX_NAMED]) or NONE_YET
    error = (NOT_HELD.format(agent=agent.name, said=said, named=named, action=action) if agent is not None
             else NO_SKILL.format(said=said, named=named, action=action))
    return None, {"success": False, "error": error}


def plugin_of(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """(the plugin enabled for this workspace the call names, by ``plugin_id`` first as the
    handler reads it, else its slug; None), or (None, why not)."""
    from core.models.marketplace_plugins import MarketplacePlugin, WorkspaceEnabledPlugin

    said_id, said_slug = params.get(PLUGIN_ID), params.get(PLUGIN_SLUG)
    if said_id in (None, "") and said_slug in (None, ""):
        return None, {"success": False, "error": NAME_THE_PLUGIN}
    query = (db.query(MarketplacePlugin)
             .join(WorkspaceEnabledPlugin, WorkspaceEnabledPlugin.plugin_id == MarketplacePlugin.id)
             .filter(WorkspaceEnabledPlugin.workspace_id == workspace_id))
    if said_id not in (None, ""):
        plugin_id = _uuid(said_id)
        plugin = query.filter(MarketplacePlugin.id == plugin_id).first() if plugin_id is not None else None
    else:
        plugin = query.filter(MarketplacePlugin.slug == str(said_slug)).first()
    if plugin is not None:
        return plugin, None
    said = said_id if said_id not in (None, "") else said_slug
    return None, {"success": False, "error": NO_PLUGIN.format(said=said, action=ASSIGN_PLUGIN)}


def _held(db: Any, agent: Any) -> List[Any]:
    """The skills the agent holds, by name."""
    from core.models.core import Skill, agent_skills

    return (db.query(Skill).join(agent_skills, agent_skills.c.skill_id == Skill.id)
            .filter(agent_skills.c.agent_id == agent.id).order_by(Skill.name).all())


def _uuid(said: Any) -> Optional[UUID]:
    if isinstance(said, UUID):
        return said
    try:
        return UUID(str(said))
    except (TypeError, ValueError, AttributeError):
        return None


__all__ = ["BINDS", "bound_to_the_assigned", "plugin_of", "skill_of"]
