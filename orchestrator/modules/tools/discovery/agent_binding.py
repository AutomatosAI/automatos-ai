"""PRD-256 FX-010: the owner's click runs on the agent its card showed.

An owner-only agent change may name its agent by ``agent_name``, which the handlers read
as any agent whose name contains it (``handlers_assignments.resolve_agent``,
``handlers_agents.delete_agent``), first row wins. Read once for the card and again at
the click, "market" could show MARKET-MANAGER and delete Market Research. So before the
ask is raised the name is bound to one agent's id (the way FX-009 binds a mission): the
one whose whole name it is, else the one whose name contains it. Two or more such agents
are refused, naming them, and none is refused with the workspace's roster: nothing is
asked about an agent the call cannot name (F091).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# The owner-only actions that take agent_id or agent_name (owner_only.OWNER_ONLY_ACTIONS).
BINDS = frozenset({
    "platform_update_agent", "platform_assign_tool_to_agent", "platform_unassign_tool_from_agent",
    "platform_configure_agent_heartbeat", "platform_delete_agent", "platform_assign_skill_to_agent",
    "platform_unassign_skill_from_agent", "platform_assign_plugin_to_agent",
})
AGENT_ID, AGENT_NAME = "agent_id", "agent_name"
MAX_NAMED = 10
AMBIGUOUS = ("{count} agents match agent_name '{said}' in this workspace: {named}. Nothing was asked or done: "
             "call {action} again with the agent_id of the one meant.")


def bound_to_the_agent(db: Any, workspace_id: Any, action: str,
                       params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call with ``agent_id`` bound to the one agent its ``agent_name`` names, the
    refusal when it names several or none). A call with an agent_id, or no name, is as it is."""
    said = params.get(AGENT_NAME)
    if action not in BINDS or params.get(AGENT_ID) not in (None, "") or not isinstance(said, str) or not said.strip():
        return params, None
    matches = _named(db, workspace_id, said)
    if len(matches) == 1:
        return {**params, AGENT_ID: matches[0].id}, None
    if not matches:
        from modules.tools.discovery.handlers_assignments import resolve_agent

        return params, resolve_agent(db, workspace_id, params)[1]  # the roster
    named = ", ".join(f"{agent.id}:{agent.name}" for agent in matches[:MAX_NAMED])
    return params, {"success": False, "error": AMBIGUOUS.format(count=len(matches), said=said, named=named,
                                                                   action=action)}


def _named(db: Any, workspace_id: Any, said: str) -> List[Any]:
    """This workspace's agents whose whole name is ``said`` (any case), else those whose
    name contains it, by id."""
    from core.models import Agent

    wanted = said.strip()
    agents = (db.query(Agent.id, Agent.name)
              .filter(Agent.workspace_id == workspace_id, Agent.name.ilike(f"%{wanted}%"))
              .order_by(Agent.id).all())
    whole = [agent for agent in agents if (agent.name or "").strip().lower() == wanted.lower()]
    return whole or agents


__all__ = ["BINDS", "bound_to_the_agent"]
