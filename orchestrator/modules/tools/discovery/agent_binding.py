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
# P256-FIX-RVW-17: an update that closes a card and gives it to an agent
# (agent_refs.gives_the_card_on_update), read the board's way: an active agent, by id or whole name.
GIVES_THE_CARD = frozenset({"platform_update_task"})
# P256-FIX-RVW-23: a timer for an agent, read the board's way too; it names the agent under its own key.
SCHEDULES_AN_AGENT = frozenset({"platform_schedule_task"})
READ_THE_BOARDS_WAY = GIVES_THE_CARD | SCHEDULES_AN_AGENT
NAMES_AN_AGENT = BINDS | READ_THE_BOARDS_WAY
AGENT_ID, AGENT_NAME = "agent_id", "agent_name"
NAME_KEYS = {"platform_schedule_task": "target_agent_name"}
MAX_NAMED = 10
AMBIGUOUS = ("{count} agents match agent_name '{said}' in this workspace: {named}. Nothing was asked or done: "
             "call {action} again with the agent_id of the one meant.")


def bound_to_the_agent(db: Any, workspace_id: Any, action: str,
                       params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call with ``agent_id`` bound to the one agent its ``agent_name`` names, the
    refusal when it names several or none). A call with an agent_id, or no name, is as it is;
    a card given to an agent is bound by the board's resolver (``_bound_on_the_card``)."""
    if action in READ_THE_BOARDS_WAY:
        return _bound_on_the_card(db, workspace_id, params, name_key(action))
    if not names_the_agent_alone(action, params):
        return params, None
    said = params[AGENT_NAME]
    matches = _named(db, workspace_id, said)
    if len(matches) == 1:
        return {**params, AGENT_ID: matches[0].id}, None
    if not matches:
        from modules.tools.discovery.handlers_assignments import resolve_agent

        return params, resolve_agent(db, workspace_id, params)[1]  # the roster
    named = ", ".join(f"{agent.id}:{agent.name}" for agent in matches[:MAX_NAMED])
    return params, {"success": False, "error": AMBIGUOUS.format(count=len(matches), said=said, named=named,
                                                                   action=action)}


def names_the_agent_alone(action: str, params: Any) -> bool:
    """The call names its agent by ``agent_name`` and no ``agent_id``. No grant is such a
    call's click: its ask binds the name to an id first, and the click runs on that id
    (P256-FIX-RVW-9)."""
    if action not in NAMES_AN_AGENT or not isinstance(params, dict) or params.get(AGENT_ID) not in (None, ""):
        return False
    said = params.get(name_key(action))
    return isinstance(said, str) and bool(said.strip())


def name_key(action: str) -> str:
    """The key ``action`` names its agent under: ``agent_name``, or a timer's ``target_agent_name``."""
    return NAME_KEYS.get(action, AGENT_NAME)


def _bound_on_the_card(db: Any, workspace_id: Any, params: Dict[str, Any],
                       said_key: str = AGENT_NAME) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(the call with the agent it gives the card to, or times, as its ``agent_id`` alone,
    the refusal when it names no active agent, or several): the card names that agent and
    the click runs on it, never on a name read again at the click."""
    if all(params.get(key) in (None, "") for key in (AGENT_ID, said_key)):
        return params, None
    from modules.tools.discovery.agent_refs import board_agent

    agent, refusal = board_agent(db, workspace_id, params, said_key)
    if agent is None:
        return params, {"success": False, "error": refusal}
    rest = {key: value for key, value in params.items() if key != said_key}
    return {**rest, AGENT_ID: agent.id}, None


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


__all__ = ["BINDS", "GIVES_THE_CARD", "SCHEDULES_AN_AGENT", "bound_to_the_agent", "name_key", "names_the_agent_alone"]
