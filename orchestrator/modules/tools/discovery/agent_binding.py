"""PRD-256 FX-010: the owner's click runs on the agent its card showed.

An owner-only agent change may name its agent by ``agent_name``, which the handlers read
as any agent whose name contains it (``handlers_assignments.resolve_agent``,
``handlers_agents.delete_agent``), first row wins. Read once for the card and again at
the click, "market" could show MARKET-MANAGER and delete Market Research. So before the
ask is raised the name is bound to one agent's id (the way FX-009 binds a mission): the
one whose whole name it is, else the one whose name contains it. Two or more such agents
are refused, naming them, and none is refused with the workspace's roster: nothing is
asked about an agent the call cannot name (F091).

PRD-256 N17-8 (F440): the same rule (``one_named``) reads a name in the ungated write resolvers
(``handlers_agents._resolve_agent``, ``handlers_assignments.resolve_agent``), which took the first
row whose name contained it. platform_get_agent by name reads the agent the board gives work to
(``the_agent_named``) and lists every agent carrying the whole name: night 16's 'WRITER' read an
inactive namesake while the board's WRITER had just finished two cards.
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
    agent_id, refusal = one_named(db, workspace_id, params[AGENT_NAME], action)
    if agent_id is not None:
        return {**params, AGENT_ID: agent_id}, None
    if refusal is None:
        from modules.tools.discovery.handlers_assignments import resolve_agent

        return params, resolve_agent(db, workspace_id, params)[1]  # the roster
    return params, refusal


def one_named(db: Any, workspace_id: Any, said: str,
              action: str = "the same tool") -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """(the id of the one agent ``said`` names, None); (None, the refusal naming each by id) when
    several do; (None, None) when none does. Its whole name, else the one name containing it."""
    matches = named_agents(db, workspace_id, said)
    if len(matches) == 1:
        return matches[0].id, None
    if not matches:
        return None, None
    named = ", ".join(f"{agent.id}:{agent.name}" for agent in matches[:MAX_NAMED])
    return None, {"success": False, "error": AMBIGUOUS.format(count=len(matches), said=said, named=named,
                                                                 action=action)}


def the_agent_named(db: Any, workspace_id: Any, said: str) -> Tuple[Optional[int], List[str]]:
    """(the id of the agent a read by name means, every namesake): the board's agent (the active
    agent whose whole name it is, ``agent_refs.active_named``; the lowest id of several), else the
    first ``named_agents`` gives, else None; then, when two or more agents of any status carry the
    whole name, each as "id · name · job title · status" by id. A name contained in another agent's
    is never listed."""
    from modules.tools.discovery.agent_refs import CANDIDATE_SEP, active_named, candidate

    active = active_named(db, workspace_id, said)
    named = named_agents(db, workspace_id, said)
    chosen = active[0].id if active else named[0].id if named else None
    whole = [agent for agent in named if _is_whole(agent, said)]
    same_name = [CANDIDATE_SEP.join((candidate(agent), str(agent.status))) for agent in whole]
    return chosen, same_name if len(same_name) > 1 else []


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


def named_agents(db: Any, workspace_id: Any, said: str) -> List[Any]:
    """This workspace's agents whose whole name is ``said`` (any case), else those whose
    name contains it, by id; each with what ``agent_refs.candidate`` names it by."""
    from core.models import Agent

    agents = (db.query(Agent.id, Agent.name, Agent.status, Agent.job_title, Agent.team, Agent.agent_type)
              .filter(Agent.workspace_id == workspace_id, Agent.name.ilike(f"%{said.strip()}%"))
              .order_by(Agent.id).all())
    whole = [agent for agent in agents if _is_whole(agent, said)]
    return whole or agents


def _is_whole(agent: Any, said: str) -> bool:
    return (agent.name or "").strip().lower() == said.strip().lower()


__all__ = ["BINDS", "GIVES_THE_CARD", "SCHEDULES_AN_AGENT", "bound_to_the_agent", "name_key", "named_agents",
           "names_the_agent_alone", "one_named", "the_agent_named"]
