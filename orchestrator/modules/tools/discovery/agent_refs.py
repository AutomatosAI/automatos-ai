"""PRD-256 FX-012: the task and playbook tools take the agent's id; a name several agents
carry lists them, and the next call, with the id, goes through.

Night 12 (A1, F390): platform_create_task took only ``assigned_agent_name``,
platform_assign_task only ``agent_name`` and platform_update_task no agent at all, and the
board's resolver refused a name two active agents carry ("Rename or deactivate the
duplicate"). c1 holds ten duplicated names over 26 agents: 39 writes were refused, "Get OPS
to…" fifteen times, and "267, the operations one" too, the id itself sent as the name (A585).

Now each takes ``agent_id`` (267, or "#267") beside the name, and an id wins over a name. An
id outside the workspace, or of an agent switched off, is refused naming it. A name two or
more active agents carry is refused listing each as "id · name · job title", with "call
again with agent_id". A name that is only a number, and no active agent's name, is read as
that id. platform_update_task gives the card to an agent the way platform_assign_task does.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

AGENT_ID, AGENT_NAME = "agent_id", "agent_name"
ACTIVE = "active"
CANDIDATE_SEP, CANDIDATES_SEP = " · ", "; "
_ID_SHAPE = re.compile(r"#?\s*(\d+)")
CLASH = ("Multiple active agents named '{said}', so I can't tell which one is meant: {candidates}. Nothing was "
         "done: call again with agent_id (the number before the name) of the one meant.")
NOT_AN_ID = ("agent_id {said!r} is not an agent's id, so nothing was done: give the number platform_list_agents "
             "shows for the agent (e.g. 267).")
NOT_HERE = ("Agent id {id} is not in this workspace, so nothing was done. platform_list_agents gives this "
            "workspace's agents and their ids.")
SWITCHED_OFF = ("Agent id {id} ({name}) is switched off ({status}), so nothing was done. Switch it on, or call "
                "again with the agent_id of an active agent.")
GIVEN_BEFORE_REFUSED = "{error} The card was given to {agent} before that edit was refused."


def agent_id_property(what: str) -> Dict[str, Any]:
    """A tool's ``agent_id`` field: the agent's id as platform_list_agents shows it."""
    return {"type": ["integer", "string"],
            "description": (f"{what}: the agent's id as platform_list_agents shows it (e.g. 267; '#267' works too). "
                            "Wins over a name; use it when several agents share a name.")}


def agent_id_said(value: Any) -> Optional[int]:
    """The agent id ``value`` names: 267, "267" or "#267"; None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    match = _ID_SHAPE.fullmatch(value.strip()) if isinstance(value, str) else None
    number = int(match.group(1)) if match else 0
    return number or None


def candidate(agent: Any) -> str:
    """One agent as a clash lists it: "267 · OPS · Operations Manager"."""
    role = getattr(agent, "job_title", None) or getattr(agent, "team", None) or getattr(agent, "agent_type", None)
    return CANDIDATE_SEP.join(str(part) for part in (agent.id, agent.name, role) if part)


def candidates(agents: List[Any]) -> str:
    """Each agent of a clash on its own, by id."""
    return CANDIDATES_SEP.join(candidate(agent) for agent in sorted(agents, key=lambda agent: agent.id))


def active_named(db: Any, workspace_id: Any, said: str) -> List[Any]:
    """The workspace's ACTIVE agents whose whole name is ``said`` (any case), by id.

    P224-RVW-4: the same active-only roster AutoBrain's classifier reads, matched in
    Python. ``Agent.name`` has no unique constraint, so 'Atlas' and 'atlas' can coexist;
    an inactive namesake is never in the roster, so it never makes a clash."""
    from core.models import Agent

    active = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.status == ACTIVE).all()
    target = str(said).strip().lower()
    matches = {agent.id: agent for agent in active if (getattr(agent, "name", "") or "").strip().lower() == target}
    return [matches[agent_id] for agent_id in sorted(matches)]


def agent_by_id(db: Any, workspace_id: Any, agent_id: int) -> Tuple[Optional[Any], Optional[str]]:
    """(the active agent ``agent_id`` is in this workspace, None), or (None, why it can't be used)."""
    from core.models import Agent

    rows = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).all()
    agent = next((row for row in rows if row.id == agent_id), None)
    if agent is None:
        return None, NOT_HERE.format(id=agent_id)
    status = getattr(agent, "status", None) or ACTIVE
    if status != ACTIVE:
        return None, SWITCHED_OFF.format(id=agent_id, name=agent.name, status=status)
    return agent, None


def resolve_active_agent(db: Any, workspace_id: Any, ref: Any) -> Tuple[Optional[Any], Optional[str]]:
    """The board writes' agent: ``ref`` is the id ``takes_the_agent_id`` bound (an int), or a name.

    * an id, or a name exactly one active agent carries -> ``(agent, None)``
    * a name no active agent carries -> ``(None, None)``: the caller decides unassigned vs 'not found'
    * a name several carry, or an id that can't be used -> ``(None, refusal)``: the caller refuses
    """
    if isinstance(ref, int) and not isinstance(ref, bool):
        return agent_by_id(db, workspace_id, ref)
    said = str(ref or "")
    matches = active_named(db, workspace_id, said)
    if len(matches) == 1:
        return matches[0], None
    if matches:
        return None, CLASH.format(said=said.strip(), candidates=candidates(matches))
    return None, None


def _bound_id(db: Any, workspace_id: Any, params: Dict[str, Any], name_key: str) -> Tuple[Optional[int], Optional[str]]:
    """(the agent id the call names, the refusal): its agent_id when given, else a name that is
    only a number no active agent is called; (None, None) leaves the name to the handler."""
    said = params.get(AGENT_ID)
    if said not in (None, ""):
        number = agent_id_said(said)
        if number is None:
            return None, NOT_AN_ID.format(said=said)
    else:
        name = params.get(name_key)
        number = agent_id_said(name)
        if number is None or active_named(db, workspace_id, str(name)):
            return None, None
    agent, refusal = agent_by_id(db, workspace_id, number)
    return (agent.id if agent is not None else None), refusal


def takes_the_agent_id(name_key: str) -> Callable[[Handler], Handler]:
    """Wrap a board write whose agent is named under ``name_key``: an ``agent_id`` (or a
    name that is only an id) is checked here and handed on as the int id the handler's
    resolver (``resolve_active_agent``) reads; a refusal runs nothing."""
    def wrap(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            given = dict(params or {})
            agent_id, refusal = _bound_id(db, workspace_id, given, name_key)
            if refusal:
                return {"success": False, "error": refusal}
            if agent_id is not None:
                given = {**given, name_key: agent_id}
            return await handler(db, workspace_id, given)
        return wrapped
    return wrap


def gives_the_card_on_update(update: Handler) -> Handler:
    """Wrap platform_update_task: an agent_id or agent_name gives the card to that agent the
    way platform_assign_task does (an answered card runs again, F309), then any other edit
    runs. A refused assignment edits nothing."""
    @functools.wraps(update)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        given = {key: value for key, value in (params or {}).items() if key not in (AGENT_ID, AGENT_NAME)}
        agent = {key: (params or {})[key] for key in (AGENT_ID, AGENT_NAME)
                 if (params or {}).get(key) not in (None, "")}
        if not agent:
            return await update(db, workspace_id, given)
        from modules.tools.discovery.handlers_board_task_assign import assign_board_task

        private = {key: value for key, value in given.items() if key.startswith("_")}
        assigned = await assign_board_task(db, workspace_id, {**private, "task_id": given.get("task_id"), **agent})
        if assigned.get("success") is not True or not _edits(given):
            return assigned
        return _both(assigned, await update(db, workspace_id, given))
    return wrapped


def _edits(params: Dict[str, Any]) -> bool:
    """True when an update asks for more than its card and an agent."""
    return any(key != "task_id" and not key.startswith("_") for key in params)


def _both(assigned: Dict[str, Any], edited: Dict[str, Any]) -> Dict[str, Any]:
    """One answer for a card given to an agent and then edited."""
    agent = assigned.get("assigned_agent")
    out = {"status": assigned.get("status"), **edited, "assigned_agent": agent}
    if edited.get("success") is True:
        return out
    return {**out, "partial": True, "error": GIVEN_BEFORE_REFUSED.format(error=edited.get("error"), agent=agent)}


__all__ = ["AGENT_ID", "CLASH", "NOT_AN_ID", "NOT_HERE", "SWITCHED_OFF", "active_named", "agent_by_id",
           "agent_id_property", "agent_id_said", "candidate", "candidates", "gives_the_card_on_update",
           "resolve_active_agent", "takes_the_agent_id"]
