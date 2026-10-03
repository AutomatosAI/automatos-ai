"""Which playbook Auto's tools act on when they are given its name (F261, night 7b).

Two of the owner's playbooks are called "New Cafe Onboarding": 102 has the Analyst on
both steps, and 103 has no agent on its steps. Asked to run "the one that actually has
an agent on its steps", Auto started 103 twice (#0183, #0184, each failing in 0.3 s)
while saying it had picked the one with the Analyst, and it read one of the two and
told the owner neither had an agent. Each tool looked a name up as
``name ILIKE '%…%'`` and took whichever row came first.

``finds_the_playbook`` reads the name before the tool runs:
- a name one playbook has (trimmed, any case) is that playbook; with none, a name only
  one playbook's name contains is that one;
- a name several playbooks have: a run takes the only one whose every step has an agent
  (a document step needs none), and the answer says which and why. A read gives every
  one of them. Anything else is refused, listing each with its steps' agents;
- a playbook_id that is a name ("Weekly Instagram posts") is read as the name.

A read names each step's agent, not only its id.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

RUNS, READS, CHANGES = "run", "read", "change"
DONE = {RUNS: "started", CHANGES: "changed"}
SEVERAL = ("{count} playbooks match '{name}': {listed}. Nothing was {done}. Call this again with the "
           "playbook_id of the one the owner means.")
CHOSEN = ("{count} playbooks are called '{name}'. Playbook {id} ran: it is the only one with an agent on every "
          "step ({others}).")
EVERY_ONE = "{count} playbooks are called '{name}'; each is in 'playbooks', with its steps' agents."
NO_AGENT = "step {n} has no agent"


def finds_the_playbook(does: str) -> Callable[[Handler], Handler]:
    """Let a playbook tool find a playbook by name without guessing (see the module)."""
    def decorate(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            params = _name_in_id(params or {})
            name = params.get("playbook_name")
            found = playbooks_called(db, workspace_id, name) if name and not params.get("playbook_id") else []
            if len(found) > 1:
                return await _several(db, workspace_id, handler, params, found, does)
            if found:
                params = {**params, "playbook_id": found[0].id}
            out = await handler(db, workspace_id, params)
            return _named_agents(db, out) if does == READS else out
        return wrapped
    return decorate


async def _several(db: Session, workspace_id: Any, handler: Handler, params: Dict[str, Any], found: List[Any],
                   does: str) -> Dict[str, Any]:
    """A name several playbooks match: a read gives each; a run takes the only namesake
    whose every step has an agent; anything else is refused, listing them."""
    name = str(params.get("playbook_name")).strip()
    if does == READS:
        return await _every_one(db, workspace_id, handler, params, found)
    namesakes = all(str(p.name).strip().lower() == name.lower() for p in found)
    runnable = [p for p in found if staffed_steps(p)] if does == RUNS and namesakes else []
    if len(runnable) != 1:
        return {"success": False, "error": SEVERAL.format(
            count=len(found), name=name, listed=_listed(db, found), done=DONE[does])}
    out = await handler(db, workspace_id, {**params, "playbook_id": runnable[0].id})
    return _chosen(db, out, runnable[0], found, name)


def _name_in_id(params: Dict[str, Any]) -> Dict[str, Any]:
    """A playbook_id that is a name, not a number, read as playbook_name."""
    said = params.get("playbook_id")
    if not isinstance(said, str) or not said.strip() or said.strip().isdigit():
        return params
    rest = {k: v for k, v in params.items() if k != "playbook_id"}
    return rest if rest.get("playbook_name") else {**rest, "playbook_name": said.strip()}


def playbooks_called(db: Session, workspace_id: Any, name: Any) -> List[Any]:
    """The workspace's playbooks with this name (trimmed, any case), oldest first; with
    none, those whose name contains it."""
    from core.models.core import WorkflowTemplate

    wanted = str(name).strip()
    mine = db.query(WorkflowTemplate).filter(WorkflowTemplate.workspace_id == workspace_id)
    exact = mine.filter(func.lower(func.trim(WorkflowTemplate.name)) == wanted.lower()).order_by(
        WorkflowTemplate.id).all()
    return exact or mine.filter(WorkflowTemplate.name.ilike(f"%{wanted}%")).order_by(WorkflowTemplate.id).all()


def staffed_steps(playbook: Any) -> bool:
    """Whether every step of ``playbook`` can run: each has an agent, or needs none."""
    steps = [s for s in (playbook.steps or []) if isinstance(s, dict)]
    return bool(steps) and not _unstaffed(steps)


def _unstaffed(steps: List[Dict[str, Any]]) -> List[int]:
    """The 1-based numbers of the steps that need an agent and have none."""
    from core.models.core import PLAYBOOK_DOCUMENT_STEP

    return [n for n, s in enumerate(steps, start=1)
            if not s.get("agent_id") and s.get("type", "agent") != PLAYBOOK_DOCUMENT_STEP]


def _agents(db: Session, ids: List[Any]) -> Dict[Any, str]:
    from core.models.core import Agent

    wanted = [i for i in ids if i]
    return dict(db.query(Agent.id, Agent.name).filter(Agent.id.in_(wanted)).all()) if wanted else {}


def _listed(db: Session, playbooks: List[Any]) -> str:
    """Each playbook's id and its steps' agents, or the steps that have none."""
    names = _agents(db, [s.get("agent_id") for p in playbooks for s in (p.steps or []) if isinstance(s, dict)])
    return "; ".join(f"{p.id} ({_steps_said(p, names)})" for p in playbooks)


def _steps_said(playbook: Any, names: Dict[Any, str]) -> str:
    steps = [s for s in (playbook.steps or []) if isinstance(s, dict)]
    missing = _unstaffed(steps)
    if not steps:
        return "no steps"
    if missing:
        return ", ".join(NO_AGENT.format(n=n) for n in missing)
    agents = [names.get(s.get("agent_id"), f"agent {s.get('agent_id')}") for s in steps if s.get("agent_id")]
    return f"{len(steps)} steps, agents: {', '.join(agents)}" if agents else f"{len(steps)} steps"


def _chosen(db: Session, out: Dict[str, Any], chosen: Any, found: List[Any], name: str) -> Dict[str, Any]:
    if not (isinstance(out, dict) and out.get("success")):
        return out
    others = _listed(db, [p for p in found if p.id != chosen.id])
    return {**out, "chosen_because": CHOSEN.format(count=len(found), name=name, id=chosen.id, others=others)}


async def _every_one(db: Session, workspace_id: Any, handler: Handler, params: Dict[str, Any],
                     found: List[Any]) -> Dict[str, Any]:
    """A read of a name several playbooks have: each of them."""
    read = [_named_agents(db, await handler(db, workspace_id, {**params, "playbook_id": p.id})) for p in found]
    playbooks = [out["playbook"] for out in read if out.get("success") and out.get("playbook")]
    return {"success": bool(playbooks), "playbooks": playbooks,
            "namesakes": EVERY_ONE.format(count=len(found), name=params.get("playbook_name"))}


def _named_agents(db: Session, out: Dict[str, Any]) -> Dict[str, Any]:
    """A read's steps name their agent beside its id."""
    playbook = out.get("playbook") if isinstance(out, dict) else None
    steps = playbook.get("steps") if isinstance(playbook, dict) else None
    if not isinstance(steps, list):
        return out
    names = _agents(db, [s.get("agent_id") for s in steps if isinstance(s, dict)])
    named = [{**s, "agent": names.get(s.get("agent_id")) if s.get("agent_id") else None} if isinstance(s, dict) else s
             for s in steps]
    return {**out, "playbook": {**playbook, "steps": named}}


__all__ = ["CHANGES", "READS", "RUNS", "finds_the_playbook", "playbooks_called", "staffed_steps"]
