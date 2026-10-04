"""F321 (night 9b): a playbook Auto builds has an agent on every step, and a run no
agent can do is refused before it starts.

Chat 5247a359 (MORNING-REPORT.md, "Playbook built with no agents"): asked for "Monday
green stock", Auto called platform_create_playbook, then platform_add_playbook_step
twice with ``agent_id: null``. The tool took both: its agent_id said "optional — uses
default agent if not set", and there is no default agent. Then
platform_execute_playbook made run #0068 and its card and answered success, and Auto
said "I've also started it for you right now … card #0068". The run failed before its
first step, "steps 1 and 2 have no agent" (F270's net in api.recipe_executor), but the
call behind Auto's "started" had succeeded, so the claim check (action_claims) had
nothing to catch. F270 refused such a run on the Run button's route and in the run
itself, never in Auto's tool.

Now:
- platform_add_playbook_step needs an agent: ``agent_id``, or ``agent_name``, a name
  or job title as the owner said it ("Inventory Watchdog") that exactly one of the
  workspace's active agents answers to. With neither, or a name no single agent
  answers to, no step is added, and the answer lists the agents and says to ask the
  owner which one.
- platform_execute_playbook refuses a playbook with no steps, or with a step that has
  no agent (in F270's words), before any run or card is made. A refused call backs no
  "started" claim.

Both wrap the handlers in handlers_playbooks for Auto's and the agents' tools only
(platform_executor imports them from here).
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from modules.tools.discovery import handlers_playbooks as _steps

logger = logging.getLogger(__name__)

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

ROSTER_SHOWN = 20
NEEDS_AN_AGENT = ("Every playbook step needs an agent to do it: a step with none can't run, so no step was added. "
                  "Call again with the agent_id (or agent_name) of the agent the owner wants; if the owner hasn't "
                  "said which, ask them. This workspace's agents: {roster}.")
NO_SINGLE_AGENT = ("No single agent answers to '{name}' ({why}), so no step was added. Call again with the agent_id "
                   "of the one the owner means; if that isn't clear, ask them. This workspace's agents: {roster}.")
NONE_DOES = "none does"
SEVERAL_DO = "several do: {agents}"
NO_AGENTS = "none is switched on; ask the owner to add an agent or switch one on"
NO_STEPS = ('Playbook {id} "{name}" has no steps, so no run was started. Add its steps, each with an agent, '
            "then run it again.")
# Words that name no agent in particular: "the Inventory Watchdog agent".
_FILLER = frozenset({"the", "a", "an", "our", "my", "agent"})


def _words(text: Any) -> List[str]:
    return [w for w in re.findall(r"[\w'’-]+", str(text or "").lower()) if w not in _FILLER]


def _staff(db: Any, workspace_id: Any) -> List[Any]:
    """The workspace's active agents that can take a step (not Auto), oldest first."""
    from core.models import Agent

    agents = (db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.status == "active")
              .order_by(Agent.id).all())
    return [a for a in agents if not getattr(a, "is_system_agent", False)]


def _roster(staff: List[Any]) -> str:
    named = [f"{a.id}={a.name}" + (f" ({a.job_title})" if getattr(a, "job_title", None) else "")
             for a in staff[:ROSTER_SHOWN]]
    return ", ".join(named) if named else NO_AGENTS


def answering_to(staff: List[Any], said: str) -> List[Any]:
    """The agents a name or job title, as the owner said it, means: those it is
    exactly; else those whose name or title holds it, or all of its words."""
    wanted = " ".join(_words(said))
    if not wanted:
        return []
    names = {a.id: " ".join(_words(f"{a.name} {getattr(a, 'job_title', None) or ''}")) for a in staff}
    exact = [a for a in staff
             if wanted in (" ".join(_words(a.name)), " ".join(_words(getattr(a, "job_title", None))))]
    if exact:
        return exact
    return [a for a in staff if wanted in names[a.id] or set(wanted.split()) <= set(names[a.id].split())]


def _agent_for(db: Any, workspace_id: Any, said: str) -> Tuple[Optional[int], Optional[str]]:
    """The one agent ``said`` means, or why there is none, in words for the model."""
    staff = _staff(db, workspace_id)
    if not said:
        return None, NEEDS_AN_AGENT.format(roster=_roster(staff))
    matches = answering_to(staff, said)
    if len(matches) == 1:
        return matches[0].id, None
    why = SEVERAL_DO.format(agents=", ".join(f"{a.id}={a.name}" for a in matches)) if matches else NONE_DOES
    return None, NO_SINGLE_AGENT.format(name=said, why=why, roster=_roster(staff))


def needs_an_agent(handler: Handler) -> Handler:
    """Wrap platform_add_playbook_step: the step gets an agent, or isn't added."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        given = dict(params or {})
        said = str(given.pop("agent_name", None) or "").strip()
        if given.get("agent_id") not in (None, ""):
            return await handler(db, workspace_id, given)
        agent_id, refusal = _agent_for(db, workspace_id, said)
        if refusal:
            return {"success": False, "error": refusal}
        return await handler(db, workspace_id, {**given, "agent_id": agent_id})
    return wrapped


def _a_number(said: Any) -> bool:
    if isinstance(said, str):
        return said.strip().isdigit()
    return isinstance(said, int) and not isinstance(said, bool)


def _the_playbook(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[Any]:
    """The one playbook a run call names, by number or by name; None when it names
    none, or a name several share (``playbook_lookup`` decides those)."""
    from core.models.core import WorkflowTemplate
    from modules.tools.discovery.playbook_lookup import playbooks_called

    said = params.get("playbook_id")
    if _a_number(said):
        return (db.query(WorkflowTemplate)
                .filter(WorkflowTemplate.id == int(said), WorkflowTemplate.workspace_id == workspace_id).first())
    name = params.get("playbook_name") or (said if isinstance(said, str) else None)
    found = playbooks_called(db, workspace_id, name) if name and str(name).strip() else []
    return found[0] if len(found) == 1 else None


def why_it_cannot_run(playbook: Any) -> Optional[str]:
    """Why ``playbook`` can't run, in the owner's words: no steps, or a step with no
    agent (F270's words); None when it can."""
    from services.playbook_run_refusal import run_refusal

    if not getattr(playbook, "steps", None):
        return NO_STEPS.format(id=playbook.id, name=playbook.name)
    return run_refusal(playbook)


def refuses_a_run_no_one_can_do(handler: Handler) -> Handler:
    """Wrap platform_execute_playbook: a playbook that can't run gets no run and no card."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        playbook = _the_playbook(db, workspace_id, params or {})
        words = why_it_cannot_run(playbook) if playbook is not None else None
        if not words:
            return await handler(db, workspace_id, params)
        logger.info("[F321] run of playbook %s refused before it started: %s", playbook.id, words)
        return {"success": False, "error": words, "playbook_id": playbook.id, "run_started": False}
    return wrapped


# The handlers Auto's tools run (platform_executor binds them under these names, so a
# test that patches its execute_playbook still reaches the patch). The plain handlers
# stay in handlers_playbooks for the board's routes and their own tests.
add_playbook_step = needs_an_agent(_steps.add_playbook_step)
execute_playbook = refuses_a_run_no_one_can_do(_steps.execute_playbook)

__all__ = ["NEEDS_AN_AGENT", "NO_SINGLE_AGENT", "NO_STEPS", "add_playbook_step", "answering_to", "execute_playbook",
           "needs_an_agent", "refuses_a_run_no_one_can_do", "why_it_cannot_run"]
