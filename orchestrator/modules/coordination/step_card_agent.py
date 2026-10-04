"""A mission step runs on the agent its card was given to (F287, night 8).

Night 8: the owner gave #0237.1 to the Inventory Watchdog on the board, and after
the plan was approved it ran on the Analyst. #0237.2 and #0237.3, given to the Ops
Manager, ran on the Content Creator; #0352.2 showed the Ops Manager and the Content
Creator did it; #0333.2, given to the Support Agent, was redone by the Content
Creator. The board's Assign writes only the card's agent, and the dispatcher chose
each step's agent by its role, honouring only a pin (``pinned_agent_id``, which a
plan edit naming one agent sets).

``runs_on_the_cards_agent`` wraps the per-step dispatch, a first run or a redo. When
the step's card shows an active agent of the workspace that the step does not have
already (pinned, assigned, or chosen when it last ran), the step is pinned to that
agent the way a plan edit pins one: the agent's id in ``input_context`` and its name
as the step's role. The matcher then puts it first ("Explicitly assigned: a person
chose agent ..."). A card that shows the step's own agent changes nothing.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Sequence, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models.core import BoardTask
from services.orchestration_board_bridge import STEP_CARD_SOURCE_TYPE

logger = logging.getLogger(__name__)

# What the matcher reads as a person's choice (AgentMatcher.rank), as a plan edit writes it.
PIN_KEY = "pinned_agent_id"
# The dispatcher's choice on the step (MissionDispatcher._dispatch_single): the agent it
# last ran on once it was dispatched, a preview while the plan waited ("plan").
MATCH_KEY = "agent_match"
CHOSEN_AT_DISPATCH = "dispatch"
ACTIVE = "active"


def runs_on_the_cards_agent(dispatch_single: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``MissionDispatcher._dispatch_single`` (see the module)."""
    @functools.wraps(dispatch_single)
    def wrapped(db: Session, run: Any, task: Any, agents: Sequence[Any]) -> Any:
        pin_the_cards_agent(db, run, task, agents)
        return dispatch_single(db, run, task, agents)
    return wrapped


def pin_the_cards_agent(db: Session, run: Any, task: Any, agents: Sequence[Any]) -> Optional[int]:
    """Pin ``task`` to the agent its card was given to, when that is an active agent of
    the mission's workspace other than the step's own. Returns its id, or None."""
    chosen = _agent_id(_cards_agent(db, run, task))
    if chosen is None or chosen in _its_own_agents(task):
        return None
    agent = _active_agent(agents, chosen, run.workspace_id)
    if agent is None:
        logger.info("[F287] step %s: its card's agent %s is not an active agent of the workspace", task.id, chosen)
        return None
    context = task.input_context if isinstance(task.input_context, dict) else {}
    task.agent_role = agent.name
    task.input_context = {**context, PIN_KEY: chosen}
    db.flush()  # the claim re-reads the step, and an unflushed pin would be lost there
    logger.info("[F287] step %s of mission %s goes to agent %s, as its card says", task.id, run.id, chosen)
    return chosen


def _cards_agent(db: Session, run: Any, task: Any) -> Any:
    """The agent on the step's card, if it has a card and an agent."""
    return db.execute(
        select(BoardTask.assigned_agent_id)
        .where(BoardTask.workspace_id == run.workspace_id,
               BoardTask.source_type == STEP_CARD_SOURCE_TYPE,
               BoardTask.orchestration_task_id == task.id)
        .order_by(BoardTask.id)
        .limit(1)
    ).scalar()


def _its_own_agents(task: Any) -> Set[int]:
    """The agents the step has already: its pin, the agent it is assigned to, and the
    one chosen when it last ran (its card shows that one after it has run)."""
    context = task.input_context if isinstance(task.input_context, dict) else {}
    match = context.get(MATCH_KEY) if isinstance(context.get(MATCH_KEY), dict) else {}
    ran_on = match.get("agent_id") if match.get("decided_at") == CHOSEN_AT_DISPATCH else None
    own = (context.get(PIN_KEY), getattr(task, "assigned_agent_id", None), ran_on)
    return {agent_id for agent_id in map(_agent_id, own) if agent_id is not None}


def _active_agent(agents: Sequence[Any], agent_id: int, workspace_id: Any) -> Optional[Any]:
    """The roster's active agent with ``agent_id`` in the mission's workspace."""
    return next((agent for agent in agents
                 if _agent_id(getattr(agent, "id", None)) == agent_id
                 and getattr(agent, "status", None) == ACTIVE
                 and getattr(agent, "workspace_id", None) == workspace_id), None)


def _agent_id(value: Any) -> Optional[int]:
    """An agent's id as a number; anything else (None, other text, a bool) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    text = str(value).strip()
    return int(text) if text.isdigit() else None


__all__ = ["pin_the_cards_agent", "runs_on_the_cards_agent"]
