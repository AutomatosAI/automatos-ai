"""Giving an answered card to another agent runs it again, with that agent (F309, night 9).

Night 9: "Please give card 1859 to the Shopify Business Analyst instead - the Analyst
couldn't find the data." Auto's platform_assign_task changed #1859's agent and nothing
else: the card sat in Review with the Analyst's old answer ("I don't have access to your
specific data") until the owner sent it back by hand. The tool only ever moved an Inbox
card to Assigned, and the dispatcher claims Assigned cards only.

The board's way back to an agent for a card that has an answer is its Reject
(``api.board_tasks.send_back``): the answer goes on record (``keep_previous_run``), the
card's result is cleared, and it goes back to Assigned for the dispatcher, woken at once
(``_back_to_its_agent``). A card in Review or Done given to another agent goes the same
way now, to its new agent: the old answer stays in the card's history, every correction
the owner made on the card stays with it, and the new agent's run is told the card was
given to it (``services.ticket_redo.GIVEN_TO_YOU``), never that its draft was sent back.
A playbook's card and a mission's step are run by their playbook or mission, as before.
"""
from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_board_tasks import assign_board_task as _assign_board_task

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

ANSWERED = ("review", "done")
BACK_TO_AN_AGENT = "assigned"
BY_AN_AGENT = "platform_tool"
RUNS_AGAIN = ("{label} is with {agent} now and runs again from its brief; the last answer is kept in the card's "
              "history.")


def an_answered_card_runs_again(assign: Handler) -> Handler:
    """Wrap ``assign_board_task``: a card in Review or Done given to another agent goes
    back to Assigned for that agent, the board's way (see the module)."""
    @functools.wraps(assign)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        before = _answered_card(db, workspace_id, (params or {}).get("task_id"))
        out = await assign(db, workspace_id, params)
        if before is None or not isinstance(out, dict) or out.get("success") is not True:
            return out
        return _run_again(db, workspace_id, params, before, out)
    return wrapped


def _answered_card(db: Session, workspace_id: Any, ref: Any) -> Optional[Tuple[int, Any]]:
    """(id, agent) of the card ``ref`` names when it has an answer and the board's
    dispatcher runs it; None for any other card, or one that can't be told."""
    from core.models.core import BoardTask
    from services.run_redo import takes_its_own_redo
    from services.ticket_refs import ticket_id_named

    task_id, _ = ticket_id_named(db, workspace_id, ref) if ref not in (None, "") else (None, None)
    task = (db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
            if task_id else None)
    if task is None or task.status not in ANSWERED or takes_its_own_redo(task):
        return None
    return task.id, task.assigned_agent_id


def _run_again(db: Session, workspace_id: Any, params: Dict[str, Any], before: Tuple[int, Any],
               out: Dict[str, Any]) -> Dict[str, Any]:
    """Back to Assigned for the card's new agent, its answer on record; the tool's own
    answer when the agent did not change, or when the card was decided meanwhile."""
    from api.board_tasks import _back_to_its_agent, _decide, keep_previous_run
    from core.models.core import BoardTask
    from services.board_consent import actor_from_user_id
    from services.ticket_numbers import ticket_label
    from services.ticket_redo import GIVEN_TO_YOU, GIVEN_WHY

    task = db.query(BoardTask).filter(BoardTask.id == before[0], BoardTask.workspace_id == workspace_id).first()
    if task is None or task.assigned_agent_id == before[1] or task.status not in ANSWERED:
        return out
    driver = params.get("_user_id")
    keep_previous_run(task, why=GIVEN_WHY, by=actor_from_user_id(driver) if driver else BY_AN_AGENT)
    if not _decide(db, task, seen=task.status, values={"status": BACK_TO_AN_AGENT}):
        return out
    _back_to_its_agent(db, SimpleNamespace(workspace_id=workspace_id, user_id=driver), task, GIVEN_TO_YOU)
    _the_owners_go_ahead(db, workspace_id, task, params)
    agent = out.get("assigned_agent") or "its new agent"
    return {**out, "status": task.status, "runs_again": True,
            "message": RUNS_AGAIN.format(label=ticket_label(task, capital=True), agent=agent)}


def _the_owners_go_ahead(db: Session, workspace_id: Any, task: Any, params: Dict[str, Any]) -> None:
    """The owner's word in chat is their consent for the run, as for any card Auto
    assigns on it (``_consent_for_chat_filed``)."""
    from modules.tools.discovery.handlers_board_tasks import _consent_for_chat_filed

    _consent_for_chat_filed(db, workspace_id, task, params)
    db.commit()


assign_board_task = an_answered_card_runs_again(_assign_board_task)

__all__ = ["an_answered_card_runs_again", "assign_board_task"]
