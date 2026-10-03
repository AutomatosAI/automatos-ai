"""F247 (night 7): a mission whose AI credit runs out pauses, and Resume continues it.

Mission #0176, from 06:53Z: the AI credit ran out mid-mission.
- Every retry of a step failed the same way, so the progress ledger read the churn as
  a stall. It re-planned twice by itself, which used up both of the re-plans the owner
  has, and filed the steps each re-plan replaced as Done: eight of them had never run.
- Then it halted ("Joiner halt: no forward progress across 3 ledger checks"), with four
  step cards left in the Inbox.
- Resume was refused (expected 'paused'), Replan was refused (maximum 2), and the board
  refuses a mission's steps, so those four cards could run nowhere.

A step that stops because the credit ran out has not failed an attempt. It goes back
to the queue with the attempt unspent. Its mission pauses, with the reason on its
card. A paused mission is never ticked, so nothing re-plans or halts, and Resume
carries on from that step once the credit is topped up. The owner's one notice of the
outage is the bell's (F197).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

PAUSED_TEXT = ("Paused: the AI provider's account ran out of credit. Nothing is lost: top up the provider "
               "account, then press Resume to carry on from the step it stopped at.")
QUEUED_AGAIN = "Out of AI credit: queued again with its attempt unspent; the mission is paused"


def pauses_when_credit_runs_out(record: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``MissionDispatcher.record_task_completion``. A step that stopped for credit
    goes back to the queue unspent and pauses its mission; any other result is recorded
    as before."""
    @functools.wraps(record)
    def wrapped(db: Session, task: Any, result: Dict[str, Any]) -> None:
        if not ran_out_of_credit(result):
            return record(db, task, result)
        from services.orchestration_board_bridge import sync_board_status

        _queue_again(db, task)
        _pause(db, task)
        sync_board_status(db, task)
        return None
    return wrapped


def ran_out_of_credit(result: Any) -> bool:
    """The step's model call was refused for credit: the providers' own refusals, or
    the plain sentence the agent factory puts in their place."""
    from core.llm.credit import is_out_of_credit

    return isinstance(result, dict) and result.get("status") != "success" and is_out_of_credit(result.get("error"))


def _queue_again(db: Session, task: Any) -> None:
    """Back to the queue with its attempt unspent: the credit stopped it, not the step."""
    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from core.models.orchestration_enums import ActorType, TaskState
    from services.orchestration_state import transition_task

    agent_id = task.assigned_agent_id
    task.assigned_agent_id = None
    task.failure_detail = OUT_OF_CREDIT_TEXT
    transition_task(db=db, task=task, new_state=TaskState.QUEUED, actor_type=ActorType.AGENT,
                    actor_id=str(agent_id or "unknown"), reason=QUEUED_AGAIN)


def _pause(db: Session, task: Any) -> None:
    """Pause the step's mission, once: a second step stopped by the same outage finds it paused."""
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import ActorType, RunState, StopReason
    from services.orchestration_state import transition_run

    run = db.query(OrchestrationRun).filter(OrchestrationRun.id == task.run_id).first()
    if run is None or run.state != RunState.RUNNING.value:
        return
    transition_run(db=db, run=run, new_state=RunState.PAUSED, actor_type=ActorType.COORDINATOR,
                   actor_id="dispatcher", reason="Out of AI credit: mission paused",
                   stop_reason=StopReason.OUT_OF_CREDIT.value, stop_detail=PAUSED_TEXT)
    logger.warning("[coordinator] run %s paused: the AI credit ran out (task %s queued again)", run.id, task.id)


__all__ = ["PAUSED_TEXT", "pauses_when_credit_runs_out", "ran_out_of_credit"]
