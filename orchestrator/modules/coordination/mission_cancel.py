"""F245 (night 7): cancelling a mission stops it, and its cards end Cancelled.

#0119 was cancelled from its page while two steps ran: they kept going (14 model
calls after the cancel), finished their drafts and stayed In progress, while
the mission's card and its five unstarted steps went to Done. The cancel only
skipped steps that had not started, and a skipped step and a cancelled mission
both showed as Done.

Now every unfinished step is skipped and its card cancelled, saying who
cancelled it (a step a Claude Code session works stops with its card, as any
cancelled ticket does). A step already running stops too: the tick runs a
mission's steps on the scheduler worker, while the cancel can come from any
worker, so a running step reads its mission's state from the database while it
works, and stops when it reads ``cancelled``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Dict, List

from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import ActorType, FailureReasonCode, RunState, TaskState
from services.orchestration_state import transition_task

logger = logging.getLogger(__name__)

# A step that is finished (verified) or already closed (skipped) is left as it is.
SETTLED_STEP_STATES = (TaskState.VERIFIED.value, TaskState.SKIPPED.value)
OPEN_CARD_STATUSES = ("inbox", "assigned", "in_progress", "review", "blocked", "failed")
# How often a running step reads whether its mission was cancelled.
CANCEL_POLL_SECONDS = 3.0
CANCELLED_WHILE_RUNNING = "The mission was cancelled while this step ran."
MISSION_CANCELLED_REASON = "the mission was cancelled"


def close_open_steps(db: Session, run_id: Any, *, by: str, reason: str) -> List[int]:
    """Skip every unfinished step of the mission and cancel its card. Returns the
    cards cancelled. Nothing is committed: the caller commits the mission's
    cancel with its cards. Each card stops in a savepoint of its own (as F224's
    stop_mission_sessions does): a card the database refuses is logged and left
    open, never costing the mission its cancel."""

    steps = (
        db.query(OrchestrationTask)
        .filter(OrchestrationTask.run_id == run_id, OrchestrationTask.state.notin_(SETTLED_STEP_STATES))
        .all()
    )
    for step in steps:
        step.failure_reason_code = FailureReasonCode.CANCELLED.value
        step.failure_detail = reason
        transition_task(db=db, task=step, new_state=TaskState.SKIPPED, actor_type=ActorType.HUMAN,
                        actor_id=by, reason=reason)
    db.flush()
    cards = (
        db.query(BoardTask)
        .join(OrchestrationTask, OrchestrationTask.id == BoardTask.orchestration_task_id)
        .filter(OrchestrationTask.run_id == run_id, BoardTask.status.in_(OPEN_CARD_STATUSES))
        .order_by(BoardTask.id)
        .all()
    )
    return [card.id for card in cards if _stopped(db, card, by=by, reason=reason)]


def _stopped(db: Session, card: Any, *, by: str, reason: str) -> bool:
    import services.board_cancel as board_cancel

    try:
        with db.begin_nested():
            return board_cancel.stop_ticket_run(db, card, by=by, reason=reason)
    except Exception:  # noqa: BLE001 -- one card never costs the mission its cancel; it is logged
        logger.exception("[MissionCancel] could not cancel step card %s", card.id)
        return False


async def until_mission_cancelled(work: Awaitable[Any], run_id: Any,
                                  *, poll_seconds: float = CANCEL_POLL_SECONDS) -> Any:
    """``work``'s result, unless its mission is cancelled first: then ``work`` is
    cancelled (an in-flight model call is dropped) and the step's result says so."""
    job = asyncio.ensure_future(work)
    if run_id is None:
        return await job
    while True:
        done, _ = await asyncio.wait({job}, timeout=poll_seconds)
        if done:
            return job.result()
        if await asyncio.to_thread(mission_cancelled, run_id):
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            logger.info("[MissionCancel] a step of mission %s stopped: the mission was cancelled", run_id)
            return cancelled_step_result()


def cancelled_step_result() -> Dict[str, Any]:
    return {"status": "cancelled", "error": CANCELLED_WHILE_RUNNING}


def mission_cancelled(run_id: Any) -> bool:
    """Whether the mission is cancelled, read on a session of its own. A read that
    fails says no: the step carries on, and the next read asks again."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        state = db.query(OrchestrationRun.state).filter(OrchestrationRun.id == run_id).scalar()
        return state == RunState.CANCELLED.value
    except Exception:  # noqa: BLE001 -- a failed read never stops a step; it is logged and asked again
        logger.warning("[MissionCancel] could not read mission %s's state", run_id, exc_info=True)
        return False
    finally:
        db.close()


def cancelled_while_it_ran(db: Session, run: Any) -> bool:
    """Whether ``run`` was cancelled since the tick read it, without reloading it
    (a reload would drop what the tick has recorded on it and not yet written)."""
    state = db.query(OrchestrationRun.state).filter(OrchestrationRun.id == run.id).scalar()
    return state == RunState.CANCELLED.value
