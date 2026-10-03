"""F242 (night 7): a mission that waits for the owner's check of each step does.

Auto launched #0139 saying "each step will wait for your check"
(``approval_mode: step_by_step``) and #0176 with ``review_mode: each_task``, and
the owner set #0113.2's card to wait before it ran. Every step still went from
Review to Done in a second or two: nothing read those keys, and a step's card
showed only what its mission did with it.

A step waits for the owner when its mission checks each step
(``check_each_step``, or either spelling night 7 saw) or its card was set to
wait (``review_mode`` human). Once it passes its own check, it stays in Review
for the owner, and its mission pauses, saying which step it waits for. Approve
lets the step through, and the mission carries on; Reject sends it back to be
redone (F243), and the mission carries on with the redo.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import ActorType, RunState, TaskState
from core.services.ticket_reasons import WAITING_FOR_YOUR_CHECK
from services.orchestration_state import transition_run, transition_task

logger = logging.getLogger(__name__)

CHECK_EACH_STEP = "check_each_step"
# The spellings Auto used on night 7 for the same wish (#0139, #0176).
SAID_STEP_BY_STEP = (("approval_mode", ("step_by_step", "each_step")), ("review_mode", ("each_task", "each_step")))
OWNER_REVIEWS = "human"
# On a step's input_context while it waits for the owner.
WAITING_KEY = "waiting_for_owner"
STEP_CARD = "orchestration_task"


def checks_each_step(config: Any) -> bool:
    """Whether a mission's config asks for the owner's check of every step."""
    config = config if isinstance(config, dict) else {}
    if config.get(CHECK_EACH_STEP) is True:
        return True
    return any(config.get(key) in values for key, values in SAID_STEP_BY_STEP)


def step_card(db: Session, task: Any) -> Optional[BoardTask]:
    return db.query(BoardTask).filter(BoardTask.source_type == STEP_CARD,
                                      BoardTask.orchestration_task_id == task.id).first()


def holds_for_the_owner(db: Session, task: OrchestrationTask) -> bool:
    """When ``task`` (a step that passed its own check) waits for the owner: it
    stays in Review, its card set to wait, and its mission pauses. True when it
    waits; False leaves the step to be verified as before."""
    from services.ticket_numbers import ticket_label

    run = db.get(OrchestrationRun, task.run_id) if getattr(task, "run_id", None) else None
    if run is None:
        return False
    card = step_card(db, task)
    if not (checks_each_step(run.config) or (card is not None and card.review_mode == OWNER_REVIEWS)):
        return False
    task.input_context = {**(task.input_context or {}), WAITING_KEY: True}
    if card is not None:
        card.review_mode = OWNER_REVIEWS
        card.status = "review"
        card.result = task.output or card.result
    if RunState(run.state) == RunState.RUNNING:
        label = ticket_label(card) if card is not None else f"step {task.sequence_number}"
        transition_run(db=db, run=run, new_state=RunState.PAUSED, actor_type=ActorType.COORDINATOR,
                       actor_id="coordinator", reason="a step waits for the owner's check",
                       stop_detail=f"{WAITING_FOR_YOUR_CHECK}{label}")
    db.flush()
    logger.info("[OwnerChecks] step %s of mission %s waits for the owner's check", task.id, task.run_id)
    return True


def let_through(db: Session, card: Any, *, by: str) -> bool:
    """Approve on a step held for the owner: the step is verified and its mission
    carries on. True when the card was such a step."""
    from services.orchestration_board_bridge import sync_board_status

    step = _held_step(db, card)
    if step is None:
        return False
    step.input_context = _released(step.input_context)
    transition_task(db=db, task=step, new_state=TaskState.VERIFIED, actor_type=ActorType.HUMAN, actor_id=by,
                    reason="approved by the owner")
    sync_board_status(db, step)
    carry_on(db, step.run_id, by=by)
    return True


def _held_step(db: Session, card: Any) -> Optional[OrchestrationTask]:
    if getattr(card, "source_type", None) != STEP_CARD or not getattr(card, "orchestration_task_id", None):
        return None
    step = db.get(OrchestrationTask, card.orchestration_task_id)
    if step is None or not (step.input_context or {}).get(WAITING_KEY) or step.state != TaskState.VERIFYING.value:
        return None
    return step


def _released(input_context: Any) -> dict:
    return {key: value for key, value in (input_context or {}).items() if key != WAITING_KEY}


def carry_on(db: Session, run_id: Any, *, by: str) -> None:
    """The mission resumes once no step waits for the owner. A mission paused for
    anything else (its budget, the owner's own Pause) stays paused."""
    run = db.get(OrchestrationRun, run_id)
    if run is None or run.state != RunState.PAUSED.value or not (run.stop_detail or "").startswith(WAITING_FOR_YOUR_CHECK):
        return
    still_waiting = [step for step in db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run_id, OrchestrationTask.state == TaskState.VERIFYING.value).all()
        if (step.input_context or {}).get(WAITING_KEY)]
    if still_waiting:
        return
    transition_run(db=db, run=run, new_state=RunState.RUNNING, actor_type=ActorType.HUMAN, actor_id=by,
                   reason="the owner checked the step it waited for")


def sent_back(db: Session, step: OrchestrationTask, *, by: str) -> None:
    """Reject on a step held for the owner (F243's redo): it no longer waits, and
    its mission carries on to redo it."""
    if not (step.input_context or {}).get(WAITING_KEY):
        return
    step.input_context = _released(step.input_context)
    carry_on(db, step.run_id, by=by)
