"""F245 (night 7): Cancel stops a playbook run or a mission and leaves its cards Cancelled.

The board's Cancel only flipped the card. On a playbook's card (#0096, #0150)
the run kept going, billed seven more model calls, and its end moved the card
from Cancelled to Done; on a mission's card (#0098) the mission finished and the
card went Done five seconds later. The playbooks page's own cancel stopped the
run but filed the card as failed.

A playbook's card and its run are now cancelled together, wherever the cancel
comes from: the run is marked cancelled (its executor checks before every step
and every model call, on any worker), the task running it on this worker is
cancelled at once, and its session step tickets stop with it. A mission's card
cancels the mission (``CoordinatorService.cancel_mission``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

PLAYBOOK_CARD = "recipe"
# A playbook step's session ticket: "recipe:<run>:<step>" (api/recipe_executor).
PLAYBOOK_STEP_PREFIX = "recipe:"
MISSION_CARD = "orchestration"
MISSION_STEP = "orchestration_task"
FINISHED_RUN_STATUSES = ("completed", "failed", "cancelled")
PLAYBOOK_CANCEL_PERMISSION = "playbooks:execute"
MISSION_CANCEL_PERMISSION = "missions:execute"
BOARD_CANCEL_REASON = "cancelled on the board"
# How much of a mission's goal a refusal quotes.
GOAL_SHOWN_CHARS = 80
PERSON_PREFIX = "user:"
# A refusal: the HTTP status and why, in the owner's words.
Refusal = Tuple[int, str]


def is_playbook_card(task: Any) -> bool:
    """A playbook run's own card: its source id is the bare execution id."""
    source_id = getattr(task, "source_id", None) or ""
    return getattr(task, "source_type", None) == PLAYBOOK_CARD and bool(source_id) \
        and not source_id.startswith(PLAYBOOK_STEP_PREFIX)


def playbook_run_of(db: Any, task: Any) -> Optional[Any]:
    """The playbook run a playbook card shows, in the card's workspace."""
    from core.models.core import RecipeExecution

    if not is_playbook_card(task):
        return None
    return db.query(RecipeExecution).filter(
        RecipeExecution.execution_id == task.source_id,
        RecipeExecution.workspace_id == task.workspace_id,
    ).first()


def cancel_playbook_run(db: Any, execution: Any, *, by: str, reason: str) -> bool:
    """Stop ``execution`` and cancel its card and its session step tickets, in one
    commit: all of them or, if any fails, none (the cancel can then be asked
    again). Then the run's task on this worker is cancelled at once. False when
    the run had already finished (its card is left as it is)."""
    if execution.status in FINISHED_RUN_STATUSES:
        return False
    execution.status = "cancelled"
    execution.error_message = reason
    execution.completed_at = datetime.now(timezone.utc)
    _cancel_playbook_cards(db, execution, by=by, reason=reason)
    db.commit()
    from api.recipe_executor import request_execution_cancel

    stopped_here = request_execution_cancel(execution.execution_id)
    logger.info("[RunCancel] playbook run %s cancelled by %s (stopped on this worker: %s)",
                execution.execution_id, by, stopped_here)
    return True


def _cancel_playbook_cards(db: Any, execution: Any, *, by: str, reason: str) -> None:
    """The run's card, then its session step tickets (F116), each saying who (and,
    in its notes, when: F273); the caller commits them with the run."""
    from core.models.core import BoardTask
    from services.board_cancel import stop_run_step_tickets, stop_ticket_run
    from services.cancel_notes import CANCELLED_THE_RUN

    card = db.query(BoardTask).filter(
        BoardTask.source_type == PLAYBOOK_CARD,
        BoardTask.source_id == execution.execution_id,
        BoardTask.workspace_id == execution.workspace_id,
    ).first()
    if card is not None:
        stop_ticket_run(db, card, by=by, reason=reason, note=CANCELLED_THE_RUN)
    stopped = stop_run_step_tickets(db, execution.execution_id, by=by,
                                    reason=f"cancelled with run {execution.execution_id}")
    if stopped:
        logger.info("[RunCancel] %s: step tickets cancelled with it: %s", execution.execution_id, stopped)


def mission_run_of(db: Any, task: Any) -> Optional[Any]:
    """The mission a mission card or step belongs to, in the card's workspace. A
    step carries its mission through its task (it has no run id of its own)."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    run_id = getattr(task, "orchestration_run_id", None)
    if run_id is None and getattr(task, "orchestration_task_id", None) is not None:
        step = db.query(OrchestrationTask.run_id).filter(OrchestrationTask.id == task.orchestration_task_id).first()
        run_id = step[0] if step else None
    if run_id is None:
        return None
    return db.query(OrchestrationRun).filter(
        OrchestrationRun.id == run_id, OrchestrationRun.workspace_id == task.workspace_id,
    ).first()


def mission_is_live(run: Any) -> bool:
    from core.models.orchestration_enums import TERMINAL_RUN_STATES, RunState

    return run is not None and RunState(run.state) not in TERMINAL_RUN_STATES


def cancel_mission_run(db: Any, run: Any, *, by: str) -> Any:
    """Cancel ``run`` the way the mission page does (its record names the person
    as the mission page's does: without the ``user:`` of a card's). Commits."""
    from services.coordinator_service import get_coordinator_service

    person = by[len(PERSON_PREFIX):] if by.startswith(PERSON_PREFIX) else by
    run = get_coordinator_service().cancel_mission(db=db, run_id=run.id, actor_id=person)
    db.commit()
    return run


def cancel_ticket(db: Any, task: Any, *, by: str, may: Callable[[str], bool]) -> Optional[Refusal]:
    """The board's Cancel: the ticket and whatever runs it (F245). A playbook's
    card stops its run; a mission's card cancels the mission; a step of a live
    mission is the mission's to stop. ``may(permission)`` says whether the person
    may stop that run, as its own page would ask. None when cancelled."""
    from services.board_cancel import cancel_board_ticket

    source = getattr(task, "source_type", None)
    if is_playbook_card(task):
        execution = playbook_run_of(db, task)
        if execution is not None and execution.status not in FINISHED_RUN_STATUSES:
            if not may(PLAYBOOK_CANCEL_PERMISSION):
                return 403, f"Permission denied: {PLAYBOOK_CANCEL_PERMISSION}"
            cancel_playbook_run(db, execution, by=by, reason=BOARD_CANCEL_REASON)
            return None
    elif source in (MISSION_CARD, MISSION_STEP):
        run = mission_run_of(db, task)
        if mission_is_live(run):
            return _cancel_live_mission(db, task, run, by=by, may=may)
    cancel_board_ticket(db, task, by=by, reason=BOARD_CANCEL_REASON)
    return None


def _cancel_live_mission(db: Any, task: Any, run: Any, *, by: str, may: Callable[[str], bool]) -> Optional[Refusal]:
    """A live mission's card cancels the mission; one of its steps is refused,
    naming the mission: the mission runs its steps (PRD-252 D6)."""
    from services.ticket_numbers import ticket_label

    if task.source_type == MISSION_STEP:
        return 409, (f"{ticket_label(task, capital=True)} is a step of the mission \"{(run.goal or '')[:GOAL_SHOWN_CHARS]}\": "
                     f"the mission runs its steps, so cancel the mission to stop it (/missions/{run.id}).")
    if not may(MISSION_CANCEL_PERMISSION):
        return 403, f"Permission denied: {MISSION_CANCEL_PERMISSION}"
    cancel_mission_run(db, run, by=by)
    return None
