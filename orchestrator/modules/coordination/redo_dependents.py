"""F286 (night 8): what was built from a redone step runs again.

#0282.1 and #0282.2 were sent back at 03:15:03 and redone at 03:15:35, but the
synthesis that had started at 03:14:49 completed from the drafts the owner had
rejected, and became the mission card's result ("Dear Club Member… delightful…!").
#0267's last card pasted the rejected letter back in, and #0446.4's summary quoted
drafts the owner had sent back. A redo changed one step; nothing built from it ran
again.

Now, when a step is sent back or re-briefed (services/run_redo), every step built from
it runs again after its redo: the steps that declare they build on it, and on those,
and a synthesis step after it that declares no inputs (it merges the mission's steps).

- One that finished (done, waiting for the owner's check, checked) or is queued waits
  again for the steps it builds on, then runs from their new work. Its card keeps what
  it showed in its history and goes back to the Inbox. What it was told on its last run
  goes: the inputs it was given whole (``upstream_results``) and in the digest, its
  check's feedback and requeue count, its last output. The owner's own words to it stay
  (with that output, which the revision prompt shows beside them), except on a
  synthesis, which is built again from what it merges (its revision prompt never shows
  its inputs);
- one already sent back waits for that redo too, keeping the owner's words;
- one running, or about to, is marked (``RERUN_KEY``): when its run ends it waits again
  instead of being checked (``runs_again_after_a_redo``, around the dispatcher's record
  of a finished run). So is one the mission is checking that moment; if it passes
  first, the next run recorded in the mission (the redo's own) sends it back to wait.

A redo whose own inputs are being redone waits for them the same way
(``waits_for_its_inputs``), with the owner's words. The steps are read under a row lock
(the redo holds its mission's lock and waits briefly, services/mission_reopen).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from sqlalchemy.orm import Session

from core.models.orchestration_enums import ActorType, FailureReasonCode, TaskState, TaskType

logger = logging.getLogger(__name__)

# On a step whose run started before a step it builds on was sent back: that step's label.
RERUN_KEY = "rerun_after_redo"
# Steps that wait again at once (one the mission is checking right now is marked instead).
WAIT_AGAIN = (TaskState.QUEUED.value, TaskState.COMPLETED.value, TaskState.VERIFYING.value,
              TaskState.VERIFIED.value, TaskState.RETRYING.value)
# Steps running or about to: marked, and they wait again when their run ends.
IN_FLIGHT = (TaskState.ASSIGNED.value, TaskState.RUNNING.value, TaskState.STALLED.value)
# A run that ended in one of these, on a marked step, waits again.
ENDS_TO_WAIT = (TaskState.COMPLETED.value, TaskState.QUEUED.value)
# A step's inputs are settled when every step it builds on is in one of these.
SETTLED = (TaskState.VERIFIED.value, TaskState.FAILED.value, TaskState.SKIPPED.value)
# What a step was told on its last run, rebuilt when it runs again (step_inputs.RESULTS_KEY,
# the PRD-164 digest, the reconciler's requeue count).
TOLD_LAST_TIME = ("upstream_results", "field_digest", "verification_requeues")
OWNERS_WORDS = "verification_feedback"
LAST_OUTPUT = "previous_output"
WHY = "built from {label}, which the owner sent back"
OWN_INPUTS = "{label} builds on a step that is being redone, so its redo waits for it"
BY_THE_MISSION = "the mission"
PERSON = "user:"


def rerun_what_builds_on(db: Session, step: Any, *, by: str, label: str) -> List[Any]:
    """Every step built from ``step`` runs again after its redo (see the module). Returns
    them. The caller commits."""
    from modules.coordination.owner_checks import carry_on

    steps = dependents_of(db, step)
    for task in steps:
        _runs_again_after(db, task, by=by, label=label)
    if steps:
        carry_on(db, step.run_id, by=by)  # a check one of them waited for no longer holds the mission
        logger.info("[F286] %d step(s) built from %s run again after its redo", len(steps), label)
    return steps


def dependents_of(db: Session, step: Any) -> List[Any]:
    """The steps built from ``step``, and from those, in plan order, their rows locked."""
    from core.models.orchestration import OrchestrationTask

    found = _built_from(db, {step.id} | _synthesis_after(db, step)) - {step.id}
    if not found:
        return []
    return (db.query(OrchestrationTask)
            .filter(OrchestrationTask.id.in_(list(found)), OrchestrationTask.run_id == step.run_id)
            .order_by(OrchestrationTask.sequence_number)
            .with_for_update().populate_existing().all())


def _built_from(db: Session, roots: Set[Any]) -> Set[Any]:
    """``roots`` and every step that declares it builds on one of them, transitively."""
    from core.models.orchestration import OrchestrationTaskDependency

    found, frontier = set(roots), set(roots)
    while frontier:
        rows = db.query(OrchestrationTaskDependency.task_id).filter(
            OrchestrationTaskDependency.depends_on_task_id.in_(list(frontier))).all()
        frontier = {row[0] for row in rows} - found
        found |= frontier
    return found


def _synthesis_after(db: Session, step: Any) -> Set[Any]:
    """The mission's synthesis steps after ``step`` that declare no inputs: they merge
    the mission's steps."""
    from sqlalchemy import exists

    from core.models.orchestration import OrchestrationTask, OrchestrationTaskDependency

    rows = db.query(OrchestrationTask.id).filter(
        OrchestrationTask.run_id == step.run_id,
        OrchestrationTask.task_type == TaskType.SYNTHESIS.value,
        OrchestrationTask.sequence_number > (step.sequence_number or 0),
        ~exists().where(OrchestrationTaskDependency.task_id == OrchestrationTask.id),
    ).all()
    return {row[0] for row in rows}


def waits_for_its_inputs(db: Session, step: Any) -> bool:
    """Whether a step ``step`` builds on is being worked again (redone, or waiting to be):
    ``step``'s own redo then waits for it, instead of reading the work it replaces."""
    from core.models.orchestration import OrchestrationTask, OrchestrationTaskDependency

    states = db.query(OrchestrationTask.state).join(
        OrchestrationTaskDependency, OrchestrationTaskDependency.depends_on_task_id == OrchestrationTask.id,
    ).filter(OrchestrationTaskDependency.task_id == step.id).all()
    return any(state not in SETTLED for (state,) in states)


def its_redo_waits(db: Session, step: Any, *, by: str, label: str) -> bool:
    """The redone ``step`` waits for its inputs when one is being redone. True when it does."""
    if not waits_for_its_inputs(db, step):
        return False
    back_to_waiting(db, step, why=OWN_INPUTS.format(label=label), by=by, keep_words=True)
    return True


def _runs_again_after(db: Session, task: Any, *, by: str, label: str) -> None:
    """One step built from a redone one: waiting again now, or marked to when its run ends."""
    if task.state in WAIT_AGAIN and not _being_checked(task):
        back_to_waiting(db, task, why=WHY.format(label=label), by=by)
    elif task.state in IN_FLIGHT or task.state == TaskState.VERIFYING.value:
        task.input_context = {**(task.input_context or {}), RERUN_KEY: label}


def _being_checked(task: Any) -> bool:
    """The mission is checking ``task`` right now (it waits for no owner's check)."""
    from modules.coordination.owner_checks import WAITING_KEY

    return task.state == TaskState.VERIFYING.value and not (task.input_context or {}).get(WAITING_KEY)


def back_to_waiting(db: Session, task: Any, *, why: str, by: str, keep_words: Optional[bool] = None) -> None:
    """``task`` waits again for the steps it builds on, then runs from them (see the
    module). Its card keeps its last result in its history. ``keep_words`` keeps the
    owner's words whatever the step is (the step the owner just corrected)."""
    from services.orchestration_state import transition_task

    redo_to_come = task.state == TaskState.RETRYING.value
    keep = _keeps_the_owners_words(task) if keep_words is None else keep_words
    task.input_context = _context_to_run_again(task, keep_words=keep)
    task.output = None  # never read, nor put on the mission's card, as current work
    task.completed_at = None
    actor = ActorType.HUMAN if by.startswith(PERSON) else ActorType.COORDINATOR
    transition_task(db=db, task=task, new_state=TaskState.PENDING, actor_type=actor, actor_id=by, reason=why)
    _its_card_waits(db, task, why=why, by=by, redo_to_come=redo_to_come)


def _context_to_run_again(task: Any, *, keep_words: bool) -> Dict[str, Any]:
    """``task``'s input context for its next run: without what it was told last time,
    with the owner's words and the output they correct when ``keep_words``."""
    from modules.coordination.owner_checks import WAITING_KEY

    context = dict(task.input_context or {})
    dropped = {WAITING_KEY, RERUN_KEY, OWNERS_WORDS, LAST_OUTPUT, *TOLD_LAST_TIME}
    rest = {key: value for key, value in context.items() if key not in dropped}
    if not (keep_words and context.get(OWNERS_WORDS)):
        return rest
    # its latest output (the draft a Reject just sent back is its output too)
    return {**rest, OWNERS_WORDS: context[OWNERS_WORDS], LAST_OUTPUT: task.output or context.get(LAST_OUTPUT)}


def _keeps_the_owners_words(task: Any) -> bool:
    """The owner's own words to ``task`` stay for its next run, unless it is a synthesis,
    which is built again from what it merges."""
    return (task.failure_reason_code == FailureReasonCode.VERIFICATION_REJECT.value
            and task.task_type != TaskType.SYNTHESIS.value)


def _its_card_waits(db: Session, task: Any, *, why: str, by: str, redo_to_come: bool) -> None:
    """The step's card: Assigned when its redo is still to come; otherwise its last result
    into its history and the card back to the Inbox. A card the session lane runs is the
    lane's (F094)."""
    from modules.coordination.owner_checks import step_card
    from services.cli_ticket_lane import is_lane_owned
    from services.orchestration_board_bridge import sync_board_status
    from services.redo_cards import card_runs_again, waits_its_turn

    card = step_card(db, task)
    if card is None or is_lane_owned(card):
        return
    if not redo_to_come:
        card_runs_again(card, why=why, by=by)
    sync_board_status(db, task)
    if redo_to_come:
        waits_its_turn(db, card)


def runs_again_after_a_redo(record: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``MissionDispatcher.record_task_completion``: a run that started before a step
    it builds on was sent back waits again once it ends, instead of being checked; and a
    marked step the mission checked meanwhile waits again too."""
    @functools.wraps(record)
    def wrapped(db: Session, task: Any, result: Dict[str, Any]) -> None:
        record(db, task, result)
        _after_its_run(db, task)
    return wrapped


def _after_its_run(db: Session, task: Any) -> None:
    context = task.input_context if isinstance(task.input_context, dict) else {}
    label = context.get(RERUN_KEY)
    if label and task.state in ENDS_TO_WAIT:
        back_to_waiting(db, task, why=WHY.format(label=label), by=BY_THE_MISSION)
    elif label and task.state == TaskState.FAILED.value:
        task.input_context = {key: value for key, value in context.items() if key != RERUN_KEY}
    for checked in _checked_meanwhile(db, task.run_id):
        back_to_waiting(db, checked, why=WHY.format(label=checked.input_context[RERUN_KEY]), by=BY_THE_MISSION)


def _checked_meanwhile(db: Session, run_id: Any) -> Iterable[Any]:
    """Marked steps that passed the mission's check before the redo they wait for ran."""
    from core.models.orchestration import OrchestrationTask

    return db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run_id, OrchestrationTask.state == TaskState.VERIFIED.value,
        OrchestrationTask.input_context.has_key(RERUN_KEY),
    ).all()


__all__ = ["RERUN_KEY", "back_to_waiting", "dependents_of", "its_redo_waits", "rerun_what_builds_on",
           "runs_again_after_a_redo", "waits_for_its_inputs"]
