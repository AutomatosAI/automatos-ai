"""A mission's own card carries its result when it completes (F268, night 7b; F286 and
F268 again, night 8).

#0188 completed, and its card on the board stayed empty: the summary was only on its
last step's card (#0188.4) and on the mission page. The card follows the mission's
state (``orchestration_board_bridge.sync_mission_board_status``) and never took its
result. Now, when the mission completes, its card takes the mission's result, from the
steps of its plan (not those a re-plan replaced). A card that already says something
keeps it.

F286 (night 8): #0428's card showed only its last step, the club email; the margin and
the front-page line the owner had approved were not on it (#0214 and #0410 the same).
A mission with a step that pulls the others together (a synthesis step, or a last step
whose brief asks to summarise, combine or compile them) puts that step's output on its
card, as before. Any other mission puts every verified step's output on it, in
sequence, each under its card number and title.

F283 (night 8): a failed mission's steps that failed for good show Failed on their
cards, not Blocked (``_failed_steps_show_failed``): nothing will move them now.

F268 (night 8): #0176, resumed, completed while its card stayed Cancelled, with no
note and its old failure still on it. A failed mission run again (Resume, Replan)
reopens its card without the failure (``reopen_the_missions_card``), and a completed
mission's card ends Done with a note saying the mission completed.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MISSION_CARD = "orchestration"
SYNTHESIS = "synthesis"
DONE = "done"
BLOCKED = "blocked"
FAILED = "failed"
# A failed mission's card, or one the owner dismissed while the mission was failed.
REOPENED_FROM = ("failed", "cancelled", "closed")
# The mission's own state moves a reopened card on from here.
REOPENED_TO = "inbox"
# What a cancel leaves on a card (services/cancel_notes, services/board_cancel); its notes stay.
CANCEL_KEYS = ("cancelled", "cancel_requested_at")
NOTE_BY = "the mission"
COMPLETED_NOTE = "The mission completed. {words}"
SECTION = "### {label}\n\n{output}"


def carries_the_missions_result(sync: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``sync_mission_board_status``: a completed mission's card takes its result,
    and a failed mission's failed steps show Failed on their cards (F283)."""
    @functools.wraps(sync)
    def wrapped(db: Any, run: Any) -> None:
        from core.models.orchestration_enums import RunState

        sync(db, run)
        state = getattr(run, "state", None)
        if state == RunState.COMPLETED.value:
            _put_the_result_on_the_card(db, run)
        elif state == RunState.FAILED.value:
            _failed_steps_show_failed(db, run)
    return wrapped


def _failed_steps_show_failed(db: Any, run: Any) -> None:
    """F283: a step that failed for good shows Blocked while its mission runs on; once
    its mission has failed, nothing will move it, so its card is Failed, keeping why."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState
    from modules.coordination.mission_ends import step_cards

    failed = db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id,
                                                OrchestrationTask.state == TaskState.FAILED.value).all()
    for card in step_cards(db, run, failed).values():
        if card.status == BLOCKED:
            card.status, card.blocked_at, card.blocked_reason = FAILED, None, None
    db.flush()


def _put_the_result_on_the_card(db: Any, run: Any) -> None:
    from modules.coordination.mission_ends import completion_words

    card = _missions_card(db, run)
    if card is None:
        return
    if not str(card.result or "").strip():
        result = missions_result(db, run)
        if result:
            card.result = result
            db.flush()
            logger.info("[F268] mission %s's card %s carries its result", run.id, card.id)
    if card.status == DONE:
        _note(db, card, COMPLETED_NOTE.format(words=completion_words(db, run)))


def missions_result(db: Any, run: Any) -> Optional[str]:
    """What a completed mission's card shows: the output of its step that pulls the
    others together, or else every verified step's output in sequence, each under its
    card number and title (one step's alone)."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState
    from modules.coordination.mission_ends import live_steps, step_cards
    from services.ticket_numbers import ticket_numbers

    steps = live_steps(db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id).all())
    cards = step_cards(db, run, steps)
    steps = sorted(steps, key=lambda t: (t.sequence_number or 0, cards[t.id].id if t.id in cards else 0, str(t.id)))
    done = [t for t in steps if t.state == TaskState.VERIFIED.value and str(t.output or "").strip()]
    summary = _pulls_the_others_together(steps)
    if summary is not None and summary in done:
        return str(summary.output)
    if len(done) <= 1:
        return str(done[0].output) if done else None
    numbers = ticket_numbers(db, run.workspace_id, cards.values()) if cards else {}
    labels = {step_id: numbers.get(card.id) for step_id, card in cards.items()}
    return "\n\n".join(SECTION.format(label=_label(t, labels), output=str(t.output).strip()) for t in done)


def _pulls_the_others_together(steps: List[Any]) -> Optional[Any]:
    """A synthesis step, or a last step whose brief asks to summarise, combine or
    compile the others (``step_inputs.asks_to_summarise``)."""
    from modules.coordination.step_inputs import asks_to_summarise

    synthesis = [t for t in steps if t.task_type == SYNTHESIS]
    if synthesis:
        return synthesis[-1]
    return steps[-1] if steps and asks_to_summarise(steps[-1]) else None


def _label(task: Any, numbers: Dict[Any, Optional[str]]) -> str:
    number = numbers.get(task.id)
    return f"{number} {task.title}" if number else str(task.title)


def reopen_the_missions_card(db: Any, run: Any, *, note: str) -> None:
    """A failed mission run again (Resume, Replan): its card comes back from Failed,
    or from Cancelled or Closed where the owner dismissed it while the mission was
    failed (#0176), without the old failure, follows its mission again, and says
    why (``note``)."""
    from services.orchestration_board_bridge import sync_mission_board_status

    card = _missions_card(db, run)
    if card is None:
        return
    if card.status in REOPENED_FROM:
        card.status = REOPENED_TO
    card.error_message = None
    card.completed_at = None
    # The owner's earlier dismissal is history now: a later cancel says who again (F273).
    card.runtime_ref = {key: value for key, value in (card.runtime_ref or {}).items() if key not in CANCEL_KEYS}
    db.flush()
    sync_mission_board_status(db, run)
    _note(db, card, note)


def _missions_card(db: Any, run: Any) -> Optional[Any]:
    from core.models.core import BoardTask

    return db.query(BoardTask).filter(BoardTask.source_type == MISSION_CARD, BoardTask.orchestration_run_id == run.id,
                                      BoardTask.workspace_id == run.workspace_id).first()


def _note(db: Any, card: Any, note: str) -> None:
    """One note on the card's notes, by the mission, in the caller's transaction. The
    card is written first, and its notes read again after (the append is one jsonb
    statement, ``append_session_note``)."""
    from services.cli_host_service import append_session_note

    db.flush()
    append_session_note(db, task_id=card.id, workspace_id=card.workspace_id, note=note, by=NOTE_BY)
    db.expire(card, ["runtime_ref"])


__all__ = ["carries_the_missions_result", "missions_result", "reopen_the_missions_card"]
