"""F243 (night 7): a redo on a playbook's or a mission's card runs on that card, or is refused up front.

Night 7: six of six rejected playbook cards never ran again (#0106, #0112,
#0149, #0070); Run now and a re-brief said "started" when nothing started
(#0112, #0123, #0095); a rejected mission step never ran again (#0083.1,
#0105.3, #0119.3); a re-brief on a finished mission's step pointed at the
mission's page, which can only re-run the whole mission (#0126.2). The board
put the card back in Assigned with the owner's words, but the board's
dispatcher never runs a playbook's card or a mission's step.

Now a Reject, a re-brief, Run now (or a drag to In progress) on a playbook's
card runs the playbook again on that card, the owner's words in every step's
prompt, as an answered question already does (F140). A Reject or re-brief on
a step of a mission that is still running is the mission's to redo: the step
goes back to it for revision with the owner's words. Everything else (a step
of a mission that has ended, a step a Claude Code session ran) is refused
before anything changes, saying why and what to do. (The mission's own card
is the mission's to decide: PRD-252 D6 refuses it before any of this.)

F284 (night 8): a step of a mission that completed or failed is redone too: its
mission opens again (services/mission_reopen). The redo decides under the
mission's row lock, which the mission's end takes as well, so a send-back that
lands as the mission completes (#0250.1, 2 s after) waits for the end and opens
it again, instead of sticking "in progress" in a mission that never runs it.
Every refusal that stays names a button that exists.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from services.run_cancel import GOAL_SHOWN_CHARS, MISSION_STEP, is_playbook_card, mission_run_of, playbook_run_of

logger = logging.getLogger(__name__)

SESSION_STEP = "mission"   # a mission step a Claude Code session ran (services/cli_ticket_lane)
LIVE_RUN_STATUSES = ("pending", "running")
# F284: what the board's refusal to run a mission's ticket (api/board_tasks.mission_runs_it) tells
# the owner to do with a step instead: #0250.1's Run now said "Retry … from the mission", which no
# page has. (A mission's own card keeps its words: that refusal is F291's.)
BOARD_REDO = {
    MISSION_STEP: "Send it back with Reject (once it is in Review or Done) to have it redone by its mission",
    SESSION_STEP: "Cancel the mission and use Re-run on its page to start it again as a new mission",
}
MISSION_GONE = "{label} belongs to a mission that can no longer be found, so it can't run again here."
# F284: a step a Claude Code session ran stays its session's; the page's Re-run (on an
# ended mission) is what runs it again.
SESSION_STEP_REFUSAL = ("{label} is a step of the mission \"{goal}\" that a Claude Code session ran, so the "
                        "board can't redo it. Cancel the mission on its page ({where}) and use Re-run there to "
                        "start it again as a new mission.")
# The step states a revision can start from: checked or being checked (VERIFYING, VERIFIED),
# or finished and about to be checked (COMPLETED goes through VERIFYING).
REDOABLE_STEP_STATES = ("completed", "verifying", "verified")
# What a redone step is shown as its last attempt when that attempt left no text: the
# dispatcher's revision prompt, which carries the owner's words, needs one to show.
NO_LAST_ATTEMPT = "(Your last attempt left no text.)"


class RedoTaken(Exception):
    """Another redo of the same card started first (a second click, a drag racing a
    button): this one changed nothing."""


class RedoRefused(RedoTaken):
    """F284: read under its mission's lock, the redo can't run after all (the mission
    was cancelled meanwhile, the step moved on, or the mission is still being written):
    nothing changed. A RedoTaken, so the board answers 409 with its words."""


def takes_its_own_redo(task: Any) -> bool:
    """Whether a redo of ``task`` starts here (a playbook's card, a mission's step)
    rather than through the board's dispatcher (a plain ticket)."""
    return is_playbook_card(task) or getattr(task, "source_type", None) == MISSION_STEP


def redo_refusal(db: Any, task: Any) -> Optional[str]:
    """Why ``task`` cannot be redone on its card, said before anything changes;
    None when it can (or when the board's dispatcher runs it)."""
    if is_playbook_card(task):
        return _playbook_refusal(db, task)
    if getattr(task, "source_type", None) in (MISSION_STEP, SESSION_STEP):  # the mission's own card: D6, upstream
        return _mission_refusal(db, task)
    return None


def _playbook_refusal(db: Any, task: Any) -> Optional[str]:
    from services.playbook_run_refusal import run_refusal
    from services.ticket_numbers import ticket_label

    run = playbook_run_of(db, task)
    playbook = _playbook_of(db, run) if run is not None else None
    if playbook is None:
        return (f"{ticket_label(task, capital=True)}'s playbook run can no longer be found, so it can't run "
                "again here. Run the playbook from its page.")
    if run.status in LIVE_RUN_STATUSES:
        return (f"{ticket_label(task, capital=True)}'s playbook is still running. Cancel it first, or wait "
                "for it to finish.")
    return run_refusal(playbook)  # F270: a step with no agent; #0183's Run now ran 103 again, and failed again


def _mission_refusal(db: Any, task: Any) -> Optional[str]:
    from services.mission_reopen import refusal_for
    from services.ticket_numbers import ticket_label

    run = mission_run_of(db, task)
    label = ticket_label(task, capital=True)
    if run is None:
        return MISSION_GONE.format(label=label)
    goal = (run.goal or "")[:GOAL_SHOWN_CHARS]
    where = f"/missions/{run.id}"
    if task.source_type == SESSION_STEP:
        return SESSION_STEP_REFUSAL.format(label=label, goal=goal, where=where)
    # F284: a mission that completed or failed opens again for the redo
    return refusal_for(run, label) or _step_refusal(db, task, label)


def _step_refusal(db: Any, task: Any, label: str) -> Optional[str]:
    from core.models.orchestration import OrchestrationTask

    step = db.get(OrchestrationTask, task.orchestration_task_id) if task.orchestration_task_id else None
    if step is None or step.state not in REDOABLE_STEP_STATES:
        state = step.state if step is not None else "unknown"
        return f"{label} is still being worked by its mission (step {state}); send it back once it is done."
    return None


def start_redo(db: Any, task: Any, *, by: str) -> str:
    """Start the redo ``redo_refusal`` cleared, on the same card, with the
    owner's words the card carries (``review_feedback``). Commits. Returns what
    started, in words."""
    if is_playbook_card(task):
        return _redo_playbook(db, task)
    return _redo_mission_step(db, task, by=by)


def owner_words(task: Any) -> Optional[str]:
    """What the redo is told: the owner's words, the draft they sent back and every
    correction (services/ticket_redo.redo_block), or the brief they agreed in a
    discussion, written out (a playbook's steps and a mission's step have their
    own prompts, which never show the card's description)."""
    from services.ticket_redo import AGREED_BRIEF_BLOCK, BRIEF_AGREED, redo_block

    if getattr(task, "review_feedback", None) == BRIEF_AGREED:
        return f"{AGREED_BRIEF_BLOCK}\n\n{task.description or ''}".rstrip()
    return redo_block(task)


def _playbook_of(db: Any, run: Any) -> Any:
    from core.models.core import WorkflowTemplate

    return db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == run.recipe_id, WorkflowTemplate.workspace_id == run.workspace_id).first()


def _redo_playbook(db: Any, card: Any) -> str:
    """The playbook runs again from step 1 on ``card`` (F140's rerun, with the
    owner's words in every step's prompt); a watch on the run follows it."""
    from services.playbook_owner_ask import ANSWERS_KEY, REDO_KEY
    from services.ticket_numbers import ticket_label
    from services.watch_rerun import TRIGGERED_BY_HUMAN, create_rerun_execution, launch_execution

    original = playbook_run_of(db, card)
    if not _hold_the_card(db, card, original.execution_id):
        db.rollback()
        raise RedoTaken(f"{ticket_label(card, capital=True)} is already being run again.")
    rerun = create_rerun_execution(db, _playbook_of(db, original), original, triggered_by=TRIGGERED_BY_HUMAN)
    words = owner_words(card)
    earlier = (original.execution_metadata or {}).get(ANSWERS_KEY)
    rerun.execution_metadata = {**(rerun.execution_metadata or {}),
                                **({ANSWERS_KEY: earlier} if earlier else {}),
                                **({REDO_KEY: words} if words else {})}
    _point_card_at(card, rerun.execution_id)
    _follow_with_the_watch(db, original, rerun)
    db.commit()  # the engine's task opens its own session
    launch_execution(rerun)
    logger.info("[F243] %s: playbook run %s runs again as %s", ticket_label(card), original.execution_id,
                rerun.execution_id)
    return f"{ticket_label(card, capital=True)}'s playbook is running again on this card."


def _hold_the_card(db: Any, card: Any, run_id: str) -> bool:
    """One redo of a card at a time (review of #886: two Run Now clicks launched
    two reruns). The card's row is locked without waiting (a wait would hold the
    event loop, F105) and must still show ``run_id``; a second request finds it
    locked, or already showing the first one's rerun."""
    from core.models.core import BoardTask

    held = (db.query(BoardTask.source_id).filter(BoardTask.id == card.id)
            .with_for_update(skip_locked=True).first())
    return held is not None and held.source_id == run_id


def _point_card_at(card: Any, execution_id: str) -> None:
    """One card for the whole job: the card follows the new run, starting clean."""
    card.source_id = execution_id
    card.status = "in_progress"
    card.started_at = datetime.now(timezone.utc)
    card.completed_at = card.error_message = card.result = card.lease_until = None
    card.blocked_at = card.blocked_reason = None
    card.review_feedback = None  # the run carries the owner's words now
    card.planning_data = {**(card.planning_data or {}), "execution_id": execution_id}


def _follow_with_the_watch(db: Any, original: Any, rerun: Any) -> None:
    from services.watch_service import WatchService

    watch = WatchService.find_live_watch(db, workspace_id=original.workspace_id,
                                         target_type="playbook_execution", target_id=original.execution_id)
    if watch is not None:
        WatchService.follow(db, watch, new_target_type="playbook_execution", new_target_id=rerun.execution_id,
                            reason="the owner sent the run back")


def _starts_clean(card: Any) -> None:
    """F267 (night 7b): a step sent back showed In progress with no start time and the
    draft the owner had just rejected, for minutes. Its card starts clean, as a
    playbook's redo does: started now, the old draft off the card (Reject keeps it
    in the card's history)."""
    card.started_at = datetime.now(timezone.utc)
    card.completed_at = card.result = None


def _redo_mission_step(db: Any, card: Any, *, by: str) -> str:
    """The step goes back to its mission for revision, with the owner's words and
    the output they sent back (the dispatcher's revision prompt reads both). F284:
    decided under the mission's row lock; a mission that ended opens again. A lock
    not had in time refuses the redo with nothing changed (RedoRefused)."""
    from sqlalchemy.exc import OperationalError

    from services.mission_reopen import busy_refusal, lost_the_lock
    from services.ticket_numbers import ticket_label

    label = ticket_label(card, capital=True)
    run = mission_run_of(db, card)
    try:
        _redo_under_the_lock(db, card, run, by=by, label=label)
    except OperationalError as exc:
        if not lost_the_lock(exc):
            raise
        db.rollback()
        logger.info("[F284] %s not sent back: its mission %s was being written", label, getattr(run, "id", None))
        raise RedoRefused(busy_refusal(run, label)) from exc
    return f"{label} went back to its mission to be redone."


def _redo_under_the_lock(db: Any, card: Any, run: Any, *, by: str, label: str) -> None:
    """The redo, its step and its mission locked (at most ``LOCK_WAIT`` each); the
    mission opens again when it ended, and what was built from the step runs again
    after its redo (F286). Commits."""
    from modules.coordination.redo_dependents import its_redo_waits, rerun_what_builds_on
    from services.mission_reopen import reopen, touched

    step, run = _held_for_the_redo(db, card, run, label)
    reopen(db, run, by=by, label=label)
    _back_for_revision(db, card, step, by=by)
    rerun_what_builds_on(db, step, by=by, label=label)  # F286: what was built from it runs again
    its_redo_waits(db, step, by=by, label=label)  # F286: a redo whose inputs are redone waits for them
    touched(run)  # a tick that read the mission before this redo can't end it on what it read
    db.commit()


def _held_for_the_redo(db: Any, card: Any, run: Any, label: str) -> tuple:
    """The step and its mission, locked and read again, in that order (the coordinator
    writes a step before it ends its mission). Refused (RedoRefused, nothing changed)
    when, read under the lock, the redo can't run after all."""
    from core.models.orchestration import OrchestrationTask
    from services.mission_reopen import locked_run, refusal_for, waits_briefly

    if run is None:
        raise RedoRefused(MISSION_GONE.format(label=label))
    waits_briefly(db)
    step = db.get(OrchestrationTask, card.orchestration_task_id, populate_existing=True, with_for_update=True)
    locked = locked_run(db, run)
    if locked is None:
        refused = MISSION_GONE.format(label=label)
    else:
        refused = refusal_for(locked, label) or _step_refusal(db, card, label)
    if refused:
        db.rollback()
        raise RedoRefused(refused)
    return step, locked


def _back_for_revision(db: Any, card: Any, step: Any, *, by: str) -> None:
    """The step to RETRYING with the owner's words and the output they sent back."""
    from core.models.orchestration_enums import ActorType, FailureReasonCode, TaskState
    from services.orchestration_board_bridge import sync_board_status
    from services.orchestration_state import transition_task

    from modules.coordination.owner_checks import sent_back

    sent_back(db, step, by=by)  # F242: a step held for the owner's check; its mission carries on
    attempt = (step.attempt_number or 0) + 1
    step.failure_reason_code = FailureReasonCode.VERIFICATION_REJECT.value
    step.attempt_number = attempt
    step.input_context = {**(step.input_context or {}),
                          "previous_output": step.output or card.result or NO_LAST_ATTEMPT,
                          "verification_feedback": {"attempt": attempt, "reasoning": owner_words(card) or "",
                                                    "scores": {}, "failures": []}}
    if step.state == TaskState.COMPLETED.value:
        transition_task(db=db, task=step, new_state=TaskState.VERIFYING, actor_type=ActorType.HUMAN, actor_id=by,
                        reason="sent back by the owner before its check")
    transition_task(db=db, task=step, new_state=TaskState.RETRYING, actor_type=ActorType.HUMAN, actor_id=by,
                    reason="the owner sent it back")
    card.review_feedback = None  # the step carries the owner's words now
    _starts_clean(card)
    sync_board_status(db, step)
