"""Stopping a board ticket's run — one way, whoever asks (PRD-234 S1a; F116; F224).

The board's cancel (``POST /api/v1/tasks/{id}/cancel``) was the only code that
knew how: terminal ``cancelled`` at once, the lease and the session credential
gone, ``cancel_requested_at`` on the ticket so a CLI host's next event batch
gets ``control: ["cancel"]`` and stops the session, and the board told. It is
lifted here so a cancelled playbook run can close its own session step tickets
the same way, and every cancel now says who and why (F015's shape:
``runtime_ref["cancelled"] = {"by", "reason", "at"}``).

F116 (run 4): two runs cancelled by the owner at 06:54 left their session step
tickets working — #725 was still in progress on a playbook nobody was running.

F224 (2 Oct, Gerard): the same for every parent — when a task, playbook run,
mission or routine is cancelled, fails or dies, every runtime session it
started stops. ``stop_ticket_run`` is the one path; it does not commit, so a
caller inside a larger transaction (a mission's state change) keeps it whole.

F245 (night 7): a failed ticket can be cancelled too, which closes it: it
waits in Needs you until the owner deals with it (F246).

F273 (night 7b): a stopped ticket also says who and when among its notes, the
list the ticket view shows (services/cancel_notes.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List

from services.cancel_notes import CANCELLED_THIS, WITH_ITS_RUN, with_cancel_recorded

logger = logging.getLogger(__name__)

RUN_STEP_LIVE = ("inbox", "assigned", "in_progress", "blocked")
FINISHED = ("done", "failed", "cancelled", "closed")
# What no cancel moves: a failed ticket can still be cancelled, which closes it (F245).
UNCANCELLABLE = ("done", "cancelled", "closed")
CANCEL_REQUESTED_KEY = "cancel_requested_at"
# A mission step's card the session lane runs carries ``source_id`` "mission:…"
# (services.cli_ticket_lane.MISSION_SOURCE_TYPE) and its run's id: only those
# have a session — the step's own card it claimed, or the ticket it filed.
MISSION_LANE_SOURCE_PREFIX = "mission:"
# The mission states that end it without finishing: its sessions stop (F224).
MISSION_STOP_REASONS = {"cancelled": "the mission was cancelled", "failed": "the mission failed"}
MISSION_STOPPED_BY = "the mission"
# A playbook run that ended without finishing (F224): failed, or killed with the
# backend (the executor marks it cancelled on any CancelledError, F116).
PLAYBOOK_RUN_BY = "the playbook run"
RUN_FAILED_REASON = "the run failed"
RUN_DIED_REASON = "the run stopped when the backend stopped"
# A routine switched off (F224): a scheduled task cancelled or paused, a
# heartbeat switched off. Each firing stands alone, so the one in flight stops.
ROUTINE_OFF_REASONS = {"cancelled": "the routine was cancelled", "paused": "the routine was paused"}
HEARTBEAT_OFF_REASON = "the heartbeat was switched off"
ROUTINE_BY = "the routine"


def stop_ticket_run(db: Any, task: Any, *, by: str, reason: str, note: str = CANCELLED_THIS) -> bool:
    """End ``task``'s run unless it is done or already closed: terminal
    ``cancelled``, the lease and the session credential gone,
    ``cancel_requested_at`` so the CLI host stops the session at its next event
    batch, the board told. The cancel goes on record with its note (F273): who,
    why and when; ``note`` is what a person or an agent did (services/cancel_notes).
    Does NOT commit. True when the ticket was stopped here."""
    if task.status in UNCANCELLABLE:
        return False
    from services.board_events import notify_board_event
    from services.cli_host_service import clear_session_token

    previous = task.status
    now = datetime.now(timezone.utc)
    task.status = "cancelled"
    task.completed_at = now
    task.lease_until = None
    if previous == "blocked":
        task.blocked_at = None
        task.blocked_reason = None
    ref = with_cancel_recorded(db, task, by=by, reason=reason, at=now, note=note)
    ref[CANCEL_REQUESTED_KEY] = now.isoformat()
    # PRD-245: the run is over, so its session credential is destroyed here too.
    clear_session_token(ref)
    task.runtime_ref = ref  # rebuild, never mutate in place (JSONB)
    # pg_notify is delivered when its transaction commits — so before the commit
    # (after it, the request's closing rollback would drop the NOTIFY).
    notify_board_event(
        db, workspace_id=task.workspace_id, task_id=task.id, status="cancelled", event="task_cancelled",
    )
    db.flush()
    logger.info("[BoardCancel] task %d stopped (was %s) by %s — %s", task.id, previous, by, reason)
    return True


def call_off_done(db: Any, task: Any, *, by: str, reason: str) -> bool:
    """F294 (night 8): a Done card the owner calls off is Cancelled, with the cancel on
    record and its note (F273), as any cancel. Nothing runs a Done card, so nothing
    stops, and its answer stays on it. #0422: Auto approved the card when asked to
    cancel it; the board's Cancel then answered ``applied: false`` with no words, and a
    drag to Cancelled moved it with no note. Does NOT commit. False unless it was Done."""
    if task.status != "done":
        return False
    from services.board_events import notify_board_event

    now = datetime.now(timezone.utc)
    task.status = "cancelled"
    task.completed_at = now
    task.runtime_ref = with_cancel_recorded(db, task, by=by, reason=reason, at=now)
    notify_board_event(
        db, workspace_id=task.workspace_id, task_id=task.id, status="cancelled", event="task_cancelled",
    )
    db.flush()
    logger.info("[BoardCancel] done task %d called off by %s — %s", task.id, by, reason)
    return True


def cancel_board_ticket(db: Any, task: Any, *, by: str, reason: str) -> bool:
    """Cancel ``task`` unless it is done or already closed. Commits and tells the
    board. True when it was cancelled here."""
    if not stop_ticket_run(db, task, by=by, reason=reason):
        return False
    db.commit()
    return True


def run_step_tickets(db: Any, execution_id: str) -> List[Any]:
    """The session step tickets a playbook run filed (``recipe:<run>:<step>``)
    that are still queued or being worked."""
    from core.models.core import BoardTask

    return (
        db.query(BoardTask)
        .filter(
            BoardTask.source_type == "recipe",
            BoardTask.source_id.like(f"recipe:{execution_id}:%"),
            BoardTask.status.in_(RUN_STEP_LIVE),
        )
        .order_by(BoardTask.id)
        .all()
    )


def stop_run_step_tickets(db: Any, execution_id: str, *, by: str, reason: str) -> List[int]:
    """Stop every live step ticket of a playbook run (F116 for a cancel, F224
    for a run that failed or died). Does NOT commit. Returns the tickets
    stopped; a step that already finished keeps its result."""
    return [task.id for task in run_step_tickets(db, execution_id)
            if stop_ticket_run(db, task, by=by, reason=reason, note=WITH_ITS_RUN)]


def live_mission_step_cards(db: Any, run_id: Any) -> List[Any]:
    """The cards of ``run_id``'s steps that the session lane runs (a Claude Code
    session works them, or will once claimed) and that have not finished."""
    from core.models.core import BoardTask

    return (
        db.query(BoardTask)
        .filter(
            BoardTask.orchestration_run_id == run_id,
            BoardTask.source_id.like(f"{MISSION_LANE_SOURCE_PREFIX}%"),
            BoardTask.status.in_(RUN_STEP_LIVE),
        )
        .order_by(BoardTask.id)
        .all()
    )


def stop_mission_sessions(db: Any, run: Any) -> List[int]:
    """F224: a mission that ended without finishing — cancelled by its owner, or
    failed (a step, the reconciler, a budget, a crash) — leaves no session
    working one of its steps, and no queued step card for the lane to claim
    later. Does NOT commit: the mission's own state change owns the
    transaction. Returns the cards stopped; nothing for any other state."""
    reason = MISSION_STOP_REASONS.get(str(getattr(run, "state", "") or ""))
    if reason is None:
        return []
    stopped: List[int] = []
    # Each read and write runs in its own SAVEPOINT: a statement that fails there
    # rolls back to it, never the caller's transaction, which holds the mission's
    # own state change (as F094's notes did).
    try:
        with db.begin_nested():
            cards = live_mission_step_cards(db, run.id)
    except Exception:
        logger.exception("[BoardCancel] could not find the step cards of mission %s to stop", run.id)
        return stopped
    for card in cards:
        try:
            with db.begin_nested():
                ended = stop_ticket_run(db, card, by=MISSION_STOPPED_BY, reason=reason)
        except Exception:
            logger.exception("[BoardCancel] could not stop step card #%s of mission %s", card.id, run.id)
            continue
        if ended:
            stopped.append(card.id)
    return stopped


def stop_routine_sessions(db: Any, *, workspace_id: Any, source_type: str, source_pattern: str,
                          reason: str) -> List[int]:
    """F224: the live tickets a routine filed (``source_type`` and a ``source_id``
    LIKE ``source_pattern``) stop with the routine. Commits. Returns the tickets
    stopped."""
    from core.models.core import BoardTask

    tickets = (
        db.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.source_type == source_type,
            BoardTask.source_id.like(source_pattern),
            BoardTask.status.in_(RUN_STEP_LIVE),
        )
        .order_by(BoardTask.id)
        .all()
    )
    stopped = [task.id for task in tickets if stop_ticket_run(db, task, by=ROUTINE_BY, reason=reason)]
    if stopped:
        db.commit()
    return stopped


def stop_scheduled_task_sessions(db: Any, workspace_id: Any, task_id: int, new_status: str) -> List[int]:
    """A scheduled task cancelled or paused: the firing in flight (``task:<id>:<when>``,
    services.cli_ticket_lane.source_id_for) stops with it. Nothing for any other status."""
    reason = ROUTINE_OFF_REASONS.get(new_status)
    if reason is None:
        return []
    return stop_routine_sessions(db, workspace_id=workspace_id, source_type="scheduled_task",
                                 source_pattern=f"task:{int(task_id)}:%", reason=reason)


def stop_heartbeat_sessions(db: Any, workspace_id: Any, agent_id: int) -> List[int]:
    """An agent's heartbeat switched off: its open heartbeat ticket (``agent:<id>``,
    one per agent) stops with it."""
    return stop_routine_sessions(db, workspace_id=workspace_id, source_type="heartbeat",
                                 source_pattern=f"agent:{int(agent_id)}", reason=HEARTBEAT_OFF_REASON)


__all__ = [
    "CANCEL_REQUESTED_KEY", "FINISHED", "HEARTBEAT_OFF_REASON", "MISSION_STOP_REASONS", "PLAYBOOK_RUN_BY",
    "ROUTINE_OFF_REASONS", "RUN_DIED_REASON", "RUN_FAILED_REASON", "RUN_STEP_LIVE", "UNCANCELLABLE",
    "call_off_done", "cancel_board_ticket", "live_mission_step_cards", "run_step_tickets", "stop_heartbeat_sessions",
    "stop_mission_sessions", "stop_routine_sessions", "stop_run_step_tickets", "stop_scheduled_task_sessions",
    "stop_ticket_run",
]
