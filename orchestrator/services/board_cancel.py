"""Cancelling a board ticket — one way, whoever asks (PRD-234 S1a; F116).

The board's cancel (``POST /api/v1/tasks/{id}/cancel``) was the only code that
knew how: terminal ``cancelled`` at once, the lease and the session credential
gone, ``cancel_requested_at`` on the ticket so a CLI host's next event batch
gets ``control: ["cancel"]`` and stops the session, and the board told. It is
lifted here so a cancelled playbook run can close its own session step tickets
the same way, and every cancel now says who and why (F015's shape:
``runtime_ref["cancelled"] = {"by", "reason", "at"}``).

F116 (run 4): two runs cancelled by the owner at 06:54 left their session step
tickets working — #725 was still in progress on a playbook nobody was running.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List

logger = logging.getLogger(__name__)

RUN_STEP_LIVE = ("inbox", "assigned", "in_progress", "blocked")


def cancel_board_ticket(db: Any, task: Any, *, by: str, reason: str) -> bool:
    """Cancel ``task`` unless it has finished. Commits and tells the board.
    True when it was cancelled here."""
    if task.status in ("done", "failed", "cancelled", "closed"):
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
    ref = dict(task.runtime_ref or {})
    ref["cancel_requested_at"] = now.isoformat()
    ref["cancelled"] = {"by": by, "reason": reason, "at": now.isoformat()}
    # PRD-245: the run is over, so its session credential is destroyed here too.
    clear_session_token(ref)
    task.runtime_ref = ref  # rebuild, never mutate in place (JSONB)
    # pg_notify is delivered when its transaction commits — so before the commit
    # (after it, the request's closing rollback would drop the NOTIFY).
    notify_board_event(
        db, workspace_id=task.workspace_id, task_id=task.id, status="cancelled", event="task_cancelled",
    )
    db.commit()
    logger.info("[BoardCancel] task %d cancelled (was %s) by %s — %s", task.id, previous, by, reason)
    return True


def cancel_run_step_tickets(db: Any, execution_id: str, *, by: str) -> List[int]:
    """F116: cancel the session step tickets a playbook run filed
    (``recipe:<run>:<step>``) that are still queued or being worked, each saying
    it was cancelled with the run. Returns the ticket ids cancelled. A step
    that already finished keeps its result."""
    from core.models.core import BoardTask

    tickets = (
        db.query(BoardTask)
        .filter(
            BoardTask.source_type == "recipe",
            BoardTask.source_id.like(f"recipe:{execution_id}:%"),
            BoardTask.status.in_(RUN_STEP_LIVE),
        )
        .order_by(BoardTask.id)
        .all()
    )
    cancelled = []
    for task in tickets:
        if cancel_board_ticket(db, task, by=by, reason=f"cancelled with run {execution_id}"):
            cancelled.append(task.id)
    return cancelled
