"""PRD-256 FX-011: what the owner's click (or their no) does to an agent's send on Auto's ticket.

The click runs the stored send once, through the same executor and the same gate: the
run's ticket rides in its context, so ``owner_only.asks_before_a_send`` finds Auto's ticket
again and claims the single-use grant (``_the_click``) before anything goes out. The outcome
is written on the ticket as a note; a session's ticket, parked on the card, goes back to
work the way an answered question sends it (``resume_on_answer``: blocked → assigned, and
the same session resumes with the outcome in its prompt). A board run's or a playbook's
ticket never left its status. A no fails the ticket with the reason.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from modules.tools.discovery.agent_sends import (
    CLICKER, FAILED, NOT_SENT, SENT, SESSION_LANE, now_said, ticket_row,
)

logger = logging.getLogger(__name__)

TRACE = "grant-resume-{grant_id}"
NO_ERROR = "no reason given"
DECLINED_ANSWER = "declined"
BLOCKED = "blocked"
# A ticket already ended this way is left as it is; a Done one fails: the send it closed on never went out.
ALREADY_ENDED = frozenset({"failed", "cancelled"})


async def send_once(db: Any, grant: Any, marker: Dict[str, Any]) -> Dict[str, Any]:
    """The stored send, run as the agent that asked, with its ticket in the context."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    details = grant.details if isinstance(grant.details, dict) else {}
    params = details.get("params")
    try:
        raw = await UnifiedToolExecutor(db).execute_tool(
            tool_name=str(details.get("action") or grant.tool_name),
            parameters=dict(params) if isinstance(params, dict) else {},
            agent_id=int(marker.get("agent_id") or grant.agent_id or 0),
            workspace_id=grant.workspace_id,
            trace_id=TRACE.format(grant_id=grant.id),
            caller_context=dict(marker.get("context") or {}),
        )
    except Exception as exc:  # noqa: BLE001 — logged; the click's result says it failed
        logger.exception("[agent_sends] the send on grant %s failed on the click", grant.id)
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    return raw if isinstance(raw, dict) else {"success": False, "error": "the executor returned no result"}


def sent_on_the_ticket(db: Any, grant: Any, marker: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """The click's outcome as a note on the ticket; a parked session goes back to work."""
    task = ticket_row(db, grant.workspace_id, marker.get("task_id"))
    if task is None:
        logger.warning("[agent_sends] grant %s: no ticket %s in workspace %s to tell",
                       grant.id, marker.get("task_id"), grant.workspace_id)
        return
    if summary.get("success"):
        note = SENT.format(at=now_said(), recipient=marker.get("recipient"), subject=marker.get("subject"))
    else:
        note = NOT_SENT.format(at=now_said(), error=summary.get("error") or NO_ERROR)
    _note(db, task, note)
    if marker.get("lane") == SESSION_LANE:
        db.refresh(task, ["runtime_ref"])  # with the note just appended, before the ledger is written whole
        _back_to_work(db, task, grant.id, note)


def fail_the_ticket(db: Any, grant: Any, marker: Dict[str, Any], reason: str) -> None:
    """The owner's no: the ticket fails with the reason."""
    task = ticket_row(db, grant.workspace_id, marker.get("task_id"))
    if task is None or task.status in ALREADY_ENDED:
        return
    if marker.get("lane") == SESSION_LANE:
        from services.cli_host_service import record_session_answer

        task.runtime_ref = record_session_answer(dict(task.runtime_ref or {}), grant_id=grant.id,
                                                 answer=DECLINED_ANSWER)
    task.status = FAILED
    task.error_message = reason
    task.completed_at = datetime.now(timezone.utc)
    task.blocked_at = None
    task.blocked_reason = None
    task.lease_until = None
    _board_event(db, task)
    logger.info("[agent_sends] ticket %s failed: %s", task.id, reason)


def _back_to_work(db: Any, task: Any, grant_id: Any, note: str) -> None:
    """The card is answered on the session's ledger. A ticket parked on THIS card, with no
    other question or plan still open, resumes (``resume_on_answer``); one parked or stopped
    for anything else keeps waiting, with the outcome recorded for its next turn."""
    from services.cli_host_service import open_session_asks, record_session_answer, resume_on_answer
    from services.session_plans import SEND_KIND, park_reason, session_plans

    ref = record_session_answer(dict(task.runtime_ref or {}), grant_id=grant_id, answer=note)
    still_open = open_session_asks(ref) or [plan for plan in session_plans(ref) if not plan.get("answered_at")]
    parked_on_it = task.blocked_reason == park_reason({"grant_id": grant_id, "kind": SEND_KIND})
    if still_open or (task.status == BLOCKED and not parked_on_it):
        task.runtime_ref = ref
        logger.info("[agent_sends] ticket %s: send card #%s answered; it waits on something else", task.id, grant_id)
        return
    resume_on_answer(db, task, ref, what=f"send card #{grant_id}")


def _note(db: Any, task: Any, note: str) -> None:
    from services.cli_host_service import append_session_note, publish_note_line

    append_session_note(db, task_id=task.id, workspace_id=task.workspace_id, note=note, by=CLICKER)
    publish_note_line(task.workspace_id, task.id, note)


def _board_event(db: Any, task: Any) -> None:
    from services.board_events import notify_board_event

    try:
        notify_board_event(db, workspace_id=str(task.workspace_id), task_id=task.id, status=task.status,
                           event="task_updated")
    except Exception:  # noqa: BLE001 — the ticket's status is written; the live board catches up
        logger.exception("[agent_sends] board event for ticket %s not sent", task.id)
