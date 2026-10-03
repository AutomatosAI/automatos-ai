"""F241 (night 7): a ticket tool changes only the ticket it was given, and says so on it.

Asked to cancel #0156 by its title, Auto moved #0014, last week's ticket, from
Closed to Done. It told the owner it had cancelled theirs, and nothing on #0014
recorded the move. ``guarded_and_recorded`` wraps Auto's ticket tools (status,
assign, edit):

- a closed ticket is refused, whole: a closed ticket is finished and filed, so a
  call that reaches one has the wrong ticket. Nothing is changed, and the refusal
  says to find the ticket by its number;
- every change that stands goes on the ticket's notes (``runtime_ref.session_notes``,
  which the board's ticket view shows): who made it, "in chat" when it came from
  one, and what changed ("Moved this from Inbox to Cancelled").

It sits under ``by_ticket_number``, so it sees ticket ids and its answer is
numbered. A bulk status call is guarded and recorded once, for all its tickets.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

STATUS, ASSIGN, EDIT = "status", "assign", "edit"
CLOSED, CANCELLED = "closed", "cancelled"
STATUS_NAMES = {"inbox": "Inbox", "assigned": "Assigned", "in_progress": "In progress", "review": "Review",
                "blocked": "Blocked", "done": "Done", "failed": "Failed", "cancelled": "Cancelled",
                "closed": "Closed"}
CLOSED_REFUSAL = ("{labels} {verb} closed: finished and filed, so it is not the ticket to change. Nothing was "
                  "changed. Find the ticket by its number (platform_list_tasks or the board snapshot) and use that.")


def guarded_and_recorded(kind: str) -> Callable[[Handler], Handler]:
    """Refuse a change to a closed ticket, and note every change that stands on its ticket."""
    def decorate(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            before = _snapshot(db, workspace_id, _targets(params))
            closed = [t for t in before.values() if t["status"] == CLOSED and not _closing(kind, params)]
            if closed:
                return {"success": False, "error": _closed_refusal(db, workspace_id, closed)}
            result = await handler(db, workspace_id, params)
            if before and isinstance(result, dict):    # a partly done bulk call still changed some
                _record(db, workspace_id, kind, before, result)
            return result
        return wrapped
    return decorate


def _targets(params: Mapping[str, Any]) -> List[int]:
    ids = params.get("task_ids") if params.get("task_ids") not in (None, []) else [params.get("task_id")]
    return [int(i) for i in ids or [] if _is_id(i)]


def _is_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and value.isdigit())


def _closing(kind: str, params: Mapping[str, Any]) -> bool:
    """A status call that leaves a ticket closed changes nothing about it."""
    return kind == STATUS and params.get("status") == CLOSED


def _snapshot(db: Session, workspace_id: Any, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Each ticket's status, agent and title, as they stand before the call."""
    if not ids:
        return {}
    from core.models.core import BoardTask

    rows = db.query(BoardTask.id, BoardTask.status, BoardTask.assigned_agent_id, BoardTask.title,
                    BoardTask.runtime_ref).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()
    return {r.id: {"id": r.id, "status": r.status, "agent": r.assigned_agent_id, "title": r.title,
                   "notes": _notes_on(getattr(r, "runtime_ref", None))} for r in rows}


def _notes_on(runtime_ref: Any) -> int:
    """How many notes a ticket carries (``runtime_ref.session_notes``)."""
    notes = runtime_ref.get("session_notes") if isinstance(runtime_ref, dict) else None
    return len(notes) if isinstance(notes, list) else 0


def _closed_refusal(db: Session, workspace_id: Any, closed: List[Dict[str, Any]]) -> str:
    from services.ticket_numbers import ticket_label_for

    labels = [f"{ticket_label_for(db, workspace_id, t['id'], capital=True)} ('{t['title']}')" for t in closed]
    return CLOSED_REFUSAL.format(labels=", ".join(labels), verb="is" if len(labels) == 1 else "are")


def _record(db: Session, workspace_id: Any, kind: str, before: Dict[int, Dict[str, Any]],
            result: Mapping[str, Any]) -> None:
    """One note per changed ticket. A note that can't be written never undoes the
    change: it is logged."""
    try:
        after = _snapshot(db, workspace_id, list(before))
        notes = {tid: _what_changed(db, kind, before[tid], after.get(tid), result) for tid in before}
        _append(db, workspace_id, {tid: note for tid, note in notes.items() if note})
    except Exception:
        logger.exception("[ticket_changes] the change to %s stands, but its note was not written", sorted(before))


def _what_changed(db: Session, kind: str, was: Mapping[str, Any], now: Optional[Mapping[str, Any]],
                  result: Mapping[str, Any]) -> Optional[str]:
    if now is None:
        return None
    if kind == STATUS and now["status"] != was["status"]:
        if now["status"] == CANCELLED and now["notes"] > was["notes"]:
            return None     # F273: the cancel noted itself on the card (services/cancel_notes), once is enough
        return f"Moved this from {_status(was['status'])} to {_status(now['status'])}"
    if kind == ASSIGN and now["agent"] != was["agent"]:
        return f"Gave this to {_agent_name(db, now['agent']) or 'no one'}"
    fields = [f for f in result.get("updated") or [] if isinstance(f, str)] if kind == EDIT else []
    return f"Changed its {', '.join(fields)}" if fields else None


def note_change(db: Session, workspace_id: Any, task_id: int, what: str) -> None:
    """One change another tool made, noted on its ticket as these tools note theirs
    ("Auto · Gave this a new brief and sent it back, in chat."). A note that can't be
    written never undoes the change: it is logged."""
    try:
        _append(db, workspace_id, {task_id: what})
    except Exception:
        logger.exception("[ticket_changes] the change to %s stands, but its note was not written", task_id)


def _append(db: Session, workspace_id: Any, notes: Mapping[int, str]) -> None:
    """The notes, in a session of their own: the tool's change is already committed,
    and its own transaction is left exactly as the tool left it."""
    if not notes:
        return
    from core.database.database import get_db_session
    from services.cli_host_service import append_session_note

    by, where = _actor(db)
    with get_db_session() as notes_db:      # commits on leaving, rolls back on an error
        for task_id, note in notes.items():
            append_session_note(notes_db, task_id=task_id, workspace_id=workspace_id, note=f"{note}{where}.", by=by)


def _actor(db: Session) -> Tuple[str, str]:
    """Who made the change, as the ticket's notes say it, and ", in chat" when a chat
    turn made it (the usage scope names the lane and the agent)."""
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    scope = current_usage_scope() or {}
    name = _agent_name(db, scope.get("agent_id")) if scope.get("agent_id") else None
    in_chat = scope.get("request_type") == LANE_CHAT
    return (name or ("Auto" if in_chat else "An agent")), (", in chat" if in_chat else "")


def _agent_name(db: Session, agent_id: Any) -> Optional[str]:
    from core.models.core import Agent

    row = db.query(Agent.name).filter(Agent.id == agent_id).first() if agent_id else None
    return row.name if row else None


def _status(value: Any) -> str:
    return STATUS_NAMES.get(str(value), str(value))


__all__ = ["ASSIGN", "CLOSED_REFUSAL", "EDIT", "STATUS", "guarded_and_recorded", "note_change"]
