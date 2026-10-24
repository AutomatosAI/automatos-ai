"""PRD-266 US-012a (§4.4, D22, K10) — 'Ready for review': a ticket that entered Review is announced.

``core/models/review_entries.py`` (every ORM writer) and the dispatch sweeper
(``services/board_dispatcher._notify_swept``, its raw UPDATE) call :func:`announce` after their
commit with each ticket's ``(id, workspace_id)``. :func:`dispatch_entries` runs on a session of
its own (``send_soon``): it re-reads each ticket in its workspace and raises ``review_pending``
through the ``NotificationDispatcher`` (in-app by default; one of the three push types, FR-7),
unless the ticket is gone, has left Review since, or is a Mission's own card. A Mission's card
shows Review while its Mission awaits plan approval or a person, which ``mission_plan_ready`` and
the question or approval already announce: one push, not two. A Mission step's card is announced.

A notice is a courtesy, never part of the write: nothing here raises.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from uuid import UUID

from core.models.core import Agent, BoardTask
from core.models.review_entries import REVIEW
from services.notification_events.soon import send_soon
from services.ticket_numbers import ticket_label

logger = logging.getLogger(__name__)

EVENT_TYPE = "review_pending"
LINK_TYPE = "task"
STATUS = "action_required"
TITLE_PREFIX = "Ready for review: "
TITLE_MAX_CHARS = 200
ELLIPSIS = "…"
# A Mission's own card (services/orchestration_board_bridge.py): its Review is its Mission's wait.
MISSION_CARD_SOURCE = "orchestration"

ReviewEntry = Tuple[int, str]  # (ticket id, workspace id)


def announce(entries: Iterable[ReviewEntry]) -> None:
    """Raise ``review_pending`` for each ticket through ``send_soon`` (a task on the running loop,
    inline without one). Never raises."""
    batch = tuple(entries)
    if batch:
        send_soon(dispatch_entries(batch), what=EVENT_TYPE)


def _open_session() -> Any:
    from core.database.database import SessionLocal

    return SessionLocal()


def _dispatcher(db: Any, workspace_id: str) -> Any:
    from core.services.notification_dispatcher import NotificationDispatcher

    return NotificationDispatcher(db, workspace_id)


async def dispatch_entries(entries: Sequence[ReviewEntry]) -> None:
    """Re-read each ticket on a session of its own and dispatch ``review_pending`` for each one
    still in Review; one commit. Never raises: a failure is logged."""
    try:
        db = _open_session()
    except Exception:
        logger.exception("[review_pending] no session for the notices of %d ticket(s)", len(entries))
        return
    try:
        for task_id, workspace_id in entries:
            await _announce_one(db, task_id, workspace_id)
        db.commit()
    except Exception:
        logger.exception("[review_pending] the notices of %d ticket(s) were not committed", len(entries))
    finally:
        db.close()


async def _announce_one(db: Any, task_id: int, workspace_id: str) -> None:
    """One ticket's notice, unless it is gone, out of Review, or a Mission's own card. Never raises.

    Each ticket runs in a savepoint: a failed write (the dispatcher logs and goes on) takes back
    only that ticket's notice, never the batch's other notices or its one commit.
    """
    try:
        with db.begin_nested():
            task = _ticket_in_review(db, task_id, workspace_id)
            if task is not None:
                await _dispatcher(db, str(workspace_id)).dispatch(**_notice(db, task))
    except Exception:
        logger.exception("[review_pending] the notice of ticket %s was not sent", task_id)


def _ticket_in_review(db: Any, task_id: int, workspace_id: str) -> Optional[BoardTask]:
    """The ticket, read in its workspace, when it is still in Review and not a Mission's own card."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id, BoardTask.workspace_id == UUID(str(workspace_id)),
    ).first()
    if task is None or task.status != REVIEW or task.source_type == MISSION_CARD_SOURCE:
        return None
    return task


def _notice(db: Any, task: BoardTask) -> Dict[str, Any]:
    """The dispatch's arguments: linked to the ticket, named by its label, its agent when it has one."""
    agent_name = _agent_name(db, task)
    return {
        "event_type": EVENT_TYPE,
        "title": _title(task.title),
        "message": ticket_label(task, capital=True),
        "link_type": LINK_TYPE,
        "link_id": str(task.id),
        "agent_id": task.assigned_agent_id if agent_name else None,
        "agent_name": agent_name,
        "status": STATUS,
    }


def _agent_name(db: Any, task: BoardTask) -> Optional[str]:
    """The name of the ticket's assigned agent, when that agent is the workspace's."""
    if task.assigned_agent_id is None:
        return None
    return db.query(Agent.name).filter(
        Agent.id == task.assigned_agent_id, Agent.workspace_id == task.workspace_id,
    ).scalar()


def _title(title: Optional[str]) -> str:
    """'Ready for review: <title>', at most ``TITLE_MAX_CHARS`` characters."""
    full = f"{TITLE_PREFIX}{title or ''}"
    if len(full) <= TITLE_MAX_CHARS:
        return full
    return full[: TITLE_MAX_CHARS - len(ELLIPSIS)] + ELLIPSIS
