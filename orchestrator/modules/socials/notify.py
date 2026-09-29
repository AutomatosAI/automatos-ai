"""PRD-251 S2.3 (US-206): approvers hear when a post needs them.

A post ENTERS ``needs_approval`` three ways: a person or an agent submits it, a
render finishes (Wave 1, S1.1c), or an edit voids its approval (D6). Each path
calls :func:`notify_if_entered` after its commit, and one ``approval_pending``
notification goes out through the platform's ``NotificationDispatcher``, linked
to the post (``link_type='social_post'``: the bell opens
``/deliverables?tab=socials&post=<id>``). Any other transition, needs_approval to
needs_approval included, sends nothing.

A notification is a courtesy, never part of the write: it runs after the commit,
on its own session, and a failure is logged, never raised into the request or
the render (the precedent is ``services/board_approval.py``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)

NEEDS_APPROVAL = "needs_approval"
EVENT_TYPE = "approval_pending"
LINK_TYPE = "social_post"
STATUS = "action_required"
TITLE_PREFIX = "Social post needs approval: "

# Scheduled dispatches, held until done so the loop does not drop them.
_PENDING: Set["asyncio.Task[None]"] = set()


def entered_review(before: Optional[str], after: Optional[str]) -> bool:
    """Whether a write moved a post INTO ``needs_approval`` (from anything else)."""
    return after == NEEDS_APPROVAL and before != NEEDS_APPROVAL


def _default_session_factory() -> Callable[[], Any]:
    from core.database.database import SessionLocal

    return SessionLocal


def _dispatcher(db: Any, workspace_id: str) -> Any:
    from core.services.notification_dispatcher import NotificationDispatcher

    return NotificationDispatcher(db, workspace_id)


async def dispatch_approval_pending(
    workspace_id: UUID | str,
    post_id: UUID | str,
    title: str,
    *,
    session_factory: Optional[Callable[[], Any]] = None,
) -> None:
    """Send the ``approval_pending`` notification for one post, on its own session.
    Never raises: a failure is logged."""
    try:
        db = (session_factory or _default_session_factory())()
    except Exception:
        logger.exception("[Socials] no session for the approval notice of post %s", post_id)
        return
    try:
        await _dispatcher(db, str(workspace_id)).dispatch(
            event_type=EVENT_TYPE,
            title=f"{TITLE_PREFIX}{title}",
            link_type=LINK_TYPE,
            link_id=str(post_id),
            status=STATUS,
        )
    except Exception:
        logger.exception("[Socials] the approval notice of post %s was not sent", post_id)
    finally:
        db.close()


def notify_approval_pending(workspace_id: UUID | str, post_id: UUID | str, title: str) -> None:
    """Send the notification without holding up the caller: on the running loop
    as a tracked task, or inline when there is none (a threadpool route, a
    script). Never raises."""
    try:
        coro = dispatch_approval_pending(workspace_id, post_id, title)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        task = loop.create_task(coro)
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    except Exception:
        logger.exception("[Socials] the approval notice of post %s could not be scheduled", post_id)


def notify_if_entered(before: Optional[str], post: Any) -> bool:
    """After a committed write: notify approvers when it moved ``post`` into
    ``needs_approval``. ``True`` when a notification was sent."""
    if not entered_review(before, getattr(post, "status", None)):
        return False
    notify_approval_pending(post.workspace_id, post.id, post.title or "")
    return True
