"""PRD-251B Wave 2 (B7, B11; US-B205, US-B206): what a plan tells the workspace.

* ``social_plan_ready``: "Today's posts are ready", once per plan and day, when the
  make tick made posts that wait for approval (linked to the plan).
* ``social_plan_bank_empty``: a slot found no topic in the bank (once per plan and day).
* ``social_plan_slot_skipped``: a slot was not made, and why (over the render quota,
  no channel connected that posts its format): never half-made.
* ``social_post_slot_moved``: the late policy moved a post to the plan's next slot
  (linked to the post).

Like every Socials notice (``notify.py``), it goes out through the platform's
``NotificationDispatcher`` on its own session after the write, and a failure is logged,
never raised. :func:`once_today` keeps the per-day notices to one: the plan's
``make.notified`` remembers the day each went out.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from modules.socials import notify

PLAN_LINK_TYPE = "social_plan"
READY = ("social_plan_ready", "Today's posts are ready: ", "action_required")
BANK_EMPTY = ("social_plan_bank_empty", "The content bank is empty: ", "error")
SLOT_SKIPPED = ("social_plan_slot_skipped", "A planned post was not made: ", "error")
SLOT_MOVED = ("social_post_slot_moved", "Moved to the plan's next slot: ", "ok")
NOTIFIED = "notified"


def once_today(plan: Any, event: str, today: date) -> bool:
    """Whether ``event`` has not gone out for the plan today; marks it as sent (the caller commits)."""
    make = dict(plan.make or {})
    sent = dict(make.get(NOTIFIED) or {})
    if sent.get(event) == today.isoformat():
        return False
    sent[event] = today.isoformat()
    plan.make = {**make, NOTIFIED: sent}
    return True


async def _send(workspace_id: UUID | str, link_type: str, link_id: UUID | str, event: tuple, title: str) -> None:
    event_type, prefix, status = event
    try:
        db = notify._default_session_factory()()
    except Exception:  # noqa: BLE001 — a notice is a courtesy
        notify.logger.exception("[Socials] no session for the %s notice", event_type)
        return
    try:
        await notify._dispatcher(db, str(workspace_id)).dispatch(
            event_type=event_type, title=f"{prefix}{title}", link_type=link_type, link_id=str(link_id), status=status,
        )
    except Exception:  # noqa: BLE001 — logged, never raised into the tick
        notify.logger.exception("[Socials] the %s notice of %s %s was not sent", event_type, link_type, link_id)
    finally:
        db.close()


def notify_plan(workspace_id: UUID | str, plan_id: UUID | str, event: tuple, title: str) -> None:
    """A plan notice, without holding up the caller. Never raises."""
    notify._send_soon(_send(workspace_id, PLAN_LINK_TYPE, plan_id, event, title), plan_id, event[0])


def notify_slot_moved(workspace_id: UUID | str, post_id: UUID | str, title: str, at: Optional[datetime]) -> None:
    when = f" ({at.isoformat(timespec='minutes')})" if at is not None else ""
    notify._send_soon(_send(workspace_id, notify.LINK_TYPE, post_id, SLOT_MOVED, f"{title}{when}"), post_id, SLOT_MOVED[0])
