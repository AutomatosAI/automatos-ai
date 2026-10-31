"""PRD-268 §6.5.3, §6.9, K9 (US-C09b) — who a suggestion push reaches, and when.

``services.proactive.delivery.deliver_run`` raises at most one ``suggestion_ready`` per person per
run, and only for a run that made a suggestion of severity ``today``, outside the person's quiet
hours and under ``config.PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY``; the dispatcher holds every
push in the workspace's quiet hours. What is left to the push is decided here, when it is sent
(``services.push.fanout.eligible_devices``, for the suggestion type only):

* ``may_push_suggestion`` — the platform switch ``config.SUGGESTION_PUSH_ENABLED`` is on (K9: it
  ships off, and only the owner turns it on), and the event links the suggestion of the one person
  it names, in the event's workspace, of severity ``today`` and still open at ``now`` (status
  ``open`` and ``expires_at`` not yet passed: a suggestion past it that no read has marked expired
  yet has lapsed all the same).
* ``category_filters`` — a push type in ``payload.OPTIONAL_PUSH_TYPES`` reaches only phones whose
  app registered its category (``mobile_devices.push_categories``, written by the phone's own
  ``POST /api/v1/mobile/devices``); the other push types add no filter.

Reads only, on the caller's session; nothing here writes or logs.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import config
from core.models.mobile_devices import MobileDevice
from core.models.proactive import SUGGESTION_OPEN, ProactiveSuggestion
from services.push.payload import OPTIONAL_PUSH_TYPES

__all__ = [
    "SEVERITY_TODAY",
    "SUGGESTION_LINK_TYPE",
    "category_filters",
    "linked_suggestion_id",
    "may_push_suggestion",
    "open_at",
    "suggestion_push_enabled",
]

# A suggestion's link is the suggestion's id (PRD-268 US-C08a).
SUGGESTION_LINK_TYPE = "suggestion"
# Only a suggestion for today is pushed (§6.5.3); one that can wait waits for the brief.
SEVERITY_TODAY = "today"


def suggestion_push_enabled() -> bool:
    """K9's platform switch, read from config on every call: off as shipped."""
    return bool(config.SUGGESTION_PUSH_ENABLED)


def linked_suggestion_id(event: Any) -> Optional[uuid.UUID]:
    """The id of the suggestion an event links, when it names its person; None otherwise."""
    if event.link_type != SUGGESTION_LINK_TYPE or event.user_id is None:
        return None
    try:
        return uuid.UUID(str(event.link_id))
    except ValueError:
        return None


def open_at(now: datetime) -> Tuple[Any, ...]:
    """A suggestion's filter for still open at ``now``: open, and not past its expiry."""
    return (
        ProactiveSuggestion.status == SUGGESTION_OPEN,
        or_(ProactiveSuggestion.expires_at.is_(None), ProactiveSuggestion.expires_at > now),
    )


def may_push_suggestion(db: Session, event: Any, now: datetime) -> bool:
    """Whether a ``suggestion_ready`` event may be pushed at all now (K9).

    ``event`` is a ``services.push.fanout.PushEvent``. True only while the switch is on and the
    event links its own person's suggestion in its workspace, of severity ``today``, still open.
    """
    suggestion_id = linked_suggestion_id(event)
    if not suggestion_push_enabled() or suggestion_id is None:
        return False
    found = db.query(ProactiveSuggestion.id).filter(
        ProactiveSuggestion.id == suggestion_id,
        ProactiveSuggestion.workspace_id == uuid.UUID(str(event.workspace_id)),
        ProactiveSuggestion.user_id == event.user_id,
        ProactiveSuggestion.severity == SEVERITY_TODAY,
        *open_at(now),
    ).first()
    return found is not None


def category_filters(push_type: Optional[str]) -> Tuple[Any, ...]:
    """The device filter a push type adds: an optional type reaches only phones whose app
    registered its category; any other type adds none."""
    if push_type not in OPTIONAL_PUSH_TYPES:
        return ()
    return (MobileDevice.push_categories.contains([push_type]),)
