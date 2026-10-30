"""PRD-268 §6.4, K7, K11 — the brief on open: the day's brief waiting the first time a person opens Automatos.

``open_brief`` serves ``POST /api/proactive/brief/open`` (``api/proactive_brief.py``) for one
person, workspace and local date, from any device or tab, web-only people included:

* a ``local_date`` more than ``DATE_WINDOW_DAYS`` from the server's UTC date is refused
  (``OutOfWindow``) before anything is read or written;
* the day is claimed in ``mobile_digest_posts`` (``services.mobile.digest.lease``), the claim
  the scheduled digest takes too. The winner builds the brief (``build_brief`` through
  ``services.mobile.digest.brief_of``: Auto's Read, made here, then the deterministic per-person
  block, in the caller's language and zone, with no question or approval rows for a person
  whose role does not answer them, PRD-266 D16) and posts it into the person's own Auto thread
  via ``open``, never announced (``post_brief``). It answers ``posted`` true;
* a claim already taken answers the day's message (``todays_brief_message``: the person's own
  Auto thread in this workspace only) with ``posted`` false, or ``pending`` while the winner is
  still building;
* a claim already taken with no day's message, older than ``config.BRIEF_CLAIM_STALE_SECONDS``,
  was left by a claimant that died before its post: this open takes it over
  (``digest.take_over``, one compare-and-set, P268W2-RVW-1) and builds and posts as the winner;
* the winner posts only while the day is still the one it holds (``digest.still_held``, just
  before the post); a winner whose day was taken over posts nothing, logs it by class only and
  answers as a lost claim: the day's message, or ``pending``;
* a build or post that fails gives back the claim this open holds (never a taker's), so the next
  open tries again, and raises ``NotPosted``. It is logged by its error's class only: a
  database error's text carries its bound parameters, the brief's words among them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.core import Message
from services.brief.claim import post_brief, todays_brief_message
from services.brief.model import VIA_OPEN
from services.mobile import digest
from services.mobile.digest import DuePerson, LeaseLost

logger = logging.getLogger(__name__)

__all__ = ["DATE_WINDOW_DAYS", "NotPosted", "Opened", "OutOfWindow", "in_window", "open_brief"]

# A client's local date is at most this many days from the server's UTC date (§6.4).
DATE_WINDOW_DAYS = 1


class OutOfWindow(ValueError):
    """The local date is more than ``DATE_WINDOW_DAYS`` from the server's UTC date."""


class NotPosted(RuntimeError):
    """The brief could not be built or posted; the claim was given back."""


@dataclass(frozen=True)
class Opened:
    """What an open answers: the day's message, whether this open posted it, or that it is pending."""

    posted: bool
    chat_id: Optional[str] = None
    message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """``{chat_id, message_id, posted}``, or ``{posted: false, pending: true}`` without a message."""
        if self.message_id is None:
            return {"posted": False, "pending": True}
        return {"chat_id": self.chat_id, "message_id": self.message_id, "posted": self.posted}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def in_window(local_date: date, now: datetime) -> bool:
    """``local_date`` is within ``DATE_WINDOW_DAYS`` of ``now``'s UTC date."""
    utc_today = now.astimezone(timezone.utc).date()
    return abs((local_date - utc_today).days) <= DATE_WINDOW_DAYS


def _answer(message: Any, posted: bool) -> Opened:
    return Opened(posted=posted, chat_id=str(message.chat_id), message_id=str(message.id))


def _taken(found: Optional[Message]) -> Opened:
    """A day another claimant holds: its message, with ``posted`` false, else pending."""
    return _answer(found, posted=False) if found is not None else Opened(posted=False)


def _found(db: Session, person: DuePerson) -> Optional[Message]:
    return todays_brief_message(db, person.workspace_id, person.user_id, person.local_date)


def _hold(db: Session, person: DuePerson) -> Tuple[Optional[datetime], Optional[Message]]:
    """The lease this open holds on the person's day: the claim won, else a stale claim taken over
    while the day's message is not there. Without one, the day's message (None while pending)."""
    held = digest.lease(db, person)
    if held is not None:
        return held, None
    found = _found(db, person)
    if found is not None:
        return None, found
    return digest.take_over(db, person), None


def _give_back(db: Session, person: DuePerson, held: datetime) -> None:
    """Release the claim this open holds on a clean session; a release that fails is logged by
    its class only."""
    try:
        db.rollback()
        digest.release(db, person, held=held)
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("Brief on open: the claim of person %s in %s was not given back (%s)",
                     person.user_id, person.workspace_id, type(failure).__name__)


def _post(db: Session, person: DuePerson, held: datetime) -> Any:
    """Build the person's brief and post it via open while this open still holds the day; the
    saved message, or None (the claim given back). Raises ``LeaseLost`` when it was taken over."""
    return post_brief(db, person, digest.brief_of(db, person), VIA_OPEN, held=held)


def open_brief(db: Session, person: DuePerson, now: Optional[datetime] = None) -> Opened:
    """The person's brief for their local date: posted by this open when it wins the day's claim
    (or takes a stale one over), else the day's message, else pending. ``person.zone`` and
    ``person.locale`` are the caller's clock and language. Raises ``OutOfWindow`` before any
    read, and ``NotPosted`` (the claim given back) when the build or the post fails."""
    if not in_window(person.local_date, now or _now()):
        raise OutOfWindow(person.local_date.isoformat())
    held, found = _hold(db, person)
    if held is None:
        return _taken(found)
    try:
        message = _post(db, person, held)
    except LeaseLost as lost:
        logger.warning("Brief on open: person %s in %s was taken over (%s)",
                       person.user_id, person.workspace_id, type(lost).__name__)
        return _taken(_found(db, person))
    except Exception as failure:  # noqa: BLE001 — logged by class only (docstring); the claim is given back
        logger.error("Brief on open: person %s in %s failed (%s)",
                     person.user_id, person.workspace_id, type(failure).__name__)
        _give_back(db, person, held)
        raise NotPosted(type(failure).__name__) from failure
    if message is None:
        logger.error("Brief on open: person %s in %s was not posted", person.user_id, person.workspace_id)
        raise NotPosted("not posted")
    return _answer(message, posted=True)
