"""PRD-268 §6.4, K7, K11 — the brief on open: the day's brief waiting the first time a person opens Automatos.

``open_brief`` serves ``POST /api/proactive/brief/open`` (``api/proactive_brief.py``) for one
person, workspace and local date, from any device or tab, web-only people included:

* a ``local_date`` more than ``DATE_WINDOW_DAYS`` from the server's UTC date is refused
  (``OutOfWindow``) before anything is read or written;
* the day is claimed in ``mobile_digest_posts`` (``services.brief.claim.claim_brief``), the
  claim the scheduled digest takes too. The winner builds the brief (``build_brief`` through
  ``services.mobile.digest.brief_of``: Auto's Read, made here, then the deterministic per-person
  block, in the caller's language and zone, with no question or approval rows for a person
  whose role does not answer them, PRD-266 D16) and posts it into the person's own Auto thread
  via ``open``, never announced (``post_brief``). It answers ``posted`` true;
* a claim already taken answers the day's message (``todays_brief_message``: the person's own
  Auto thread in this workspace only) with ``posted`` false, or ``pending`` while the winner is
  still building;
* a build or post that fails gives the claim back, so the next open tries again, and raises
  ``NotPosted``. It is logged by its error's class only: a database error's text carries its
  bound parameters, the brief's words among them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from services.brief.claim import claim_brief, post_brief, todays_brief_message
from services.brief.model import VIA_OPEN
from services.mobile import digest
from services.mobile.digest import DuePerson

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


def _give_back(db: Session, person: DuePerson) -> None:
    """Release the claim on a clean session; a release that fails is logged by its class only."""
    try:
        db.rollback()
        digest.release(db, person)
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("Brief on open: the claim of person %s in %s was not given back (%s)",
                     person.user_id, person.workspace_id, type(failure).__name__)


def _post(db: Session, person: DuePerson) -> Any:
    """Build the person's brief and post it via open; the saved message, or None."""
    return post_brief(db, person, digest.brief_of(db, person), VIA_OPEN)


def open_brief(db: Session, person: DuePerson, now: Optional[datetime] = None) -> Opened:
    """The person's brief for their local date: posted by this open when it wins the day's claim,
    else the day's message, else pending. ``person.zone`` and ``person.locale`` are the caller's
    clock and language. Raises ``OutOfWindow`` before any read, and ``NotPosted`` (the claim
    given back) when the build or the post fails."""
    if not in_window(person.local_date, now or _now()):
        raise OutOfWindow(person.local_date.isoformat())
    if not claim_brief(db, person.workspace_id, person.user_id, person.local_date):
        found = todays_brief_message(db, person.workspace_id, person.user_id, person.local_date)
        return _answer(found, posted=False) if found is not None else Opened(posted=False)
    try:
        message = _post(db, person)
    except Exception as failure:  # noqa: BLE001 — logged by class only (docstring); the claim is given back
        logger.error("Brief on open: person %s in %s failed (%s)",
                     person.user_id, person.workspace_id, type(failure).__name__)
        _give_back(db, person)
        raise NotPosted(type(failure).__name__) from failure
    if message is None:
        logger.error("Brief on open: person %s in %s was not posted", person.user_id, person.workspace_id)
        _give_back(db, person)
        raise NotPosted("not posted")
    return _answer(message, posted=True)
