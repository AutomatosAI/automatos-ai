"""PRD-268 §6.5.1, K8, K9 — a check-in run's suggestions reach their person: the thread and the bell.

``deliver_run(db, run_id)`` runs after the run's row has committed (``services.proactive.runner``),
and again for a failing streak's card once the streak filed it (``services.proactive.failing_streaks``
reports the run it filed the card under). It delivers only while the workspace's mode is ``live``
(``services.proactive.switches.effective_mode``, K8): off and shadow deliver nothing, though the
list routes and the brief still read the open suggestions.

* **The thread.** Each open suggestion the run created that is not delivered yet is claimed first
  (``delivered_at`` set where it is NULL, committed, so it is never posted twice), then posted
  once in its addressee's Auto thread through ``services.chat_messenger.deliver_background_message``:
  source ``{origin 'suggestion', suggestion_id}``, parts ``[text, {type 'suggestion@1',
  suggestion_id}]``. The text quotes the suggestion as data (``services.proactive.delivery_text``);
  the message is Auto's, never the person's. ``suggestion`` is not an announced origin
  (``services.notification_events.auto_message.ANNOUNCED_ORIGINS``): the post raises no
  ``auto_message``. A post that does not land gives its claim back.
* **The bell.** One ``suggestion_ready`` per addressee per run (K9), through
  ``NotificationDispatcher`` (in-app by default; no preference row is written), when the
  suggestions just posted for them hold one of severity ``today``, they are not in quiet hours
  (``services.proactive.quiet.is_person_quiet``; the workspace's window is the dispatcher's own,
  ``auto_reporting``), and fewer than ``config.PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY``
  ``suggestion_ready`` rows were written for them in the workspace since their local midnight
  (in the zone their quiet hours are read in). Its title comes from the strings table, and it
  links to the run's newest today suggestion posted (``link_type`` suggestion). A window or a cap
  that cannot be read rings nothing. The cap is counted from the bell's own rows (K9; no column
  holds it): two runs at the same instant may both ring, and a person whose preferences or
  ``auto_reporting`` route send ``suggestion_ready`` only to Telegram, Slack or a webhook has no
  bell row to count.

Delivery is once per suggestion, at most: a run is delivered when it ends, and nothing sweeps
later. A post that did not land gives its claim back but is not tried again; a worker that dies
between a claim and its post leaves the suggestion marked delivered. Suggestions recorded in
shadow are never delivered when the mode turns live. They all stay in the list and the brief.
Delivery is never part of the run: a failure is logged, and the check-in's result stands.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import List, Mapping, Optional, Sequence, Tuple
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.models.proactive import SUGGESTION_OPEN, ProactiveRun, ProactiveSuggestion
from services.chat_messenger import deliver_background_message
from services.proactive.delivery_text import bell_copy, thread_text
from services.proactive.quiet import is_person_quiet, person_timezone
from services.proactive.switches import MODE_LIVE, effective_mode

logger = logging.getLogger(__name__)

__all__ = [
    "BELL_EVENT",
    "BELL_LINK",
    "Bell",
    "Delivery",
    "ORIGIN",
    "PART_TYPE",
    "deliver_run",
]

# The post's provenance (chat_messenger's source) and its card's part (services.chat_parts).
ORIGIN = "suggestion"
PART_TYPE = "suggestion@1"
# The bell's event (core.services.notification_dispatcher.VALID_EVENT_TYPES) and what its row opens.
BELL_EVENT = "suggestion_ready"
BELL_LINK = "suggestion"
AUTO_NAME = "Auto"
SEVERITY_TODAY = "today"

# The person's suggestion_ready rows in the workspace since a moment. Bound parameters only.
_RUNG_SINCE_SQL = text(
    "SELECT count(*) FROM notifications "
    "WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid AND event_type = :event "
    "  AND created_at >= :since"
)


@dataclass(frozen=True)
class Bell:
    """One person's ``suggestion_ready`` row: who, what it says, and the suggestion it opens."""

    workspace_id: UUID
    user_id: int
    title: str
    message: str
    link_id: str


@dataclass(frozen=True)
class Delivery:
    """What a run's delivery did: how many suggestions it posted and how many bells it rang."""

    posted: int = 0
    rung: int = 0


@dataclass(frozen=True)
class _Card:
    """A suggestion to deliver, as plain values read before any commit."""

    id: str
    workspace_id: UUID
    user_id: int
    title: str
    detail: str
    severity: str
    created_at: datetime


@dataclass(frozen=True)
class _Person:
    """An addressee as the messenger finds them (a Clerk id, else an email) and their language."""

    recorded: Optional[str]
    language: str


@dataclass(frozen=True)
class _Posted:
    posted: int = 0
    bells: Tuple[Bell, ...] = ()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _undelivered(db: Session, run_id: UUID) -> List[_Card]:
    """The open suggestions the run created that are not delivered yet, oldest first; [] for an unknown run."""
    run = db.query(ProactiveRun.workspace_id).filter(ProactiveRun.id == run_id).first()
    if run is None:
        return []
    rows = db.query(
        ProactiveSuggestion.id, ProactiveSuggestion.workspace_id, ProactiveSuggestion.user_id,
        ProactiveSuggestion.title, ProactiveSuggestion.detail, ProactiveSuggestion.severity,
        ProactiveSuggestion.created_at,
    ).filter(
        ProactiveSuggestion.workspace_id == run.workspace_id, ProactiveSuggestion.run_id == run_id,
        ProactiveSuggestion.status == SUGGESTION_OPEN, ProactiveSuggestion.delivered_at.is_(None),
    ).order_by(ProactiveSuggestion.created_at, ProactiveSuggestion.id).all()
    return [_Card(str(row.id), row.workspace_id, int(row.user_id), row.title, row.detail or "", row.severity,
                  row.created_at) for row in rows]


def _person(db: Session, workspace_id: UUID, user_id: int, now: datetime) -> _Person:
    """How the messenger finds the person, and the language of the phone their brief follows (K7)."""
    from services.brief.strings import brief_language
    from services.mobile.digest import person_today, recorded_person

    return _Person(recorded_person(db, user_id), brief_language(person_today(db, workspace_id, user_id, now).locale))


def _card_row(card: _Card) -> tuple:
    return ProactiveSuggestion.workspace_id == card.workspace_id, ProactiveSuggestion.id == UUID(card.id)


def _claim(db: Session, card: _Card, now: datetime) -> bool:
    """Mark the card delivered where it is not yet, committed; False when it already was."""
    claimed = db.query(ProactiveSuggestion).filter(
        *_card_row(card), ProactiveSuggestion.delivered_at.is_(None),
    ).update({ProactiveSuggestion.delivered_at: now}, synchronize_session=False)
    db.commit()
    return bool(claimed)


def _release(db: Session, card: _Card) -> None:
    """Give a claim back (its post did not land), committed."""
    db.query(ProactiveSuggestion).filter(*_card_row(card)).update(
        {ProactiveSuggestion.delivered_at: None}, synchronize_session=False)
    db.commit()


def _post(db: Session, card: _Card, person: _Person) -> bool:
    """Post the card in the person's Auto thread, as Auto; True when it landed."""
    message = deliver_background_message(
        db,
        workspace_id=card.workspace_id,
        text=thread_text(card.title, card.detail, person.language),
        source={"origin": ORIGIN, "suggestion_id": card.id},
        clerk_user_id=person.recorded,
        parts=[{"type": PART_TYPE, "suggestion_id": card.id}],
    )
    return message is not None


def _deliver_card(db: Session, card: _Card, person: _Person, now: datetime) -> bool:
    """Claim and post one card; True when it was posted. A post that does not land gives the claim
    back. Never raises: a failure is logged, and the run's other cards still go."""
    try:
        if not _claim(db, card, now):
            return False
        if person.recorded is not None and _post(db, card, person):
            db.commit()  # the open chat's live notice; the message itself is committed already
            return True
        _release(db, card)
    except Exception:  # noqa: BLE001 — logged; one card that fails never stops the others
        db.rollback()
        logger.exception("[proactive] suggestion %s could not be delivered", card.id)
    return False


def _local_midnight(zone: str, now: datetime) -> datetime:
    """The start of ``now``'s day on the wall clock of ``zone`` (an IANA name zoneinfo knows)."""
    tz = ZoneInfo(zone)
    return datetime.combine(now.astimezone(tz).date(), time.min, tzinfo=tz)


def _rung_since(db: Session, workspace_id: UUID, user_id: int, since: datetime) -> int:
    return int(db.execute(_RUNG_SINCE_SQL, {"ws": str(workspace_id), "uid": user_id, "event": BELL_EVENT,
                                            "since": since}).scalar() or 0)


def _may_ring(db: Session, workspace_id: UUID, user_id: int, now: datetime) -> bool:
    """Whether the person may be rung now: not in quiet hours, and under the day's cap. A read that
    fails is a no, and logged; the session stays usable (the reads run in a savepoint)."""
    from core.services.auto_reporting import load_auto_reporting_settings
    from services.proactive.settings import read_person

    try:
        with db.begin_nested():
            mine, window = read_person(db, workspace_id, user_id), load_auto_reporting_settings(db, workspace_id)
            if is_person_quiet(mine, window, now):
                return False
            rung = _rung_since(db, workspace_id, user_id, _local_midnight(person_timezone(mine, window), now))
    except Exception:  # noqa: BLE001 — a window or a cap that cannot be read rings nothing
        logger.exception("[proactive] the bell of user %s in %s could not be checked: not rung", user_id, workspace_id)
        return False
    return rung < config.PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY


def _bell(db: Session, user_id: int, today: Sequence[_Card], language: str, now: datetime) -> Optional[Bell]:
    """The person's bell for the today cards just posted for them, or None when they may not be rung."""
    workspace_id = today[0].workspace_id
    if not _may_ring(db, workspace_id, user_id, now):
        return None
    newest = max(today, key=lambda card: (card.created_at, card.id))
    title, message = bell_copy(len(today), language)
    return Bell(workspace_id, user_id, title, message, newest.id)


def _bells(db: Session, posted: Sequence[_Card], people: Mapping[int, _Person], now: datetime) -> Tuple[Bell, ...]:
    """One bell per addressee who was just posted a suggestion for today, as far as each may be rung."""
    today = [card for card in posted if card.severity == SEVERITY_TODAY]
    found = (_bell(db, user_id, [card for card in today if card.user_id == user_id], people[user_id].language, now)
             for user_id in sorted({card.user_id for card in today}))
    return tuple(bell for bell in found if bell is not None)


def _post_run(db: Session, run_id: UUID) -> _Posted:
    """Post the run's undelivered suggestions, when the workspace is live, and plan its bells."""
    cards = _undelivered(db, run_id)
    if not cards or effective_mode(db, cards[0].workspace_id) != MODE_LIVE:
        return _Posted()
    now = _now()
    people = {user_id: _person(db, cards[0].workspace_id, user_id, now)
              for user_id in sorted({card.user_id for card in cards})}
    posted = [card for card in cards if _deliver_card(db, card, people[card.user_id], now)]
    return _Posted(len(posted), _bells(db, posted, people, now))


async def _ring(db: Session, bell: Bell) -> bool:
    """Raise the person's ``suggestion_ready``, committed; True when it went. Never raises."""
    from core.services.notification_dispatcher import NotificationDispatcher

    try:
        await NotificationDispatcher(db, bell.workspace_id).dispatch(
            event_type=BELL_EVENT, title=bell.title, message=bell.message, link_type=BELL_LINK,
            link_id=bell.link_id, agent_name=AUTO_NAME, user_id=bell.user_id,
        )
        db.commit()
    except Exception:  # noqa: BLE001 — logged; a bell that fails never fails the run
        db.rollback()
        logger.exception("[proactive] the suggestion bell of user %s in %s failed", bell.user_id, bell.workspace_id)
        return False
    return True


async def deliver_run(db: Session, run_id: UUID) -> Delivery:
    """Deliver a check-in run's new suggestions while the workspace is live (K8): each posted once in
    its addressee's Auto thread, then one ``suggestion_ready`` per addressee under K9's rules. What
    it did; a failure is logged and never reaches the run."""
    try:
        planned = await asyncio.to_thread(_post_run, db, run_id)
    except Exception:  # noqa: BLE001 — logged; delivery is never part of the run
        db.rollback()
        logger.exception("[proactive] the suggestions of run %s could not be delivered", run_id)
        return Delivery()
    rung = [await _ring(db, bell) for bell in planned.bells]
    if planned.posted:
        logger.info("[proactive] run %s: %d suggestion(s) posted, %d bell(s) rung", run_id, planned.posted, sum(rung))
    return Delivery(planned.posted, sum(rung))
