"""PRD-266 US-013 (§4.5, D24, FR-17, K12) — the daily digest: Auto's Read in each person's Auto thread.

``run_digests`` runs on the leader's scheduler every ``config.DIGEST_JOB_INTERVAL_SECONDS``
(``register``, called from ``services.schedule_reconcile.start_schedule_reconcile``):

1. Claims older than ``DIGEST_POSTS_KEEP_DAYS`` are deleted.
2. ``due_people``: a person's schedule in a workspace follows their most recently seen active
   ios/android phone there that has a timezone (``last_seen_at``, then ``created_at``): its
   ``digest_time``, else ``config.DIGEST_DEFAULT_TIME``, and its ``locale``, the brief's
   language (PRD-268 K7). They are due while their local time is
   within [digest time, digest time + ``DIGEST_CATCH_UP_MINUTES``) on that local date. A phone
   without a timezone and a paired browser schedule nothing; neither does a person whose
   ``daily_digest`` push row is off (theirs, else the workspace's; no row means on: US-014a,
   K12) or who no longer belongs to the workspace (``services.push.recipients``).
3. Each due person's (workspace, person, local date) is claimed in ``mobile_digest_posts``
   (``INSERT … ON CONFLICT DO NOTHING``, committed): a second claim from any tick, worker or
   replica posts nothing. Several phones make one digest. The brief on open
   (``services.brief.claim``, PRD-268 K11) takes the same claim, so a person gets one brief a
   day, whichever comes first.
4. The text is ``services.digest_service.generate_digest`` (Auto's Read, cached per workspace
   state; once per workspace per run), booked to the workspace like the Command Center's. The
   brief (``services.brief.build.build_brief``, ``Brief@1``) is built over it for the person, in
   their phone's language and zone, with no second model call.
5. ``deliver_background_message`` posts it as Auto's message in the person's Auto thread:
   ``[text(Auto's Read), brief@1]``, origin ``daily_digest``, its local date and via
   ``schedule``. That raises ``auto_message`` (US-012b; never coalesced), so the phone's push
   carries it (D24). A build or post that fails releases the claim, so a later tick in the
   window, or the person's next open, tries again.

``run_digests`` is async, on the leader's loop; every database step (and the digest's own
snapshot) runs in a worker thread through ``anyio.to_thread.run_sync``, each on a session of
its own. It never raises and never logs the digest's text: a brief that is not built or posted
is logged by its error's class only (a database error's text carries its bound parameters,
the brief's words among them).

``post_now`` is the same brief for one person on request ("Send me today's brief now",
``POST /api/v1/mobile/digest/now``, ``services.mobile.digest_now``): the same text and brief,
posted the same way (origin ``daily_digest``, their local date by ``person_today``, no via, as
today), with no claim read or written, so the scheduled digest still goes out. Their
``daily_digest`` switch is not read: it is an explicit ask. It never raises; a failure is
logged by its class only.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from itertools import groupby
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from config import config
from core.models.core import User
from core.models.mobile_devices import MobileDevice, MobileDeviceStatus
from core.models.mobile_push import MobileDigestPost
from services.brief.build import build_brief
from services.brief.model import PART_TYPE as BRIEF_PART
from services.brief.model import VIA_SCHEDULE
from services.chat_messenger import deliver_background_message
from services.digest_service import generate_digest
from services.notification_events.auto_message import DAILY_DIGEST as DIGEST_ORIGIN
from services.push.payload import NEEDS_ANSWER
from services.push.preferences import DAILY_DIGEST_EVENT, PUSH_DESTINATION
from services.push.recipients import AUTO_MESSAGE, recipients

logger = logging.getLogger(__name__)

DIGEST_JOB_ID = "daily_digest_tick"
# A person is due from their digest time until this many minutes after it (K12).
DIGEST_CATCH_UP_MINUTES = 90
# A claim only matters on its own local date; older ones are deleted at each run.
DIGEST_POSTS_KEEP_DAYS = 30
# What DIGEST_DEFAULT_TIME reads as when it is not HH:MM.
FALLBACK_DIGEST_TIME = time(8, 0)
HHMM = re.compile(r"^([01][0-9]|2[0-3]):([0-5][0-9])$")
# Only phones keep a schedule: a paired browser ('web') never does.
DIGEST_PLATFORMS = ("ios", "android")

POSTED, TAKEN, RELEASED = "posted", "taken", "released"
COUNT_KEYS = ("due", POSTED, TAKEN, RELEASED)

SessionFactory = Callable[[], Session]

# Per (workspace, person): the most recently seen active phone with a timezone, and its
# language, unless the person's daily_digest push row is off (theirs, else the workspace's; none
# means on). The digest's switch is read as the fan-out reads a push switch. Bound parameters only.
_SCHEDULES_SQL = text(
    "SELECT DISTINCT ON (d.workspace_id, d.user_id) d.workspace_id, d.user_id, d.timezone, d.digest_time, "
    "  d.locale "
    "FROM mobile_devices d "
    "WHERE d.status = :active AND d.platform IN :platforms "
    "  AND d.timezone IS NOT NULL AND d.timezone <> '' "
    "  AND COALESCE("
    "    (SELECT bool_and(p.enabled) FROM notification_preferences p "
    "      WHERE p.workspace_id = d.workspace_id AND p.user_id = d.user_id "
    "        AND p.event_type = :event_type AND p.destination = :destination), "
    "    (SELECT bool_and(p.enabled) FROM notification_preferences p "
    "      WHERE p.workspace_id = d.workspace_id AND p.user_id IS NULL "
    "        AND p.event_type = :event_type AND p.destination = :destination), "
    "    true) "
    "ORDER BY d.workspace_id, d.user_id, d.last_seen_at DESC NULLS LAST, d.created_at DESC, d.id"
).bindparams(bindparam("platforms", expanding=True))


@dataclass(frozen=True, order=True)
class DuePerson:
    """One digest owed: the workspace, the person (``users.id``) and their local date.

    ``zone`` and ``locale`` are those of the phone their schedule follows (the brief's clock and
    language, PRD-268 K7): how to say it, never who is owed, so they take no part in equality
    or order.
    """

    workspace_id: uuid.UUID
    user_id: int
    local_date: date
    zone: Optional[str] = field(default=None, compare=False)
    locale: Optional[str] = field(default=None, compare=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(moment: datetime) -> datetime:
    """``moment`` in UTC; a naive one is read as UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def parse_digest_time(value: Optional[str]) -> Optional[time]:
    """``'HH:MM'`` as a time, or None for anything else."""
    match = HHMM.match(str(value or "").strip())
    return time(int(match.group(1)), int(match.group(2))) if match else None


def default_digest_time() -> time:
    """``config.DIGEST_DEFAULT_TIME``, or 08:00 when it is not HH:MM."""
    return parse_digest_time(config.DIGEST_DEFAULT_TIME) or FALLBACK_DIGEST_TIME


def _zone(name: Optional[str]) -> Optional[ZoneInfo]:
    try:
        return ZoneInfo(str(name or ""))
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def due_date(zone_name: Optional[str], digest_time: time, now: datetime) -> Optional[date]:
    """The local date whose digest is due at ``now`` in this zone, or None.

    Due while ``now`` is within [digest time, digest time + ``DIGEST_CATCH_UP_MINUTES``) of a
    local date, measured in real minutes (zoneinfo: a DST change moves the UTC instant, never
    the window's length). Yesterday's window is checked too: it can run past local midnight.
    """
    zone = _zone(zone_name)
    if zone is None:
        return None
    moment = _utc(now)
    today = moment.astimezone(zone).date()
    for local in (today, today - timedelta(days=1)):
        start = datetime.combine(local, digest_time, tzinfo=zone).astimezone(timezone.utc)
        if start <= moment < start + timedelta(minutes=DIGEST_CATCH_UP_MINUTES):
            return local
    return None


def belongs(db: Session, workspace_id: uuid.UUID, user_id: int) -> bool:
    """The person still belongs to the workspace (the fan-out's own read of who does)."""
    return bool(recipients(db, workspace_id, AUTO_MESSAGE, user_id))


def due_people(db: Session, now: datetime) -> List[DuePerson]:
    """Every (workspace, person, local date) whose digest is due at ``now``, by workspace."""
    rows = db.execute(_SCHEDULES_SQL, {
        "active": MobileDeviceStatus.ACTIVE.value,
        "platforms": list(DIGEST_PLATFORMS),
        "event_type": DAILY_DIGEST_EVENT,
        "destination": PUSH_DESTINATION,
    }).fetchall()
    default = default_digest_time()
    due: List[DuePerson] = []
    for workspace_id, user_id, zone_name, digest_time, locale in rows:
        local = due_date(zone_name, parse_digest_time(digest_time) or default, now)
        person = DuePerson(uuid.UUID(str(workspace_id)), int(user_id), local, zone_name, locale) if local else None
        if person is not None and belongs(db, person.workspace_id, person.user_id):
            due.append(person)
    return sorted(due)


def _schedule_phone(db: Session, workspace_id: uuid.UUID, user_id: int) -> Optional[Any]:
    """The timezone and locale of the phone the person's schedule follows; None without one.

    That phone is ``due_people``'s: their most recently seen active ios/android phone in the
    workspace with a timezone (``last_seen_at``, then ``created_at``).
    """
    return (
        db.query(MobileDevice.timezone, MobileDevice.locale)
        .filter(
            MobileDevice.workspace_id == workspace_id,
            MobileDevice.user_id == user_id,
            MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
            MobileDevice.platform.in_(DIGEST_PLATFORMS),
            MobileDevice.timezone.isnot(None),
            MobileDevice.timezone != "",
        )
        .order_by(MobileDevice.last_seen_at.desc().nulls_last(), MobileDevice.created_at.desc(), MobileDevice.id)
        .first()
    )


def schedule_zone(db: Session, workspace_id: uuid.UUID, user_id: int) -> Optional[str]:
    """The timezone of the phone the person's schedule follows, as stored; None without one."""
    phone = _schedule_phone(db, workspace_id, user_id)
    return phone.timezone if phone is not None else None


def date_in_zone(zone_name: Optional[str], now: datetime) -> date:
    """``now``'s date in the IANA zone ``zone_name``; in UTC when it is missing or not one."""
    return _utc(now).astimezone(_zone(zone_name) or timezone.utc).date()


def person_today(db: Session, workspace_id: uuid.UUID, user_id: int, now: datetime) -> DuePerson:
    """The person's brief at ``now``: their local date in the zone of the phone their schedule
    follows, else UTC, with that phone's zone and language (none without one)."""
    phone = _schedule_phone(db, workspace_id, user_id)
    zone, locale = (phone.timezone, phone.locale) if phone is not None else (None, None)
    return DuePerson(workspace_id, user_id, date_in_zone(zone, now), zone, locale)


def _claim_filter(person: DuePerson) -> tuple:
    return (
        MobileDigestPost.workspace_id == person.workspace_id,
        MobileDigestPost.user_id == person.user_id,
        MobileDigestPost.local_date == person.local_date,
    )


def claim(db: Session, person: DuePerson) -> bool:
    """Claim the person's digest for their local date, committed; False when it was claimed already."""
    statement = (
        pg_insert(MobileDigestPost)
        .values(workspace_id=person.workspace_id, user_id=person.user_id, local_date=person.local_date)
        .on_conflict_do_nothing(index_elements=["workspace_id", "user_id", "local_date"])
        .returning(MobileDigestPost.user_id)
    )
    won = db.execute(statement).first() is not None
    db.commit()
    return won


def release(db: Session, person: DuePerson) -> None:
    """Give the claim back (the post did not land), committed, so a later tick tries again."""
    db.query(MobileDigestPost).filter(*_claim_filter(person)).delete(synchronize_session=False)
    db.commit()


def prune(db: Session, now: datetime) -> int:
    """Delete claims older than ``DIGEST_POSTS_KEEP_DAYS``, committed; how many went."""
    cutoff = _utc(now).date() - timedelta(days=DIGEST_POSTS_KEEP_DAYS)
    gone = db.query(MobileDigestPost).filter(MobileDigestPost.local_date < cutoff).delete(synchronize_session=False)
    db.commit()
    return gone


def _plan(db: Session, now: datetime) -> List[DuePerson]:
    prune(db, now)
    return due_people(db, now)


def _digest_read(db: Session, workspace_id: uuid.UUID) -> Dict[str, Any]:
    """Auto's Read for the workspace (``generate_digest``'s answer); runs in a worker thread, so
    its own loop runs the coroutine."""
    return dict(asyncio.run(generate_digest(db, workspace_id)) or {})


def _said(read: Mapping[str, Any]) -> bool:
    """Auto's Read has words to post."""
    return bool(str(read.get("text") or "").strip())


def recorded_person(db: Session, user_id: int) -> Optional[str]:
    """The person as ``resolve_recorded_person`` reads them: their Clerk id, else their email."""
    row = db.query(User.clerk_user_id, User.email).filter(User.id == user_id).first()
    if row is None:
        return None
    return (row.clerk_user_id or "").strip() or (row.email or "").strip() or None


def _may_answer(db: Session, person: DuePerson) -> bool:
    """PRD-266 D16: the person's role in the workspace answers questions (the badge's rule)."""
    return person.user_id in recipients(db, person.workspace_id, NEEDS_ANSWER, person.user_id)


def brief_of(db: Session, person: DuePerson, read: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The person's ``Brief@1`` for their local date, in ``person``'s language and zone (PRD-268
    K7): over Auto's Read already made (no second model call), else made here (the brief on open)."""
    return build_brief(db, person.workspace_id, person.user_id, person.local_date, person.locale,
                       may_answer=_may_answer(db, person), zone=person.zone, digest=read)


def brief_source(local_date: date, via: Optional[str]) -> Dict[str, str]:
    """The day's brief's source (PRD-268 K11): origin ``daily_digest``, its local date and how it
    came to be posted; no via for ``post_now``'s ask, as today."""
    source = {"origin": DIGEST_ORIGIN, "local_date": local_date.isoformat()}
    return {**source, "via": via} if via else source


def post_brief_message(db: Session, person: DuePerson, brief: Mapping[str, Any], via: Optional[str]) -> Optional[Any]:
    """Post the brief into the person's Auto thread: ``[text(prose), brief@1]`` with
    ``brief_source``; the saved message, or None when it did not land."""
    recorded = recorded_person(db, person.user_id)
    if recorded is None:
        return None
    message = deliver_background_message(
        db,
        workspace_id=person.workspace_id,
        text=str(brief.get("prose") or ""),
        source=brief_source(person.local_date, via),
        clerk_user_id=recorded,
        parts=[{"type": BRIEF_PART, "brief": dict(brief)}],
    )
    if message is not None:
        _commit_live_notice(db)
    return message


def _post(db: Session, person: DuePerson, read: Mapping[str, Any], via: Optional[str]) -> bool:
    """Build the person's brief over Auto's Read and post it; True when it landed."""
    return post_brief_message(db, person, brief_of(db, person, read), via) is not None


def _commit_live_notice(db: Session) -> None:
    """Commit the open chat's live NOTIFY (the message itself is committed already).

    A failure here is logged and the post stands: releasing the claim would post twice.
    """
    try:
        db.commit()
    except Exception:
        logger.exception("Daily digest: the open chat's live notice was not committed")
        db.rollback()


def _in_session(factory: SessionFactory, work: Callable[..., Any], *args: Any) -> Any:
    """``work(db, *args)`` on a session of its own, closed after."""
    db = factory()
    try:
        return work(db, *args)
    finally:
        db.close()


async def _in_thread(factory: SessionFactory, work: Callable[..., Any], *args: Any) -> Any:
    return await anyio.to_thread.run_sync(_in_session, factory, work, *args)


async def _generated(factory: SessionFactory, workspace_id: uuid.UUID) -> Dict[str, Any]:
    """The workspace's Auto's Read, or {} when it could not be made (logged)."""
    try:
        return await _in_thread(factory, _digest_read, workspace_id)
    except Exception:
        logger.exception("Daily digest: workspace %s's digest could not be made", workspace_id)
        return {}


async def _post_or_release(factory: SessionFactory, person: DuePerson, read: Mapping[str, Any]) -> str:
    """Build and post one claimed brief; a build or post that does not land gives the claim back."""
    try:
        posted = _said(read) and bool(await _in_thread(factory, _post, person, read, VIA_SCHEDULE))
    except Exception as failure:  # noqa: BLE001 — logged by class only (docstring); the claim is given back
        logger.error("Daily digest: the brief for person %s in %s failed (%s)",
                     person.user_id, person.workspace_id, type(failure).__name__)
        posted = False
    if posted:
        return POSTED
    await _in_thread(factory, release, person)
    return RELEASED


async def _serve_workspace(factory: SessionFactory, workspace_id: uuid.UUID, people: Sequence[DuePerson]) -> Counter:
    """Claim and post each due person's brief; Auto's Read is made once, at the first claim won."""
    outcomes: Counter = Counter()
    read: Optional[Dict[str, Any]] = None
    for person in people:
        if not await _in_thread(factory, claim, person):
            outcomes[TAKEN] += 1
            continue
        if read is None:
            read = await _generated(factory, workspace_id)
        outcomes[await _post_or_release(factory, person, read)] += 1
    return outcomes


def _open_session() -> Session:
    from core.database.database import SessionLocal

    return SessionLocal()


async def run_digests(
    now: Optional[datetime] = None, *, session_factory: Optional[SessionFactory] = None,
) -> Dict[str, int]:
    """One pass: every due person's digest claimed and posted, or left to its other claimant.

    Never raises: a failure is logged, and the counts say what was done before it. The counts:
    due, posted, taken (claimed already, by an earlier run or a rival) and released.
    """
    factory = session_factory or _open_session
    moment = _utc(now or _now())
    counts: Counter = Counter(dict.fromkeys(COUNT_KEYS, 0))
    try:
        due = await _in_thread(factory, _plan, moment)
        counts["due"] = len(due)
        for workspace_id, people in groupby(due, key=lambda person: person.workspace_id):
            counts.update(await _serve_workspace(factory, workspace_id, list(people)))
    except Exception:
        logger.exception("Daily digest: the run failed (%s)", dict(counts))
    if counts["due"]:
        logger.info("Daily digest: %s", dict(counts))
    return dict(counts)


async def post_now(
    workspace_id: uuid.UUID, user_id: int, *, now: Optional[datetime] = None,
    session_factory: Optional[SessionFactory] = None,
) -> bool:
    """Today's brief for one person, made and posted as ``run_digests`` does, with no claim.

    Each step runs in a worker thread on a session of its own. Never raises: a failure is
    logged by its class only (an error's text can carry the workspace's content or the bound
    parameters), and never the digest's text. True when the post landed.
    """
    factory = session_factory or _open_session
    moment = _utc(now or _now())
    try:
        person = await _in_thread(factory, person_today, workspace_id, user_id, moment)
        read = await _in_thread(factory, _digest_read, workspace_id)
        posted = _said(read) and bool(await _in_thread(factory, _post, person, read, None))
    except Exception as failure:  # noqa: BLE001 — logged by class only (docstring); the ask is answered
        logger.error("Daily digest now: person %s in %s failed (%s)", user_id, workspace_id, type(failure).__name__)
        return False
    logger.info("Daily digest now: person %s in %s %s", user_id, workspace_id, POSTED if posted else "not posted")
    return posted


def register(scheduler: Any) -> bool:
    """The digest job on the leader's scheduler; ``False`` when this worker hosts none.

    No job args: a Redis job store pickles them (``schedule_reconcile._tick``), and the job
    needs none.
    """
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    interval = int(config.DIGEST_JOB_INTERVAL_SECONDS)
    scheduler.add_job(
        run_digests, IntervalTrigger(seconds=interval), id=DIGEST_JOB_ID,
        replace_existing=True, max_instances=1, coalesce=True,
    )
    logger.info("Daily digest: checked every %ds", interval)
    return True


__all__ = [
    "DIGEST_CATCH_UP_MINUTES",
    "DIGEST_JOB_ID",
    "DIGEST_POSTS_KEEP_DAYS",
    "DuePerson",
    "belongs",
    "brief_of",
    "brief_source",
    "claim",
    "date_in_zone",
    "default_digest_time",
    "due_date",
    "due_people",
    "parse_digest_time",
    "person_today",
    "post_brief_message",
    "post_now",
    "prune",
    "recorded_person",
    "register",
    "release",
    "run_digests",
    "schedule_zone",
]
