"""PRD-266 US-013 (§4.5, D24, FR-17, K12) — the daily digest: Auto's Read in each person's Auto thread.

``run_digests`` runs on the leader's scheduler every ``config.DIGEST_JOB_INTERVAL_SECONDS``
(``register``, called from ``services.schedule_reconcile.start_schedule_reconcile``):

1. Claims older than ``DIGEST_POSTS_KEEP_DAYS`` are deleted.
2. ``due_people``: a person's schedule in a workspace follows their most recently seen active
   ios/android phone there that has a timezone (``last_seen_at``, then ``created_at``): its
   ``digest_time``, else ``config.DIGEST_DEFAULT_TIME``. They are due while their local time is
   within [digest time, digest time + ``DIGEST_CATCH_UP_MINUTES``) on that local date. A phone
   without a timezone and a paired browser schedule nothing; neither does a person whose
   ``daily_digest`` push row is off (theirs, else the workspace's; no row means on: US-014a,
   K12) or who no longer belongs to the workspace (``services.push.recipients``).
3. Each due person's (workspace, person, local date) is claimed in ``mobile_digest_posts``
   (``INSERT … ON CONFLICT DO NOTHING``, committed): a second claim from any tick, worker or
   replica posts nothing. Several phones make one digest.
4. The text is ``services.digest_service.generate_digest`` (Auto's Read, cached per workspace
   state; once per workspace per run), booked to the workspace like the Command Center's.
5. ``deliver_background_message`` posts it as Auto's message in the person's Auto thread
   (origin ``daily_digest``), which raises ``auto_message`` (US-012b; never coalesced), so the
   phone's push carries it (D24). A post that does not land releases the claim, so a later
   tick in the window tries again.

``run_digests`` is async, on the leader's loop; every database step (and the digest's own
snapshot) runs in a worker thread through ``anyio.to_thread.run_sync``, each on a session of
its own. It never raises and never logs the digest's text.

``post_now`` is the same digest for one person on request ("Send me today's brief now",
``POST /api/v1/mobile/digest/now``, ``services.mobile.digest_now``): the same text, posted the
same way (origin ``daily_digest``, their local date by ``local_today``), with no claim read or
written, so the scheduled digest still goes out. Their ``daily_digest`` switch is not read: it
is an explicit ask. It never raises; a failure is logged by its class only.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from itertools import groupby
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from config import config
from core.models.core import User
from core.models.mobile_devices import MobileDevice, MobileDeviceStatus
from core.models.mobile_push import MobileDigestPost
from services.chat_messenger import deliver_background_message
from services.digest_service import generate_digest
from services.notification_events.auto_message import DAILY_DIGEST as DIGEST_ORIGIN
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

# Per (workspace, person): the most recently seen active phone with a timezone, unless the
# person's daily_digest push row is off (theirs, else the workspace's; none means on). The
# digest's switch is read as the fan-out reads a push switch. Bound parameters only.
_SCHEDULES_SQL = text(
    "SELECT DISTINCT ON (d.workspace_id, d.user_id) d.workspace_id, d.user_id, d.timezone, d.digest_time "
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


class DuePerson(NamedTuple):
    """One digest owed: the workspace, the person (``users.id``) and their local date."""

    workspace_id: uuid.UUID
    user_id: int
    local_date: date


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
    for workspace_id, user_id, zone_name, digest_time in rows:
        local = due_date(zone_name, parse_digest_time(digest_time) or default, now)
        person = DuePerson(uuid.UUID(str(workspace_id)), int(user_id), local) if local else None
        if person is not None and belongs(db, person.workspace_id, person.user_id):
            due.append(person)
    return sorted(due)


def local_today(db: Session, workspace_id: uuid.UUID, user_id: int, now: datetime) -> date:
    """The person's local date at ``now``: in the zone of the phone their schedule follows, else UTC.

    That phone is ``due_people``'s: their most recently seen active ios/android phone in the
    workspace with a timezone (``last_seen_at``, then ``created_at``).
    """
    zone_name = (
        db.query(MobileDevice.timezone)
        .filter(
            MobileDevice.workspace_id == workspace_id,
            MobileDevice.user_id == user_id,
            MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
            MobileDevice.platform.in_(DIGEST_PLATFORMS),
            MobileDevice.timezone.isnot(None),
            MobileDevice.timezone != "",
        )
        .order_by(MobileDevice.last_seen_at.desc().nulls_last(), MobileDevice.created_at.desc(), MobileDevice.id)
        .limit(1)
        .scalar()
    )
    zone = _zone(zone_name) or timezone.utc
    return _utc(now).astimezone(zone).date()


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


def _digest_text(db: Session, workspace_id: uuid.UUID) -> str:
    """Auto's Read for the workspace; runs in a worker thread, so its own loop runs the coroutine."""
    result = asyncio.run(generate_digest(db, workspace_id))
    return str((result or {}).get("text") or "").strip()


def _recorded_person(db: Session, user_id: int) -> Optional[str]:
    """The person as ``resolve_recorded_person`` reads them: their Clerk id, else their email."""
    row = db.query(User.clerk_user_id, User.email).filter(User.id == user_id).first()
    if row is None:
        return None
    return (row.clerk_user_id or "").strip() or (row.email or "").strip() or None


def _post(db: Session, person: DuePerson, digest: str) -> bool:
    """Post the digest into the person's Auto thread; True when it landed."""
    recorded = _recorded_person(db, person.user_id)
    if recorded is None:
        return False
    message = deliver_background_message(
        db,
        workspace_id=person.workspace_id,
        text=digest,
        source={"origin": DIGEST_ORIGIN, "local_date": person.local_date.isoformat()},
        clerk_user_id=recorded,
    )
    if message is None:
        return False
    _commit_live_notice(db)
    return True


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


async def _generated(factory: SessionFactory, workspace_id: uuid.UUID) -> str:
    """The workspace's digest, or '' when it could not be made (logged)."""
    try:
        return await _in_thread(factory, _digest_text, workspace_id)
    except Exception:
        logger.exception("Daily digest: workspace %s's digest could not be made", workspace_id)
        return ""


async def _post_or_release(factory: SessionFactory, person: DuePerson, digest: str) -> str:
    """Post one claimed digest; a post that does not land gives the claim back."""
    try:
        posted = bool(digest) and bool(await _in_thread(factory, _post, person, digest))
    except Exception:
        logger.exception("Daily digest: the post for person %s in %s failed", person.user_id, person.workspace_id)
        posted = False
    if posted:
        return POSTED
    await _in_thread(factory, release, person)
    return RELEASED


async def _serve_workspace(factory: SessionFactory, workspace_id: uuid.UUID, people: Sequence[DuePerson]) -> Counter:
    """Claim and post each due person's digest; the text is made once, at the first claim won."""
    outcomes: Counter = Counter()
    digest: Optional[str] = None
    for person in people:
        if not await _in_thread(factory, claim, person):
            outcomes[TAKEN] += 1
            continue
        if digest is None:
            digest = await _generated(factory, workspace_id)
        outcomes[await _post_or_release(factory, person, digest)] += 1
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
    """Today's digest for one person, made and posted as ``run_digests`` does, with no claim.

    Each step runs in a worker thread on a session of its own. Never raises: a failure is
    logged by its class only (an error's text can carry the workspace's content or the bound
    parameters), and never the digest's text. True when the post landed.
    """
    factory = session_factory or _open_session
    moment = _utc(now or _now())
    try:
        local = await _in_thread(factory, local_today, workspace_id, user_id, moment)
        brief = await _in_thread(factory, _digest_text, workspace_id)
        person = DuePerson(workspace_id, user_id, local)
        posted = bool(brief) and bool(await _in_thread(factory, _post, person, brief))
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
    "claim",
    "default_digest_time",
    "due_date",
    "due_people",
    "local_today",
    "parse_digest_time",
    "post_now",
    "prune",
    "register",
    "release",
    "run_digests",
]
