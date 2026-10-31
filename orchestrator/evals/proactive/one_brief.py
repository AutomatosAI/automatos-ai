"""PRD-268 §6.7 (US-C13c) — the one-brief eval: every device a person opens, and still one brief a day.

There is no gold set: the eval drives W2's claim (``services.brief.claim.claim_brief``) against the
configured database and counts the briefs it made there. Each open is one claimant, as the brief on
open makes one: it claims the person's (workspace, person, local date), its local date the one its
device's clock shows in its own zone; the claimant that wins builds the person's brief
(``digest.brief_of`` over a Read given here, so no model is called) and posts it via ``open``
(``claim.post_brief``); every other finds the day taken. The opens of a wave are released together,
each on a session of its own in a thread of its own, so the claim is raced as devices race it.

Three scenarios, one scratch person each, laid out from the hour the run starts (``scenarios``):

* three devices and two tabs opening at once, in Lisbon;
* a phone in Auckland and a tab in Lisbon opening together every 3 hours for a day, so for part of
  each day their local dates differ and for part they agree;
* a phone and a tab opening every half hour from 6 hours before a DST change to 6 hours after it,
  the next change found in Lisbon's zone (else New York's, else Sydney's; none in the next 400 days
  means the eval cannot run), so the night crosses local midnight and the repeated or skipped hour.

The briefs are then read back from the database, not from what the claimants answered: every
``daily_digest`` message in the scratch workspace, as (person, local date), and scored with
``score_one_brief`` against the (person, local date) of every open. A second brief for one person
and local date is a duplicate, and any duplicate is a miss; so is a day opened with no brief, or a
brief for a day nobody opened. The audit lines say each scenario's opens, days and briefs, and each
day that did not get exactly one.

It writes: one scratch workspace and its three people (named ``one-brief-eval-<hex>``), their
claims, their Auto threads and their briefs, all removed when the run ends, whatever its outcome
(a removal that fails is logged with the workspace's id). Before the first write it logs a warning
naming the database and its host. Run it against a test database. No brief it posts is announced
(a brief posted via open never is), and the scratch people have no phone. A day in
the future is claimed like any other (the claim reads no clock), and no claim of the run is old
enough for the digest job's prune to remove it mid-run.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from functools import partial
from types import MappingProxyType
from typing import Callable, FrozenSet, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.core import Chat, Message, User
from core.models.mobile_push import MobileDigestPost
from core.models.workspaces import Workspace
from evals.proactive.runtime import EvalCannotRun
from evals.proactive.scoring import Score, score_one_brief
from services.brief.claim import claim_brief, post_brief
from services.brief.model import VIA_OPEN
from services.mobile.digest import DuePerson, brief_of, date_in_zone
from services.notification_events.auto_message import DAILY_DIGEST

logger = logging.getLogger(__name__)

__all__ = ["NAME", "Opening", "Scenario", "briefs_in", "evaluate", "next_offset_change", "open_session", "scenarios"]

NAME = "one_brief"
HOME_ZONE = "Europe/Lisbon"
AWAY_ZONE = "Pacific/Auckland"
DST_ZONES = ("Europe/Lisbon", "America/New_York", "Australia/Sydney")
DEVICES, TABS = 3, 2
AWAY_SPAN, AWAY_STEP = timedelta(hours=24), timedelta(hours=3)
DST_SPAN, DST_STEP = timedelta(hours=6), timedelta(minutes=30)
DST_SEARCH, DST_SEARCH_STEP = timedelta(days=400), timedelta(hours=1)
NO_DST_CHANGE = "None of {zones} changes its clocks in the next {days} days, so there is no DST change to open across."
# The Read every winner's brief is built over (``digest.brief_of``): given, so no model is called.
EVAL_READ: Mapping[str, object] = MappingProxyType({
    "text": "This brief was posted by the one-brief eval.", "state_hash": "one-brief-eval", "needs_attention_count": 0})
SCRATCH_PREFIX = "one-brief-eval"
SCRATCH_DOMAIN = "one-brief.eval.invalid"
ASSISTANT = "assistant"
# A wave whose claimants are not all released within this long is a run that cannot finish.
RELEASE_SECONDS = 30
SCENARIO_LINE = "{name}: {opens} opens on {days} local dates, {briefs} briefs"
WRITES_TO = ("[evals.one_brief] writing a scratch workspace and %d people to database %s on %s; "
             "they are removed when the run ends")
LOCAL_SOCKET = "the local socket"
DAY_LINE = "   {date}: {briefs} briefs"

SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class Opening:
    """One device opening Automatos: which, the IANA zone its clock is in, and when (UTC)."""

    device: str
    zone: str
    at: datetime

    @property
    def local_date(self) -> date:
        """The date its clock shows (``digest.date_in_zone``)."""
        return date_in_zone(self.zone, self.at)


@dataclass(frozen=True)
class Scenario:
    """One person's opens: each wave's openings are released together."""

    name: str
    waves: Tuple[Tuple[Opening, ...], ...]

    def local_dates(self) -> FrozenSet[date]:
        """Every local date one of its opens claims."""
        return frozenset(opening.local_date for wave in self.waves for opening in wave)


@dataclass(frozen=True)
class _Scratch:
    """The run's own workspace and its people, one per scenario."""

    workspace_id: uuid.UUID
    people: Tuple[int, ...]


def open_session() -> Session:
    """A session on the configured database."""
    from core.database.database import SessionLocal

    return SessionLocal()


def next_offset_change(zone_name: str, after: datetime) -> Optional[datetime]:
    """The first whole hour after ``after`` at which ``zone_name``'s UTC offset differs from the hour
    before (a DST change), looked for up to ``DST_SEARCH`` ahead; None when there is none."""
    zone = ZoneInfo(zone_name)
    at = after.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    offset, last = at.astimezone(zone).utcoffset(), at + DST_SEARCH
    while at < last:
        at += DST_SEARCH_STEP
        if at.astimezone(zone).utcoffset() != offset:
            return at
    return None


def _every(start: datetime, stop: datetime, step: timedelta) -> List[datetime]:
    count = int((stop - start) / step)
    return [start + step * n for n in range(count + 1)]


def _devices_and_tabs(at: datetime) -> Scenario:
    phones = tuple(Opening(f"phone {n}", HOME_ZONE, at) for n in range(1, DEVICES + 1))
    tabs = tuple(Opening(f"tab {n}", HOME_ZONE, at) for n in range(1, TABS + 1))
    return Scenario("three devices and two tabs", ((*phones, *tabs),))


def _another_timezone(start: datetime) -> Scenario:
    waves = tuple((Opening("phone", AWAY_ZONE, at), Opening("tab", HOME_ZONE, at))
                  for at in _every(start, start + AWAY_SPAN, AWAY_STEP))
    return Scenario("a phone in another timezone", waves)


def _dst_change(start: datetime) -> Scenario:
    for zone in DST_ZONES:
        change = next_offset_change(zone, start)
        if change is not None:
            waves = tuple((Opening("phone", zone, at), Opening("tab", zone, at))
                          for at in _every(change - DST_SPAN, change + DST_SPAN, DST_STEP))
            return Scenario(f"a DST change ({zone}, {change.isoformat()})", waves)
    raise EvalCannotRun(NO_DST_CHANGE.format(zones=", ".join(DST_ZONES), days=DST_SEARCH.days))


def scenarios(now: datetime) -> Tuple[Scenario, ...]:
    """The three scenarios, laid out from the whole hour ``now`` falls in."""
    start = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return _devices_and_tabs(start), _another_timezone(start), _dst_change(start)


def _scratch(factory: SessionFactory, count: int) -> _Scratch:
    """A workspace and ``count`` people made for the run, committed; the first owns it."""
    tag = f"{SCRATCH_PREFIX}-{uuid.uuid4().hex[:12]}"
    db = factory()
    try:
        url = db.get_bind().url
        logger.warning(WRITES_TO, count, url.database, url.host or LOCAL_SOCKET)
        people = [User(username=f"{tag}-{n}", email=f"{tag}-{n}@{SCRATCH_DOMAIN}", clerk_user_id=f"{tag}-{n}",
                       name=f"Eval {n}") for n in range(1, count + 1)]
        db.add_all(people)
        db.flush()
        workspace = Workspace(id=uuid.uuid4(), name=tag, owner_id=people[0].id)
        db.add(workspace)
        db.commit()
        return _Scratch(workspace_id=workspace.id, people=tuple(int(person.id) for person in people))
    finally:
        db.close()


def _remove(factory: SessionFactory, scratch: _Scratch) -> None:
    """Delete everything the run made: the workspace's briefs, Auto threads and claims, the
    workspace, and its people."""
    db = factory()
    try:
        for model in (Message, Chat, MobileDigestPost):
            db.query(model).filter(model.workspace_id == scratch.workspace_id).delete(synchronize_session=False)
        db.query(Workspace).filter(Workspace.id == scratch.workspace_id).delete(synchronize_session=False)
        db.query(User).filter(User.id.in_(scratch.people)).delete(synchronize_session=False)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("[evals.one_brief] the scratch workspace %s and its people %s were not removed",
                         scratch.workspace_id, scratch.people)
        raise
    finally:
        db.close()


def _open(factory: SessionFactory, workspace_id: uuid.UUID, user_id: int, opening: Opening) -> None:
    """One open: claim the day; the winner builds the brief and posts it via open. What was posted is
    read back from the database afterwards, never taken from here."""
    person = DuePerson(workspace_id, user_id, opening.local_date, opening.zone)
    db = factory()
    try:
        if claim_brief(db, workspace_id, user_id, person.local_date):
            post_brief(db, person, brief_of(db, person, EVAL_READ), VIA_OPEN)
    finally:
        db.close()


def _at_once(calls: Sequence[Callable[[], None]]) -> None:
    """Each call in a thread of its own, released together; the first error is raised."""
    gate = threading.Barrier(len(calls), timeout=RELEASE_SECONDS)

    def released(call: Callable[[], None]) -> None:
        gate.wait()
        call()

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        list(pool.map(released, calls))


def _drive(factory: SessionFactory, scratch: _Scratch, plan: Sequence[Scenario]) -> None:
    for person, scenario in zip(scratch.people, plan):
        for wave in scenario.waves:
            _at_once([partial(_open, factory, scratch.workspace_id, person, opening) for opening in wave])


def briefs_in(db: Session, workspace_id: uuid.UUID) -> List[Tuple[int, str]]:
    """Every brief in the workspace, as (person, local date): Auto's ``daily_digest`` messages in
    their chats, whatever the chat."""
    rows = (
        db.query(Chat.user_id, Message.source["local_date"].astext)
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(Chat.workspace_id == workspace_id, Message.workspace_id == workspace_id,
                Message.role == ASSISTANT, Message.source["origin"].astext == DAILY_DIGEST)
        .all()
    )
    return [(int(user_id), str(local_date)) for user_id, local_date in rows]


def _read(factory: SessionFactory, workspace_id: uuid.UUID) -> List[Tuple[int, str]]:
    db = factory()
    try:
        return briefs_in(db, workspace_id)
    finally:
        db.close()


def _audit(plan: Sequence[Scenario], people: Sequence[int], briefs: Sequence[Tuple[int, str]]) -> Tuple[str, ...]:
    """Each scenario's opens, days and briefs, and each of its days that did not get exactly one."""
    counts = Counter(briefs)
    lines: List[str] = []
    for person, scenario in zip(people, plan):
        posted = {day: n for (who, day), n in counts.items() if who == person}
        days = {day.isoformat() for day in scenario.local_dates()}
        opens = sum(len(wave) for wave in scenario.waves)
        lines.append(SCENARIO_LINE.format(name=scenario.name, opens=opens, days=len(days), briefs=sum(posted.values())))
        lines += [DAY_LINE.format(date=day, briefs=posted.get(day, 0))
                  for day in sorted(days | set(posted)) if posted.get(day, 0) != 1]
    return tuple(lines)


def _run(factory: SessionFactory, plan: Sequence[Scenario]) -> Score:
    scratch = _scratch(factory, len(plan))
    try:
        _drive(factory, scratch, plan)
        briefs = _read(factory, scratch.workspace_id)
    finally:
        _remove(factory, scratch)
    expected = {(person, day.isoformat()) for person, scenario in zip(scratch.people, plan)
                for day in scenario.local_dates()}
    return replace(score_one_brief(briefs, expected), audit=_audit(plan, scratch.people, briefs))


async def evaluate(session_factory: Optional[SessionFactory] = None) -> Score:
    """Drive the claim through every scenario on the configured database, count the briefs it made,
    and remove what the run wrote."""
    plan = scenarios(datetime.now(timezone.utc))
    return await asyncio.to_thread(_run, session_factory or open_session, plan)
