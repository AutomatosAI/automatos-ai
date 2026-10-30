"""PRD-266 US-013 (§4.5, D24, K12) — the daily digest job: Auto's Read once per person, workspace and local date.

``services/mobile/digest.py`` on Postgres, with ``generate_digest`` and
``deliver_background_message`` patched to record. Proven here:

* When: ``due_people`` at frozen instants in UTC, Europe/Lisbon, America/New_York, Asia/Kolkata
  (+05:30) and Pacific/Auckland, over both 2026 DST changes in Lisbon (29 Mar, 25 Oct) and New
  York (8 Mar, 1 Nov), the window [digest time, digest time + 90 minutes) and one that runs past
  local midnight; ``DIGEST_DEFAULT_TIME`` for a phone that names no time, 08:00 when it is not
  HH:MM.
* Whose schedule: the person's most recently seen active ios/android phone with a timezone
  (``last_seen_at``, then ``created_at``); a phone without a timezone, a paired browser and a
  revoked phone schedule nothing; the ``daily_digest`` push row (theirs, else the workspace's)
  turns it off; a person who no longer belongs to the workspace gets none.
* Once: three phones make one post; a second run in the window posts nothing; two runs racing
  for one claim post once; a post that returns None releases the claim and the next run posts;
  a digest that cannot be made releases it too and nothing raises; claims older than 30 days go.
* What: one ``generate_digest`` per workspace per run; the post goes to the person's Auto thread
  (their Clerk id, else their email) with origin ``daily_digest`` and the local date, the origin
  US-012b announces and never coalesces.
* Wiring: ``register`` adds one job with no args; ``start_schedule_reconcile`` calls it;
  discovery lists ``digest`` on ``GET /api/notification-preferences``.

CI's test database has no ``notification_preferences`` (no model; ``init_test_db`` does not
build it), so the fixture runs the real ``prd128_notifications`` revision's DDL inside the
test's transaction, which is rolled back with it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from config import config  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.mobile_push import MobileDigestPost  # noqa: E402
from services.mobile import digest  # noqa: E402
from services.mobile.digest import DuePerson  # noqa: E402
from services.notification_events import auto_message  # noqa: E402

PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
PREFERENCES = "/api/notification-preferences"
DIGEST_TEXT = "Two tickets wait for review; the Lisbon deck shipped."
# Saturday 24 Oct 2026, 08:00 in Lisbon (WEST, +01:00).
T_LISBON_8 = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
SAT = date(2026, 10, 24)


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The job commits each claim, release and post; here a commit reaches the last savepoint
    only, and every seed commits (``_settle``), so nothing escapes the test's transaction.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    db.commit()


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture(autouse=True)
def preferences_table(db_session):
    """``notification_preferences`` as production has it (prd128's DDL; IF NOT EXISTS)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


@pytest.fixture(autouse=True)
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "DIGEST_DEFAULT_TIME", "08:00")


class _Recorder:
    """``generate_digest`` and ``deliver_background_message``, recorded; ``delivers`` is what each post returns."""

    def __init__(self):
        self.generated: list = []
        self.posts: list = []
        self.delivers = object()
        self.during_generate = None
        self.generate_raises = None

    async def generate(self, db, workspace_id, period="1d", **_):
        self.generated.append(workspace_id)
        if self.during_generate is not None:
            await self.during_generate()
        if self.generate_raises is not None:
            raise self.generate_raises
        return {"text": DIGEST_TEXT, "state_hash": "h", "needs_attention_count": 0}

    def deliver(self, db, **kwargs):
        self.posts.append(kwargs)
        return self.delivers


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    recorded = _Recorder()
    monkeypatch.setattr(digest, "generate_digest", recorded.generate)
    monkeypatch.setattr(digest, "deliver_background_message", recorded.deliver)
    return recorded


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
    _settle(db_session)
    return workspace_id


# --------------------------------------------------------------------------- seeds


def _user(db, *, clerk=None, email=None) -> int:
    tag = uuid.uuid4().hex[:12]
    pk = db.execute(
        text("INSERT INTO users (username, email, clerk_user_id, is_active) VALUES (:u, :e, :c, true) RETURNING id"),
        {"u": f"prd266-digest-{tag}", "e": email or f"prd266-digest-{tag}@t.test", "c": clerk},
    ).scalar()
    _settle(db)
    return pk


def _owner(db, workspace_id, **who) -> int:
    user_pk = _user(db, **who)
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": user_pk, "ws": workspace_id})
    _settle(db)
    return user_pk


def _member(db, workspace_id, role="editor", *, active=True, **who) -> int:
    user_pk = _user(db, **who)
    db.execute(text(
        "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
        "VALUES (CAST(:ws AS uuid), :uid, :role, :active)"),
        {"ws": workspace_id, "uid": user_pk, "role": role, "active": active})
    _settle(db)
    return user_pk


def _phone(db, workspace_id, user_pk, *, zone="Europe/Lisbon", at="08:00", platform="ios",
           status="active", seen=None, created=None) -> MobileDevice:
    tag = uuid.uuid4().hex
    device = MobileDevice(
        workspace_id=uuid.UUID(str(workspace_id)), user_id=user_pk, kind="clerk", status=status,
        install_id=f"prd266-digest-{tag[:12]}", platform=platform, locale="en-IE",
        timezone=zone, digest_time=at, last_seen_at=seen, events=[],
        **({"created_at": created} if created is not None else {}),
    )
    db.add(device)
    _settle(db)
    return device


def _digest_row(db, workspace_id, user_pk, enabled, *, mine=True) -> None:
    db.execute(text(
        "INSERT INTO notification_preferences (workspace_id, user_id, event_type, destination, enabled) "
        "VALUES (CAST(:ws AS uuid), :uid, 'daily_digest', 'push', :enabled)"),
        {"ws": workspace_id, "uid": user_pk if mine else None, "enabled": enabled})
    _settle(db)


def _claims(db, workspace_id) -> list:
    return sorted(
        (row.user_id, row.local_date) for row in
        db.query(MobileDigestPost).filter(MobileDigestPost.workspace_id == uuid.UUID(workspace_id)).all()
    )


def _run(db, at):
    return asyncio.run(digest.run_digests(at, session_factory=lambda: _BorrowedSession(db)))


def _due_in(db, workspace_id, at) -> list:
    return [person for person in digest.due_people(db, at) if str(person.workspace_id) == workspace_id]


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- when: zones and DST


ZONE_CASES = [
    # (zone, digest time, instant in UTC, the local date due, or None)
    ("UTC", "08:00", "2026-10-24T08:00", SAT),
    ("UTC", "08:00", "2026-10-24T07:59", None),
    ("UTC", "08:00", "2026-10-24T09:29", SAT),
    ("UTC", "08:00", "2026-10-24T09:30", None),
    ("UTC", "23:30", "2026-10-25T00:15", SAT),  # the window runs past local midnight
    ("Europe/Lisbon", "08:00", "2026-10-24T07:00", SAT),  # WEST, +01:00
    ("Europe/Lisbon", "08:00", "2026-10-24T08:30", None),
    ("Europe/Lisbon", "08:00", "2026-03-28T08:00", date(2026, 3, 28)),  # WET, the day before
    ("Europe/Lisbon", "08:00", "2026-03-29T07:00", date(2026, 3, 29)),  # WEST from 29 Mar
    ("Europe/Lisbon", "08:00", "2026-03-29T06:59", None),
    ("Europe/Lisbon", "08:00", "2026-03-29T08:30", None),
    ("Europe/Lisbon", "08:00", "2026-10-25T08:00", date(2026, 10, 25)),  # WET again from 25 Oct
    ("Europe/Lisbon", "08:00", "2026-10-25T07:00", None),  # 07:00 local
    ("America/New_York", "08:00", "2026-03-07T13:00", date(2026, 3, 7)),  # EST
    ("America/New_York", "08:00", "2026-03-08T12:00", date(2026, 3, 8)),  # EDT from 8 Mar
    ("America/New_York", "08:00", "2026-03-08T11:59", None),
    ("America/New_York", "08:00", "2026-03-08T13:30", None),  # 09:30 local
    ("America/New_York", "08:00", "2026-11-01T13:00", date(2026, 11, 1)),  # EST from 1 Nov
    ("America/New_York", "08:00", "2026-11-01T12:00", None),  # 07:00 local
    ("America/New_York", "09:00", "2026-10-24T13:00", SAT),
    ("Asia/Kolkata", "08:00", "2026-10-24T02:30", SAT),  # +05:30
    ("Asia/Kolkata", "08:00", "2026-10-24T02:29", None),
    ("Asia/Kolkata", "08:00", "2026-10-24T03:59", SAT),
    ("Asia/Kolkata", "08:00", "2026-10-24T04:00", None),
    ("Pacific/Auckland", "08:00", "2026-10-24T19:00", date(2026, 10, 25)),  # NZDT +13: Sunday there
    ("Pacific/Auckland", "07:00", "2026-10-24T18:00", date(2026, 10, 25)),
    ("Pacific/Auckland", "08:00", "2026-10-24T18:59", None),
    ("Europe/Lisbon", None, "2026-10-24T07:00", SAT),  # no digest time: DIGEST_DEFAULT_TIME
]


@pytest.mark.parametrize("zone,at,instant,expected", ZONE_CASES)
def test_due_people_follows_the_phones_zone_and_dst(db_session, ws, zone, at, instant, expected):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, zone=zone, at=at)

    due = _due_in(db_session, ws, _utc(instant))

    assert due == ([DuePerson(uuid.UUID(ws), person, expected)] if expected else [])


@pytest.mark.parametrize("zone,at,instant,expected", ZONE_CASES)
def test_due_date_is_the_same_rule_without_a_database(zone, at, instant, expected):
    digest_time = digest.parse_digest_time(at) or digest.default_digest_time()
    assert digest.due_date(zone, digest_time, _utc(instant)) == expected


@pytest.mark.parametrize("setting,due_at", [("09:00", "2026-10-24T08:00"), ("9am", "2026-10-24T07:00"),
                                            ("25:00", "2026-10-24T07:00"), ("", "2026-10-24T07:00")])
def test_the_default_time_is_the_setting_or_eight_when_it_is_not_hhmm(db_session, ws, monkeypatch, setting, due_at):
    monkeypatch.setattr(config, "DIGEST_DEFAULT_TIME", setting)
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, at=None)

    assert _due_in(db_session, ws, _utc(due_at)) == [DuePerson(uuid.UUID(ws), person, SAT)]
    assert _due_in(db_session, ws, _utc(due_at) - timedelta(minutes=1)) == []


def test_parse_digest_time_takes_hhmm_only():
    assert digest.parse_digest_time("07:30") == time(7, 30)
    assert [digest.parse_digest_time(v) for v in (None, "", "7:30", "24:00", "07:60", "07:30:00")] == [None] * 6


# --------------------------------------------------------------------------- whose schedule


def test_three_phones_make_one_post(db_session, ws, recorder):
    person = _owner(db_session, ws, clerk=f"user_{uuid.uuid4().hex[:10]}")
    for minutes in (5, 30, 90):
        _phone(db_session, ws, person, seen=T_LISBON_8 - timedelta(minutes=minutes))

    counts = _run(db_session, T_LISBON_8)

    assert len(recorder.posts) == 1 and recorder.generated == [uuid.UUID(ws)]
    assert counts == {"due": 1, "posted": 1, "taken": 0, "released": 0}
    assert _claims(db_session, ws) == [(person, SAT)]


def test_the_most_recently_seen_phones_time_wins(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, zone="Europe/Lisbon", at="08:00", seen=T_LISBON_8 - timedelta(hours=2))
    _phone(db_session, ws, person, zone="America/New_York", at="09:00", seen=T_LISBON_8 - timedelta(minutes=5))
    _phone(db_session, ws, person, zone="Asia/Kolkata", at="08:00", seen=None)

    assert _due_in(db_session, ws, T_LISBON_8) == []  # Lisbon's 08:00: not the phone seen last
    assert _due_in(db_session, ws, _utc("2026-10-24T02:30")) == []  # Kolkata's: never seen
    assert _due_in(db_session, ws, _utc("2026-10-24T13:00")) == [DuePerson(uuid.UUID(ws), person, SAT)]


def test_unseen_phones_fall_back_to_the_newest_one(db_session, ws):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, zone="Europe/Lisbon", created=T_LISBON_8 - timedelta(days=3))
    _phone(db_session, ws, person, zone="America/New_York", created=T_LISBON_8 - timedelta(days=1))

    assert _due_in(db_session, ws, T_LISBON_8) == []
    assert _due_in(db_session, ws, _utc("2026-10-24T12:00")) == [DuePerson(uuid.UUID(ws), person, SAT)]


def test_a_phone_without_a_timezone_is_passed_over_for_one_with_a_timezone(db_session, ws):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, zone="Europe/Lisbon", seen=T_LISBON_8 - timedelta(hours=5))
    _phone(db_session, ws, person, zone=None, seen=T_LISBON_8 - timedelta(minutes=1))

    assert _due_in(db_session, ws, T_LISBON_8) == [DuePerson(uuid.UUID(ws), person, SAT)]


@pytest.mark.parametrize("platform,zone,status,schedules", [
    ("ios", None, "active", False),  # no timezone: nothing
    ("android", None, "active", False),
    ("web", "Europe/Lisbon", "active", False),  # a paired browser never schedules
    ("ios", "Europe/Lisbon", "revoked", False),
    ("ios", "Europe/Lisbon", "pending", False),
    ("android", "Europe/Lisbon", "active", True),
    ("ios", "Europe/Lisbon", "active", True),
], ids=["ios-no-zone", "android-no-zone", "web", "revoked", "pending", "android", "ios"])
def test_only_an_active_phone_with_a_timezone_schedules(db_session, ws, recorder, platform, zone, status, schedules):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person, zone=zone, platform=platform, status=status)

    _run(db_session, T_LISBON_8)

    assert len(recorder.posts) == (1 if schedules else 0)
    assert _claims(db_session, ws) == ([(person, SAT)] if schedules else [])


@pytest.mark.parametrize("mine,workspaces,posts", [
    (False, None, 0),  # their switch off
    (None, False, 0),  # the workspace's off, theirs unset
    (True, False, 1),  # theirs beats the workspace's
    (False, True, 0),
    (None, True, 1),
    (None, None, 1),  # no row means on (K12)
])
def test_the_daily_digest_switch_turns_it_off(db_session, ws, recorder, mine, workspaces, posts):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    if mine is not None:
        _digest_row(db_session, ws, person, mine)
    if workspaces is not None:
        _digest_row(db_session, ws, person, workspaces, mine=False)

    _run(db_session, T_LISBON_8)

    assert len(recorder.posts) == posts


def test_another_events_push_row_leaves_the_digest_on(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    db_session.execute(text(
        "INSERT INTO notification_preferences (workspace_id, user_id, event_type, destination, enabled) "
        "VALUES (CAST(:ws AS uuid), :uid, 'auto_message', 'push', false)"), {"ws": ws, "uid": person})
    _settle(db_session)

    _run(db_session, T_LISBON_8)

    assert len(recorder.posts) == 1


def test_a_person_who_no_longer_belongs_gets_none(db_session, ws, recorder):
    gone = _user(db_session)
    _phone(db_session, ws, gone)
    _phone(db_session, ws, _member(db_session, ws, active=False))
    viewer = _member(db_session, ws, role="viewer")
    _phone(db_session, ws, viewer)

    _run(db_session, T_LISBON_8)

    assert _claims(db_session, ws) == [(viewer, SAT)]  # any member gets theirs; the others none
    assert len(recorder.posts) == 1


# --------------------------------------------------------------------------- once


def test_a_second_run_in_the_window_posts_nothing(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)

    first = _run(db_session, T_LISBON_8)
    second = _run(db_session, T_LISBON_8 + timedelta(minutes=15))

    assert (first["posted"], second["posted"], second["taken"]) == (1, 0, 1)
    assert len(recorder.posts) == 1 and len(recorder.generated) == 1


def test_two_runs_racing_for_one_claim_post_once(db_session, ws, recorder):
    """The rival runs while the first holds the claim and is still making the digest."""
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    rival: list = []

    async def rival_run():
        recorder.during_generate = None
        rival.append(await asyncio.to_thread(asyncio.run, digest.run_digests(
            T_LISBON_8 + timedelta(seconds=1), session_factory=lambda: _BorrowedSession(db_session))))

    recorder.during_generate = rival_run
    first = _run(db_session, T_LISBON_8)

    assert rival == [{"due": 1, "posted": 0, "taken": 1, "released": 0}]
    assert first["posted"] == 1
    assert len(recorder.posts) == 1
    assert _claims(db_session, ws) == [(person, SAT)]


def test_claim_is_won_once(db_session, ws):
    person = DuePerson(uuid.UUID(ws), _owner(db_session, ws), SAT)

    assert (digest.claim(db_session, person), digest.claim(db_session, person)) == (True, False)
    digest.release(db_session, person)
    assert digest.claim(db_session, person) is True


def test_a_post_that_returns_none_releases_the_claim_and_the_next_run_posts(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    recorder.delivers = None

    first = _run(db_session, T_LISBON_8)

    assert first == {"due": 1, "posted": 0, "taken": 0, "released": 1}
    assert _claims(db_session, ws) == []

    recorder.delivers = object()
    second = _run(db_session, T_LISBON_8 + timedelta(minutes=15))

    assert second["posted"] == 1 and len(recorder.posts) == 2
    assert _claims(db_session, ws) == [(person, SAT)]


def test_a_digest_that_cannot_be_made_releases_the_claim_and_never_raises(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    recorder.generate_raises = RuntimeError("snapshot failed")

    counts = _run(db_session, T_LISBON_8)

    assert counts == {"due": 1, "posted": 0, "taken": 0, "released": 1}
    assert recorder.posts == [] and _claims(db_session, ws) == []


def test_a_failed_plan_never_raises(db_session, monkeypatch):
    def broken(db, now):
        raise RuntimeError("no database")

    monkeypatch.setattr(digest, "due_people", broken)

    assert _run(db_session, T_LISBON_8) == {"due": 0, "posted": 0, "taken": 0, "released": 0}


def test_claims_older_than_thirty_days_are_deleted(db_session, ws, recorder):
    person = _owner(db_session, ws)
    for days in (31, 30, 1):
        db_session.add(MobileDigestPost(workspace_id=uuid.UUID(ws), user_id=person,
                                        local_date=SAT - timedelta(days=days)))
    _settle(db_session)

    _run(db_session, T_LISBON_8)

    assert digest.DIGEST_POSTS_KEEP_DAYS == 30
    assert _claims(db_session, ws) == [(person, SAT - timedelta(days=30)), (person, SAT - timedelta(days=1))]


# --------------------------------------------------------------------------- what


def test_the_post_is_auto_s_read_in_each_persons_auto_thread(db_session, ws, recorder):
    clerk_id = f"user_{uuid.uuid4().hex[:10]}"
    email = f"prd266-digest-{uuid.uuid4().hex[:8]}@t.test"
    with_clerk = _owner(db_session, ws, clerk=clerk_id)
    with_email = _member(db_session, ws, email=email)
    _phone(db_session, ws, with_clerk)
    _phone(db_session, ws, with_email)

    _run(db_session, T_LISBON_8)

    assert recorder.generated == [uuid.UUID(ws)]  # once per workspace per run
    by_person = {post["clerk_user_id"]: post for post in recorder.posts}
    assert set(by_person) == {clerk_id, email}
    for post in recorder.posts:
        assert post["workspace_id"] == uuid.UUID(ws)
        assert post["text"] == DIGEST_TEXT
        assert post["source"] == {"origin": "daily_digest", "local_date": "2026-10-24", "via": "schedule"}
        assert "chat_id" not in post  # the person's Auto thread


def test_the_origin_is_the_one_auto_message_announces_and_never_coalesces():
    assert digest.DIGEST_ORIGIN == "daily_digest"
    assert digest.DIGEST_ORIGIN in auto_message.ANNOUNCED_ORIGINS
    assert digest.DIGEST_ORIGIN in auto_message.NEVER_COALESCED


def test_two_workspaces_make_two_digests(db_session, ws, seed_workspace, recorder):
    other = seed_workspace()
    _settle(db_session)
    _phone(db_session, ws, _owner(db_session, ws))
    _phone(db_session, other, _owner(db_session, other))

    _run(db_session, T_LISBON_8)

    assert sorted(map(str, recorder.generated)) == sorted([ws, other])
    assert len(recorder.posts) == 2


# --------------------------------------------------------------------------- wiring


class _FakeScheduler:
    def __init__(self, running=True):
        self.running = running
        self.jobs: list = []

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs.append(SimpleNamespace(func=func, trigger=trigger, kwargs=kwargs))


def test_register_adds_one_job_with_no_args(monkeypatch):
    monkeypatch.setattr(config, "DIGEST_JOB_INTERVAL_SECONDS", 600)
    scheduler = _FakeScheduler()

    assert digest.register(scheduler) is True

    (job,) = scheduler.jobs
    assert job.func is digest.run_digests
    assert job.trigger.interval == timedelta(seconds=600)
    assert "args" not in job.kwargs and "kwargs" not in job.kwargs
    assert job.kwargs == {"id": digest.DIGEST_JOB_ID, "replace_existing": True, "max_instances": 1, "coalesce": True}


def test_register_needs_a_running_scheduler():
    assert digest.register(None) is False
    assert digest.register(_FakeScheduler(running=False)) is False


def test_start_schedule_reconcile_registers_the_digest_on_the_leader(monkeypatch):
    from services import schedule_reconcile

    monkeypatch.setattr(schedule_reconcile, "run_reconcile_once", lambda s, db=None: {})
    scheduler = _FakeScheduler()

    assert asyncio.run(schedule_reconcile.start_schedule_reconcile(scheduler)) is True

    jobs = [job for job in scheduler.jobs if job.kwargs.get("id") == digest.DIGEST_JOB_ID]
    assert len(jobs) == 1
    assert jobs[0].func is digest.run_digests and "args" not in jobs[0].kwargs


def test_the_settings_are_declared():
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert {"DIGEST_JOB_INTERVAL_SECONDS", "DIGEST_DEFAULT_TIME"} <= set(surface)
    assert int(config.DIGEST_JOB_INTERVAL_SECONDS) >= 60
    assert isinstance(config.DIGEST_DEFAULT_TIME, str)


def test_discovery_lists_digest_on_the_preferences_route(monkeypatch):
    from api import notifications
    from config import HOSTED_PUBLIC_API_HOST
    from services.mobile import discovery
    from services.push import consent

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(consent, "get_system_setting", lambda category, key, default=None: default)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())

    document = discovery.build_discovery("http://192.168.1.20:8000/")

    assert "digest" in document["capabilities"]
    assert discovery.CAPABILITY_ROUTES["digest"] == ("GET", PREFERENCES)
    assert {"method": "GET", "path": PREFERENCES} in manifest["routes"]
    assert any(route.path == PREFERENCES and "GET" in route.methods for route in notifications.preferences_router.routes)
