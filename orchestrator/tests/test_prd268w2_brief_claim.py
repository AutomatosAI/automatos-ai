"""PRD-268 W2 US-C10a (§6.4, FR-10, K7, K11) — one brief per person per day: the shared claim.

The scheduled digest (``services.mobile.digest.run_digests``, via ``schedule``) and the brief on
open (``services.brief.claim``, via ``open``) share one claim per (workspace, person, local date)
in ``mobile_digest_posts``. Proven here, on Postgres with committing sessions:

* three devices and two tabs opening, and the scheduled digest, all at once for the same day,
  post exactly one message, and every claimant then finds that message;
* a phone in another timezone, and a DST change (Lisbon 25 Oct, New York 1 Nov), each give one
  brief per local date;
* a build or a post that fails leaves no claim, and the next tick or the next open posts;
* the scheduled digest's message is ``[text(Auto's Read), brief@1]`` in the person's Auto
  thread, in their phone's language, with source {origin daily_digest, local_date, via
  schedule}; ``post_now`` posts the same brief with no claim and no via, as today;
* a brief posted on open raises no ``auto_message``; the scheduled one still does, and it and
  ``post_now``'s each name their message's id (US-G03: the push opens the thread at the brief);
* ``todays_brief_message`` is the newest of the day's messages in the person's own Auto thread
  in the workspace; ``post_brief`` refuses a via that is neither; ``claim_brief`` is the
  digest's claim; ``due_people`` carries the phone's zone and language without comparing them;
* P268W2-RVW-1, the claim is a lease on its ``posted_at``: a claim older than
  ``BRIEF_CLAIM_STALE_SECONDS`` with no day's message (its claimant died before its post) is
  taken over by the next open, or by a tick, which posts it via schedule; a younger one is
  left alone; a run whose day was taken over while it built posts nothing, and its failure
  gives back nothing; a release with the lease held deletes only that lease.

Auto's Read is given (``digest.generate_digest`` recorded), the schedule is empty (the
calendar's own read faked), the readiness check is held as all ready, and ``auto_message``'s
dispatch is recorded (CI's test database has no ``notifications`` table). CI's test database
has no ``notification_preferences`` either (no model), and ``due_people`` reads the person's
``daily_digest`` switch there, so the fixture runs ``prd128_notifications``' own DDL for that
table, committed, and drops it when it made it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from config import config
from services.brief import claim as brief_claim
from services.brief import on_open
from services.brief.model import PART_TYPE, VIA_OPEN, VIA_SCHEDULE
from services.brief.strings import brief_language, say
from services.mobile import digest
from services.mobile.digest import DuePerson

PROSE = "Two things need you: the oat milk order and a question about Sunday opening."
READ = {"text": PROSE, "state_hash": "state-hash-claim", "needs_attention_count": 2}
LISBON = "Europe/Lisbon"
SAT = date(2026, 10, 24)
# Saturday 24 Oct 2026, 08:00 in Lisbon (WEST, +01:00).
T_LISBON_8 = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
TICK = timedelta(minutes=15)
CLAIMANTS = 6  # three devices and two tabs opening, and the scheduled digest
PREFERENCES_REVISION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "prd128_notifications.py"
PREFERENCES = "notification_preferences"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT status FROM proactive_suggestions LIMIT 1"))
            c.execute(text("SELECT locale, push_categories FROM mobile_devices LIMIT 1"))
            c.execute(text("SELECT posted_at FROM mobile_digest_posts LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the brief claim tests need Postgres with the PRD-266 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag, name):
    clerk = f"user_prd268w2bc_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id, name) VALUES (:u, :e, :c, :n) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk, "n": name}).scalar()
    return clerk, int(pk)


def _workspace(s, name, owner_pk):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), :n, :o)"),
              {"id": ws, "n": name, "o": owner_pk})
    return ws


def _preferences_ddl():
    """``prd128_notifications``' own statements for ``notification_preferences`` (IF NOT EXISTS)."""
    spec = importlib.util.spec_from_file_location(f"revision_{PREFERENCES_REVISION.stem}", PREFERENCES_REVISION)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements = []
    revision.op = NS(execute=statements.append)
    revision.upgrade()
    return [statement for statement in statements if PREFERENCES in statement]


def _made_preferences(s):
    """Make ``notification_preferences`` as production has it when the database lacks it; True
    when this call made it (the fixture then drops it)."""
    if s.execute(text("SELECT to_regclass(:t) IS NOT NULL"), {"t": PREFERENCES}).scalar():
        return False
    for statement in _preferences_ddl():
        s.execute(text(statement))
    return True


@pytest.fixture
def held(monkeypatch):
    """Auto's Read given (each read recorded), no schedule, readiness held as all ready (``hold``
    a workspace), and every ``auto_message`` notice recorded instead of dispatched."""
    import services.brief.today as today
    import services.readiness.service as readiness
    from services.notification_events import auto_message
    from services.readiness import items as i
    from services.readiness.probes import ITEMS

    reads, notices = [], []

    async def generate(db, workspace_id, period="1d", **_):
        reads.append(str(workspace_id))
        return dict(READ)

    class NoSchedule:
        def __init__(self, db, workspace_id):
            pass

        def get_schedule(self, *, range_days=7):
            return {"scheduled": [], "scheduler_active": True}

    async def told(notice):
        notices.append(notice)

    checks = readiness.HeldChecks(600)

    def hold(ws):
        found = i.Finding(i.READY, i.CORE_READY)
        now = datetime.now(timezone.utc)
        checks.put(UUID(ws), readiness.Checked(now, tuple(readiness.Result(item, found, 3) for item in ITEMS)))

    monkeypatch.setattr(config, "DIGEST_DEFAULT_TIME", "08:00")
    monkeypatch.setattr(digest, "generate_digest", generate)
    monkeypatch.setattr(today, "ActivityService", NoSchedule)
    monkeypatch.setattr(readiness, "held", checks)
    monkeypatch.setattr(auto_message, "dispatch_auto_message", told)
    return NS(reads=reads, notices=notices, hold=hold)


@pytest.fixture
def cafe(engine, new_session, held, monkeypatch):
    """Ana owns the café's workspace and another; Bo is an editor in the café's. No phones yet.
    The schedule serves these two workspaces only: the CI database is shared, and a phone
    another file left behind is never this file's to post for. The push switches' table is made
    when missing (no row: every switch is on)."""
    s = new_session()
    made = _made_preferences(s)
    (ana_clerk, ana), (bo_clerk, bo) = _person(s, "ana", "Ana Silva"), _person(s, "bo", "Bo Lund")
    ws, other = _workspace(s, "prd268w2-brief-claim", ana), _workspace(s, "prd268w2-brief-claim-other", ana)
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, 'editor', true)"), {"w": ws, "u": bo})
    s.commit()
    held.hold(ws)
    held.hold(other)
    every_due = digest.due_people
    monkeypatch.setattr(digest, "due_people",
                        lambda db, now: [p for p in every_due(db, now) if str(p.workspace_id) in (ws, other)])
    yield NS(ws=ws, other=other, ana=ana, bo=bo)
    s = new_session.sweep()
    for table in ("messages", "chats", "mobile_digest_posts", "mobile_devices", "workspace_members"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) IN (:a, :b)"), {"a": ws, "b": other})
    if made:
        s.execute(text("DROP TABLE IF EXISTS notification_preferences"))
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) IN (:a, :b)"), {"a": ws, "b": other})
    s.execute(text("DELETE FROM users WHERE clerk_user_id IN (:a, :b)"), {"a": ana_clerk, "b": bo_clerk})
    s.commit()


# --------------------------------------------------------------------------- helpers


def _phone(new_session, ws, user_pk, zone=LISBON, *, locale="en-IE", seen=T_LISBON_8):
    s = new_session()
    s.execute(text(
        "INSERT INTO mobile_devices (id, workspace_id, user_id, kind, status, install_id, platform, locale, "
        "timezone, digest_time, last_seen_at) VALUES (CAST(:id AS uuid), CAST(:w AS uuid), :u, 'clerk', 'active', "
        ":i, 'ios', :l, :z, '08:00', :at)"),
        {"id": str(uuid.uuid4()), "w": ws, "u": user_pk, "i": f"prd268w2bc-{uuid.uuid4().hex[:12]}",
         "l": locale, "z": zone, "at": seen})
    s.commit()


def _run(new_session, at):
    return asyncio.run(digest.run_digests(at, session_factory=new_session))


def _open(new_session, person, read=READ):
    """One open, as the brief route makes it: claim the day; the winner builds and posts via open."""
    s = new_session()
    if not brief_claim.claim_brief(s, person.workspace_id, person.user_id, person.local_date):
        return "taken"
    posted = brief_claim.post_brief(s, person, digest.brief_of(s, person, dict(read)), VIA_OPEN)
    return "posted" if posted is not None else "not posted"


def _briefs(new_session, ws):
    """Every daily digest message in the workspace, oldest first."""
    with new_session() as s:
        return s.execute(text(
            "SELECT CAST(m.id AS text) AS id, CAST(m.chat_id AS text) AS chat_id, m.source, m.parts, "
            "c.user_id, c.kind FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE m.workspace_id = CAST(:w AS uuid) AND m.source->>'origin' = 'daily_digest' "
            "ORDER BY m.created_at, m.id"), {"w": ws}).all()


def _claims(new_session, ws):
    with new_session() as s:
        return sorted(tuple(row) for row in s.execute(text(
            "SELECT user_id, local_date FROM mobile_digest_posts WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}))


def _told(held, ws):
    """The ``auto_message`` notices raised for the workspace's chats."""
    return [notice for notice in held.notices if notice.workspace_id == ws]


def _all_at_once(work):
    """Run each callable in its own thread, released together; their answers, or the first error."""
    gate, answers, errors = threading.Barrier(len(work)), [None] * len(work), []

    def go(index, call):
        try:
            gate.wait()
            answers[index] = call()
        except Exception as exc:  # noqa: BLE001 — re-raised in the test's thread below
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(n, call)) for n, call in enumerate(work)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return answers


# --------------------------------------------------------------------------- once a day


def test_three_devices_and_two_tabs_and_the_schedule_claiming_the_same_day_post_one_message(cafe, new_session):
    for minutes in (5, 30, 90):
        _phone(new_session, cafe.ws, cafe.ana, seen=T_LISBON_8 - timedelta(minutes=minutes))
    person = DuePerson(UUID(cafe.ws), cafe.ana, SAT, LISBON, "en-IE")
    opens = [lambda: _open(new_session, person) for _ in range(CLAIMANTS - 1)]

    answers = _all_at_once([*opens, lambda: _run(new_session, T_LISBON_8)])

    (brief,) = _briefs(new_session, cafe.ws)
    scheduled_won = brief.source["via"] == VIA_SCHEDULE
    assert answers[:-1].count("posted") + int(scheduled_won) == 1
    assert answers[:-1].count("taken") == CLAIMANTS - 1 - (0 if scheduled_won else 1)
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)]
    found = brief_claim.todays_brief_message(new_session(), cafe.ws, cafe.ana, SAT)
    assert str(found.id) == brief.id  # every claimant now answers the same message


def test_a_phone_in_another_timezone_gives_one_brief_per_its_local_date(cafe, new_session):
    _phone(new_session, cafe.ws, cafe.ana, "Pacific/Auckland", seen=None)
    sunday_there = datetime(2026, 10, 24, 19, 0, tzinfo=timezone.utc)  # 08:00 NZDT on Sunday 25 Oct

    for at in (sunday_there, sunday_there + TICK, sunday_there + 4 * TICK):
        _run(new_session, at)
    person = digest.person_today(new_session(), UUID(cafe.ws), cafe.ana, sunday_there + TICK)

    assert (person.local_date, person.zone) == (date(2026, 10, 25), "Pacific/Auckland")
    assert _open(new_session, person) == "taken"
    (brief,) = _briefs(new_session, cafe.ws)
    assert brief.source["local_date"] == "2026-10-25"  # the phone's date, not the server's 24 Oct
    assert _claims(new_session, cafe.ws) == [(cafe.ana, date(2026, 10, 25))]


@pytest.mark.parametrize("zone,start,end,days", [
    (LISBON, "2026-10-24T05:00", "2026-10-25T10:00", ["2026-10-24", "2026-10-25"]),  # WEST to WET
    ("America/New_York", "2026-10-31T10:00", "2026-11-01T15:00", ["2026-10-31", "2026-11-01"]),  # EDT to EST
], ids=["lisbon-25-oct", "new-york-1-nov"])
def test_a_dst_change_gives_one_brief_per_local_date(cafe, new_session, zone, start, end, days):
    _phone(new_session, cafe.ws, cafe.ana, zone, seen=None)
    at, stop = (datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc) for stamp in (start, end))

    while at < stop:
        _run(new_session, at)
        at += TICK

    briefs = _briefs(new_session, cafe.ws)
    assert [brief.source["local_date"] for brief in briefs] == days
    assert {brief.source["via"] for brief in briefs} == {VIA_SCHEDULE}
    assert [day.isoformat() for _, day in _claims(new_session, cafe.ws)] == days


# --------------------------------------------------------------------------- a failure gives the claim back


@pytest.mark.parametrize("failure", ["build", "post"])
@pytest.mark.parametrize("then", ["tick", "open"])
def test_a_failed_build_or_post_leaves_no_claim_and_the_next_tick_or_open_posts(
        cafe, new_session, monkeypatch, failure, then):
    _phone(new_session, cafe.ws, cafe.ana)

    def broken_build(*args, **kwargs):
        raise RuntimeError("the brief could not be built")

    with monkeypatch.context() as patched:
        if failure == "build":
            patched.setattr(digest, "build_brief", broken_build)
        else:
            patched.setattr(digest, "deliver_background_message", lambda db, **kwargs: None)
        _run(new_session, T_LISBON_8)
        assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []

    if then == "tick":
        _run(new_session, T_LISBON_8 + TICK)
    else:
        assert _open(new_session, DuePerson(UUID(cafe.ws), cafe.ana, SAT, LISBON, "en-IE")) == "posted"

    (brief,) = _briefs(new_session, cafe.ws)
    assert brief.source["via"] == (VIA_SCHEDULE if then == "tick" else VIA_OPEN)
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)]


def test_a_failed_brief_is_logged_by_its_class_only(cafe, new_session, monkeypatch, caplog):
    """A database error's text carries its bound parameters, the brief's words among them."""
    _phone(new_session, cafe.ws, cafe.ana)

    def broken_build(*args, **kwargs):
        raise RuntimeError(f"[parameters: {PROSE}]")

    monkeypatch.setattr(digest, "build_brief", broken_build)
    with caplog.at_level(logging.INFO, logger=digest.__name__):
        _run(new_session, T_LISBON_8)

    assert "RuntimeError" in caplog.text and PROSE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records if record.name == digest.__name__)
    assert _claims(new_session, cafe.ws) == []


# --------------------------------------------------------------------------- what is posted


def test_the_scheduled_digests_message_carries_brief_at_1_and_its_text_is_auto_s_read(cafe, held, new_session):
    _phone(new_session, cafe.ws, cafe.ana, locale="pt-BR")

    _run(new_session, T_LISBON_8)

    (brief,) = _briefs(new_session, cafe.ws)
    text_part, brief_part = brief.parts
    assert text_part == {"type": "text", "text": PROSE}  # Auto's Read, unchanged
    assert brief_part["type"] == PART_TYPE == "brief@1"
    card = brief_part["brief"]
    assert (card["v"], card["date"], card["prose"], card["state_hash"]) == (1, "2026-10-24", PROSE, READ["state_hash"])
    portuguese = say(brief_language("pt-BR"), "section_needs_you")
    assert card["labels"]["sections"]["needs_you"] == portuguese != say(brief_language("en"), "section_needs_you")
    assert {k: brief.source[k] for k in ("origin", "local_date", "via")} == {
        "origin": "daily_digest", "local_date": "2026-10-24", "via": VIA_SCHEDULE}
    assert (brief.kind, brief.user_id) == ("auto", cafe.ana)
    assert held.reads.count(cafe.ws) == 1  # one Read for the workspace, no second model call


def test_post_now_posts_the_same_brief_with_no_claim_and_no_via(cafe, held, new_session):
    _phone(new_session, cafe.ws, cafe.ana)

    assert asyncio.run(digest.post_now(UUID(cafe.ws), cafe.ana, now=T_LISBON_8 - timedelta(hours=1),
                                       session_factory=new_session)) is True

    (brief,) = _briefs(new_session, cafe.ws)
    assert [part["type"] for part in brief.parts] == ["text", PART_TYPE]
    assert brief.parts[0]["text"] == brief.parts[1]["brief"]["prose"] == PROSE
    assert brief.source["local_date"] == "2026-10-24" and "via" not in brief.source
    assert _claims(new_session, cafe.ws) == []
    assert [notice.chat_id for notice in _told(held, cafe.ws)] == [brief.chat_id]  # announced, as today


def test_a_brief_posted_on_open_raises_no_auto_message(cafe, held, new_session):
    assert _open(new_session, DuePerson(UUID(cafe.ws), cafe.ana, SAT, LISBON, "en-IE")) == "posted"

    (brief,) = _briefs(new_session, cafe.ws)
    assert brief.source["via"] == VIA_OPEN and brief.kind == "auto"
    assert _told(held, cafe.ws) == []


def test_the_scheduled_brief_still_raises_auto_message(cafe, held, new_session):
    _phone(new_session, cafe.ws, cafe.ana)

    _run(new_session, T_LISBON_8)

    (brief,) = _briefs(new_session, cafe.ws)
    assert [(notice.chat_id, notice.origin, notice.excerpt) for notice in _told(held, cafe.ws)] == [
        (brief.chat_id, "daily_digest", PROSE)]


def test_the_scheduled_brief_and_post_now_announce_their_message_ids(cafe, held, new_session):
    """PRD-268 US-G03: each announced brief names its message, so its push opens the thread at it."""
    _phone(new_session, cafe.ws, cafe.ana)

    _run(new_session, T_LISBON_8)
    assert asyncio.run(digest.post_now(UUID(cafe.ws), cafe.ana, now=T_LISBON_8, session_factory=new_session)) is True

    scheduled, asked = _briefs(new_session, cafe.ws)
    assert (scheduled.source["via"], "via" in asked.source) == (VIA_SCHEDULE, False)
    assert [(notice.chat_id, notice.message_id) for notice in _told(held, cafe.ws)] == [
        (scheduled.chat_id, scheduled.id), (asked.chat_id, asked.id)]


# --------------------------------------------------------------------------- the day's message, the claim


def _post(new_session, ws, user_pk, day, via=VIA_OPEN):
    s = new_session()
    person = DuePerson(UUID(ws), user_pk, day)
    return brief_claim.post_brief(s, person, digest.brief_of(s, person, dict(READ)), via)


def test_todays_brief_message_is_the_newest_of_the_day_in_the_persons_own_auto_thread(cafe, new_session):
    _post(new_session, cafe.ws, cafe.ana, SAT - timedelta(days=1))
    _post(new_session, cafe.ws, cafe.bo, SAT)
    _post(new_session, cafe.other, cafe.ana, SAT)
    first = _post(new_session, cafe.ws, cafe.ana, SAT)
    newest = _post(new_session, cafe.ws, cafe.ana, SAT, VIA_SCHEDULE)

    s = new_session()
    assert first is not None and str(brief_claim.todays_brief_message(s, cafe.ws, cafe.ana, SAT).id) == str(newest.id)
    assert brief_claim.todays_brief_message(s, cafe.ws, cafe.ana, SAT + timedelta(days=1)) is None
    bo_s = brief_claim.todays_brief_message(s, cafe.ws, cafe.bo, SAT)
    assert bo_s is not None and bo_s.chat_id != newest.chat_id  # Bo's own thread, never Ana's
    other_s = brief_claim.todays_brief_message(s, cafe.other, cafe.ana, SAT)
    assert other_s is not None and other_s.workspace_id == UUID(cafe.other) != newest.workspace_id


@pytest.mark.parametrize("via", [None, "", "now", "OPEN", "schedule "])
def test_post_brief_refuses_a_via_that_is_neither_and_writes_nothing(cafe, new_session, via):
    s = new_session()
    person = DuePerson(UUID(cafe.ws), cafe.ana, SAT)

    with pytest.raises(ValueError, match="posted via one of"):
        brief_claim.post_brief(s, person, {"prose": PROSE}, via)

    assert _briefs(new_session, cafe.ws) == []
    assert new_session().execute(text("SELECT count(*) FROM chats WHERE workspace_id = CAST(:w AS uuid)"),
                                 {"w": cafe.ws}).scalar() == 0


def test_claim_brief_is_the_digests_claim(cafe, new_session):
    s = new_session()

    assert (brief_claim.claim_brief(s, cafe.ws, cafe.ana, SAT), brief_claim.claim_brief(s, cafe.ws, cafe.ana, SAT)) \
        == (True, False)
    assert digest.claim(s, DuePerson(UUID(cafe.ws), cafe.ana, SAT)) is False  # the same row
    digest.release(s, DuePerson(UUID(cafe.ws), cafe.ana, SAT))
    assert brief_claim.claim_brief(s, UUID(cafe.ws), cafe.ana, SAT) is True


def test_due_people_carries_the_phones_zone_and_language_without_comparing_them(cafe, new_session):
    _phone(new_session, cafe.ws, cafe.ana, locale="pt-BR")

    (person,) = [p for p in digest.due_people(new_session(), T_LISBON_8) if str(p.workspace_id) == cafe.ws]

    assert (person.zone, person.locale) == (LISBON, "pt-BR")
    assert person == DuePerson(UUID(cafe.ws), cafe.ana, SAT)


# --------------------------------------------------------------------------- a stranded claim (P268W2-RVW-1)


@pytest.fixture
def read_given(monkeypatch):
    """Auto's Read given to the brief on open too (``build.read_digest``), as the schedule's is."""
    from services.brief import build

    monkeypatch.setattr(build, "read_digest", lambda db, workspace_id: dict(READ))


def _ana(cafe):
    return DuePerson(UUID(cafe.ws), cafe.ana, SAT, LISBON, "en-IE")


def _stale_seconds():
    return int(config.BRIEF_CLAIM_STALE_SECONDS)


def _stale_claim(new_session, ws, user_pk, day, age=None):
    """A claim on the person's day, ``age`` seconds old (past the setting unless given), with no
    message: its claimant died between the claim and the post. Its ``posted_at``."""
    with new_session() as s:
        held = s.execute(text(
            "INSERT INTO mobile_digest_posts (workspace_id, user_id, local_date, posted_at) "
            "VALUES (CAST(:w AS uuid), :u, :d, now() - make_interval(secs => :age)) RETURNING posted_at"),
            {"w": ws, "u": user_pk, "d": day, "age": _stale_seconds() + 60 if age is None else age}).scalar()
        s.commit()
    return held


def _age(new_session, ws, user_pk, day):
    """The person's claim made older than the setting."""
    with new_session() as s:
        s.execute(text(
            "UPDATE mobile_digest_posts SET posted_at = now() - make_interval(secs => :age) "
            "WHERE workspace_id = CAST(:w AS uuid) AND user_id = :u AND local_date = :d"),
            {"w": ws, "u": user_pk, "d": day, "age": _stale_seconds() + 60})
        s.commit()


def _lease(new_session, ws, user_pk, day):
    with new_session() as s:
        return s.execute(text(
            "SELECT posted_at FROM mobile_digest_posts WHERE workspace_id = CAST(:w AS uuid) AND user_id = :u "
            "AND local_date = :d"), {"w": ws, "u": user_pk, "d": day}).scalar()


def test_a_stale_claim_with_no_message_is_taken_over_by_an_open_and_a_following_tick_posts_nothing_more(
        cafe, held, read_given, new_session):
    _phone(new_session, cafe.ws, cafe.ana)
    left = _stale_claim(new_session, cafe.ws, cafe.ana, SAT)

    opened = on_open.open_brief(new_session(), _ana(cafe), now=T_LISBON_8)
    counts = _run(new_session, T_LISBON_8 + TICK)

    (brief,) = _briefs(new_session, cafe.ws)
    assert opened.to_dict() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert brief.source["via"] == VIA_OPEN and brief.user_id == cafe.ana
    assert counts == {"due": 1, "posted": 0, "taken": 1, "released": 0}
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)] and _lease(new_session, cafe.ws, cafe.ana, SAT) > left
    assert _told(held, cafe.ws) == []  # posted on open: never announced


def test_a_scheduled_tick_over_a_stale_claim_takes_it_over_and_posts_it_via_schedule(cafe, held, new_session):
    _phone(new_session, cafe.ws, cafe.ana)
    left = _stale_claim(new_session, cafe.ws, cafe.ana, SAT)

    counts = _run(new_session, T_LISBON_8)

    (brief,) = _briefs(new_session, cafe.ws)
    assert counts == {"due": 1, "posted": 1, "taken": 0, "released": 0}
    assert brief.source["via"] == VIA_SCHEDULE and (brief.kind, brief.user_id) == ("auto", cafe.ana)
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)] and _lease(new_session, cafe.ws, cafe.ana, SAT) > left
    assert [notice.chat_id for notice in _told(held, cafe.ws)] == [brief.chat_id]  # scheduled: announced


def test_a_tick_over_a_claim_younger_than_the_setting_posts_nothing(cafe, held, new_session):
    _phone(new_session, cafe.ws, cafe.ana)
    left = _stale_claim(new_session, cafe.ws, cafe.ana, SAT, age=_stale_seconds() - 60)

    counts = _run(new_session, T_LISBON_8)

    assert counts == {"due": 1, "posted": 0, "taken": 1, "released": 0}
    assert _briefs(new_session, cafe.ws) == [] and held.reads == []
    assert _lease(new_session, cafe.ws, cafe.ana, SAT) == left


@pytest.mark.parametrize("then", ["posts", "fails"])
def test_a_run_whose_day_was_taken_over_while_it_built_posts_nothing_and_gives_back_nothing(
        cafe, held, read_given, new_session, monkeypatch, then):
    """The run holds the day and is making Auto's Read when its claim goes stale (at once here)
    and an open takes it over and posts. The run then posts nothing, or fails and gives back
    nothing: the claim stays the taker's."""
    _phone(new_session, cafe.ws, cafe.ana)
    monkeypatch.setattr(config, "BRIEF_CLAIM_STALE_SECONDS", 0)
    opens = []

    async def generate(db, workspace_id, period="1d", **_):
        opens.append(await asyncio.to_thread(on_open.open_brief, new_session(), _ana(cafe), T_LISBON_8))
        if then == "fails":
            raise RuntimeError("the digest could not be made")
        return dict(READ)

    monkeypatch.setattr(digest, "generate_digest", generate)
    counts = _run(new_session, T_LISBON_8)

    (brief,) = _briefs(new_session, cafe.ws)
    assert [opened.to_dict() for opened in opens] == [{"chat_id": brief.chat_id, "message_id": brief.id,
                                                       "posted": True}]
    assert brief.source["via"] == VIA_OPEN
    assert counts == {"due": 1, "posted": 0, "taken": int(then == "posts"), "released": int(then == "fails")}
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)]


def test_a_delivery_that_raises_after_the_look_gives_the_stamped_claim_back(cafe, new_session, monkeypatch):
    """The look before the post re-stamps the claim, so the run's own release (the first stamp)
    would miss it: the post gives the stamped claim back itself, and the next tick posts."""
    _phone(new_session, cafe.ws, cafe.ana)

    def broken_delivery(db, **kwargs):
        raise RuntimeError("the delivery failed")

    with monkeypatch.context() as patched:
        patched.setattr(digest, "deliver_background_message", broken_delivery)
        counts = _run(new_session, T_LISBON_8)
    assert counts == {"due": 1, "posted": 0, "taken": 0, "released": 1}
    assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []

    _run(new_session, T_LISBON_8 + TICK)

    (brief,) = _briefs(new_session, cafe.ws)
    assert brief.source["via"] == VIA_SCHEDULE and _claims(new_session, cafe.ws) == [(cafe.ana, SAT)]


def test_the_lease_is_the_claims_posted_at_and_only_its_holder_posts_or_gives_it_back(cafe, new_session):
    s = new_session()
    person = DuePerson(UUID(cafe.ws), cafe.ana, SAT)
    held = digest.lease(s, person)
    assert held is not None and digest.lease(s, person) is None
    assert digest.take_over(s, person) is None  # fresh: nobody takes it over

    _age(new_session, cafe.ws, cafe.ana, SAT)
    taken = digest.take_over(s, person)

    assert taken is not None and digest.take_over(s, person) is None  # one taker
    with pytest.raises(digest.LeaseLost):
        digest.still_held(s, person, held)
    digest.release(s, person, held=held)
    assert _claims(new_session, cafe.ws) == [(cafe.ana, SAT)]  # the taker's claim, never the loser's to give back
    stamped = digest.still_held(s, person, taken)
    assert stamped >= taken
    digest.release(s, person, held=stamped)
    assert _claims(new_session, cafe.ws) == []

