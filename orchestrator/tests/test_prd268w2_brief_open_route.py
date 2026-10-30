"""PRD-268 W2 US-C10b (§6.4, K7, K11, K21) — ``POST /api/proactive/brief/open``: the brief waiting on open.

``api/proactive_brief.py`` (the route) over ``services/brief/on_open.py`` (the rules), on
Postgres with committing sessions. Proven here:

* won, taken and pending: the first open of the day posts the brief into the caller's own Auto
  thread (``[text(Auto's Read), brief@1]``, origin ``daily_digest``, via ``open``, never
  announced) and answers ``{chat_id, message_id, posted: true}``; every later open, from three
  devices and two tabs, answers that message with ``posted`` false; a claim whose winner is
  still building answers ``{posted: false, pending: true}``; opens released together post one;
* a ``local_date`` more than a day from the server's UTC date, and a timezone that is not an
  IANA name, are 422 and claim nothing; a body naming anyone or anything else is 422;
* a person with no phone row gets a brief; a viewer gets one, with no question or approval
  rows, a Mission's plan included (PRD-266 D16, P268W2-RVW-2), while an owner's lists them;
  the brief is in the body's language, else ``Accept-Language``, else English (K7);
* the per-person block makes no model call: Auto's Read books one ``llm_usage`` row at most,
  and none when its cache answers;
* another workspace's message, and another person's, are never answered; a stranger, a caller
  with no person, a super admin outside the workspace and a caller with no workspace are
  refused; a paired device on the local edition opens its own person's brief;
* a build or post that fails is 503, gives the claim back (the next open posts), and is logged
  by its class only;
* P268W2-RVW-1: a claim older than ``BRIEF_CLAIM_STALE_SECONDS`` with no message (its claimant
  died before its post) is taken over by the next open, which posts one brief, and by exactly
  one of three opens released together; a younger one answers pending; a winner whose day was
  taken over while it built posts nothing (it answers the taker's message), and its failure
  gives back nothing;
* P268W2-RVW-3, the rollout gate: where the workspace's effective proactive mode is off (the
  platform mode off, the workspace switch off, or a mode that cannot be read) an open is 404
  ``brief_off`` and claims, reads and builds nothing; in shadow and live it posts as before;
* the route is a plain ``def``, gated by ``require_workspace_permission``, registered, in the
  route manifest, and listed in discovery as ``brief``.

Auto's Read is given (``build.read_digest`` recorded) except where the real ``generate_digest``
runs with the model faked; the schedule is empty, the readiness check held, ``auto_message``'s
dispatch recorded and the route's clock frozen. The brief is switched on as W1's tests switch
check-ins on: the platform ``proactive.mode`` (``held.mode``, live unless a test sets it) where
the workspace's ``settings.proactive.enabled`` is true, as every workspace here is created.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

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

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.proactive_brief as proactive_brief  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.brief import build, on_open  # noqa: E402
from services.brief.claim import claim_brief, post_brief  # noqa: E402
from services.brief.model import PART_TYPE, VIA_OPEN, VIA_SCHEDULE  # noqa: E402
from services.brief.strings import say  # noqa: E402
from services.mobile import digest  # noqa: E402
from services.mobile.digest import DuePerson  # noqa: E402

PATH = "/api/proactive/brief/open"
MANIFEST = ORCH / "reports" / "route-manifest.json"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
TODAY = NOW.date()
LISBON = "Europe/Lisbon"
PROSE = "Two things need you: the oat milk order and a question about Sunday opening."
READ = {"text": PROSE, "state_hash": "state-hash-open", "needs_attention_count": 2}
SNAPSHOT = {"counts": {}, "needs_attention": [], "active": [], "recent_completions": [],
            "needs_attention_count": 0, "state_hash": "state-hash-open-read"}
REAL_READ_DIGEST = build.read_digest
PLATFORM_MODE = ("proactive", "mode")  # the system setting services.proactive.switches reads
SWITCHED_ON = {"proactive": {"enabled": True}}  # a workspace's own switch (settings.proactive.enabled)


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
        pytest.skip(f"the brief route tests need Postgres with the PRD-266 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag, name):
    clerk = f"user_prd268w2bo_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id, name, is_active) VALUES (:u, :e, :c, :n, true) "
        "RETURNING id"), {"u": clerk, "e": f"{clerk}@t.test", "c": clerk, "n": name}).scalar()
    return NS(clerk=clerk, pk=int(pk), email=f"{clerk}@t.test")


def _workspace(s, name, owner_pk):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) "
                   "VALUES (CAST(:id AS uuid), :n, :o, CAST(:s AS jsonb))"),
              {"id": ws, "n": name, "o": owner_pk, "s": json.dumps(SWITCHED_ON)})
    return ws


def _member(s, ws, user_pk, role):
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, :r, true)"), {"w": ws, "u": user_pk, "r": role})


def _ticket(s, ws, title, status, *, done=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, completed_at) "
        "VALUES (CAST(:w AS uuid), :t, :s, 'user', :done) RETURNING id"),
        {"w": ws, "t": title, "s": status, "done": done}).scalar()


def _grant(s, ws, kind, subject_id, *, question=None, reason=None):
    return s.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, status, kind, question_md, reason, "
        "details, requested_at) VALUES (CAST(:w AS uuid), 'board_task', :sid, 'pending', :k, :q, :r, "
        "CAST('{}' AS jsonb), :at) RETURNING id"),
        {"w": ws, "sid": str(subject_id), "k": kind, "q": question, "r": reason, "at": NOW}).scalar()


def _plan(new_session, ws, goal):
    """A Mission waiting for its plan's approval."""
    s = new_session()
    plan = str(s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), :g, 'owner', 'awaiting_approval', 'blocked') RETURNING id"),
        {"w": ws, "g": goal}).scalar())
    s.commit()
    return plan


@pytest.fixture
def held(monkeypatch):
    """Auto's Read given (each read recorded), no schedule, readiness held as all ready (``hold``
    a workspace), every ``auto_message`` notice recorded, the route's clock at ``NOW``, and the
    platform ``proactive.mode`` read as ``mode`` (live; an exception is raised as the read's)."""
    import services.brief.today as today
    import services.readiness.service as readiness
    from core.llm import manager
    from services.notification_events import auto_message
    from services.readiness import items as i
    from services.readiness.probes import ITEMS

    reads, notices = [], []

    def read_digest(db, workspace_id):
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
        checks.put(UUID(ws), readiness.Checked(NOW, tuple(readiness.Result(item, found, 3) for item in ITEMS)))

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(build, "read_digest", read_digest)
    monkeypatch.setattr(today, "ActivityService", NoSchedule)
    monkeypatch.setattr(readiness, "held", checks)
    monkeypatch.setattr(auto_message, "dispatch_auto_message", told)
    rig = NS(reads=reads, notices=notices, hold=hold, mode="live")
    real_setting = manager.read_system_setting

    def setting(category, key):
        if (category, key) != PLATFORM_MODE:
            return real_setting(category, key)
        if isinstance(rig.mode, Exception):
            raise rig.mode
        return rig.mode

    monkeypatch.setattr(on_open, "_now", lambda: NOW)
    monkeypatch.setattr(manager, "read_system_setting", setting)
    return rig


@pytest.fixture
def cafe(engine, new_session, held):
    """Ana owns the café's workspace and another; Bo is an editor and Cy a viewer in the café's;
    Kim belongs to neither. Nobody has a phone. The café's workspace holds a question and an
    approval waiting, and a ticket waiting for review."""
    s = new_session()
    ana, bo, cy, kim = (_person(s, tag, name) for tag, name in
                        (("ana", "Ana Silva"), ("bo", "Bo Lund"), ("cy", "Cy Moreau"), ("kim", "Kim Park")))
    ws, other = _workspace(s, "prd268w2-brief-open", ana.pk), _workspace(s, "prd268w2-brief-open-other", ana.pk)
    _member(s, ws, bo.pk, "editor")
    _member(s, ws, cy.pk, "viewer")
    review = _ticket(s, ws, "Price list for the cafés", "review", done=NOW - timedelta(hours=1))
    blocked = _ticket(s, ws, "Rota for October", "blocked")
    question = _grant(s, ws, "question", blocked, question="Which café opens first?")
    approval = _grant(s, ws, "approval", review, reason="Send the price list to 40 cafés")
    s.commit()
    held.hold(ws)
    held.hold(other)
    yield NS(ws=ws, other=other, ana=ana, bo=bo, cy=cy, kim=kim, review=review, question=question, approval=approval)
    s = new_session.sweep()
    for table in ("llm_usage", "messages", "chats", "mobile_digest_posts", "approval_grants", "board_tasks",
                  "orchestration_runs", "mobile_devices", "workspace_members"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) IN (:a, :b)"), {"a": ws, "b": other})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) IN (:a, :b)"), {"a": ws, "b": other})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": [p.clerk for p in (ana, bo, cy, kim)]})
    s.commit()


class _BookedLLM:
    """The digest's model: each answer books one llm_usage row, as the real manager does."""

    def __init__(self, new_session, workspace_id):
        self.new_session, self.workspace_id = new_session, workspace_id

    async def generate_response(self, messages):
        s = self.new_session()
        s.execute(text(
            "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, input_tokens, "
            "output_tokens, total_tokens, input_cost, output_cost, total_cost) VALUES (CAST(:w AS uuid), 'm', "
            "'openrouter', 'direct', 'digest', 10, 5, 15, 0, 0, 0)"), {"w": str(self.workspace_id)})
        s.commit()
        return NS(content=PROSE)


@pytest.fixture
def model(monkeypatch, new_session, held):
    """Auto's Read through the real ``generate_digest``: a fixed snapshot, no cache, the model
    faked (``_BookedLLM``). Each model manager made is recorded by its workspace."""
    import core.llm as core_llm
    import services.digest_service as digest_service

    made = []

    def factory(**kwargs):
        made.append(str(kwargs["workspace_id"]))
        return _BookedLLM(new_session, kwargs["workspace_id"])

    monkeypatch.setattr(build, "read_digest", REAL_READ_DIGEST)
    monkeypatch.setattr(digest_service, "build_digest_snapshot", lambda db, ws, period="1d": dict(SNAPSHOT))
    monkeypatch.setattr(digest_service, "_default_redis", lambda: None)
    monkeypatch.setattr(core_llm, "create_llm_manager", factory)
    return made


# --------------------------------------------------------------------------- helpers


def _ctx(person, ws, role="user") -> RequestContext:
    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role=role)
    return RequestContext(workspace_id=UUID(ws) if ws else None, user=user, auth_type="clerk")


def _client(new_session, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(proactive_brief.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: new_session()
    return TestClient(app)


def _open(new_session, ctx, day=TODAY, zone=LISBON, *, headers=None, **extra):
    body = {"local_date": day.isoformat() if hasattr(day, "isoformat") else day, "timezone": zone, **extra}
    return _client(new_session, ctx).post(PATH, json=body, headers=headers or {})


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


def _card(row):
    (brief_part,) = [part for part in row.parts if part["type"] == PART_TYPE]
    return brief_part["brief"]


def _asks(brief):
    """The question and approval rows in Needs you."""
    return [row["id"] for row in brief["sections"]["needs_you"]
            if row["id"].startswith(("question:", "approval:")) or (row.get("opens") or {}).get("type") == "question"]


def _usage_rows(new_session, ws):
    return new_session().execute(text(
        "SELECT count(*) FROM llm_usage WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalar()


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


# --------------------------------------------------------------------------- won, taken, pending


def test_the_first_open_posts_the_brief_into_the_callers_auto_thread(cafe, held, new_session):
    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.status_code == 200, response.text
    (brief,) = _briefs(new_session, cafe.ws)
    assert response.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert (brief.kind, brief.user_id) == ("auto", cafe.ana.pk)
    assert [part["type"] for part in brief.parts] == ["text", PART_TYPE]
    assert brief.parts[0]["text"] == _card(brief)["prose"] == PROSE  # Auto's Read, unchanged
    assert {k: brief.source[k] for k in ("origin", "local_date", "via")} == {
        "origin": "daily_digest", "local_date": TODAY.isoformat(), "via": VIA_OPEN}
    assert _card(brief)["date"] == TODAY.isoformat()
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]
    assert held.notices == []  # opening is the person's own action: never announced


def test_three_devices_and_two_tabs_answer_the_same_message(cafe, new_session, monkeypatch):
    callers = [
        (_ctx(cafe.bo, cafe.ws), LISBON, {"Accept-Language": "en-IE"}),  # the web, first tab
        (_ctx(cafe.bo, cafe.ws), LISBON, {}),  # the web, second tab
        (_ctx(cafe.bo, cafe.ws), "Europe/Dublin", {"User-Agent": "Automatos Go (iOS)"}),  # an iPhone
        (_ctx(cafe.bo, cafe.ws), "Europe/London", {"User-Agent": "Automatos Go (Android)"}),  # an Android phone
    ]
    answers = [_open(new_session, ctx, zone=zone, headers=headers) for ctx, zone, headers in callers]
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    device = UserContext(id=cafe.bo.email, email=cafe.bo.email, system_role="user",
                         raw_claims={"source": "mobile_device", "device_id": str(uuid.uuid4()), "user_id": cafe.bo.pk})
    answers.append(_open(new_session, RequestContext(workspace_id=UUID(cafe.ws), user=device, auth_type="device")))

    (brief,) = _briefs(new_session, cafe.ws)
    assert [answer.status_code for answer in answers] == [200] * 5
    assert answers[0].json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert [answer.json() for answer in answers[1:]] == [
        {"chat_id": brief.chat_id, "message_id": brief.id, "posted": False}] * 4
    assert brief.user_id == cafe.bo.pk and _claims(new_session, cafe.ws) == [(cafe.bo.pk, TODAY)]


def test_devices_and_tabs_opening_at_once_post_one_message(cafe, new_session):
    ctx = _ctx(cafe.ana, cafe.ws)

    answers = _all_at_once([lambda: _open(new_session, ctx).json() for _ in range(5)])

    (brief,) = _briefs(new_session, cafe.ws)
    assert [answer.get("posted") for answer in answers].count(True) == 1
    for answer in answers:
        assert answer in ({"chat_id": brief.chat_id, "message_id": brief.id, "posted": answer["posted"]},
                          {"posted": False, "pending": True})
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


def test_a_claim_whose_winner_is_still_building_answers_pending(cafe, held, new_session):
    assert claim_brief(new_session(), cafe.ws, cafe.ana.pk, TODAY) is True  # the winner, still building

    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.status_code == 200, response.text
    assert response.json() == {"posted": False, "pending": True}
    assert _briefs(new_session, cafe.ws) == [] and held.reads == []  # nothing built, nothing posted


def test_a_day_the_scheduled_digest_took_answers_its_message(cafe, new_session):
    s = new_session()
    person = DuePerson(UUID(cafe.ws), cafe.ana.pk, TODAY, LISBON, "en-IE")
    assert claim_brief(s, cafe.ws, cafe.ana.pk, TODAY) is True
    scheduled = post_brief(s, person, digest.brief_of(s, person, dict(READ)), VIA_SCHEDULE)

    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.json() == {"chat_id": str(scheduled.chat_id), "message_id": str(scheduled.id), "posted": False}
    assert len(_briefs(new_session, cafe.ws)) == 1


# --------------------------------------------------------------------------- the 422s


@pytest.mark.parametrize("days", [-2, 2, -400, 400])
def test_a_local_date_more_than_a_day_from_utc_is_422_and_claims_nothing(cafe, held, new_session, days):
    response = _open(new_session, _ctx(cafe.ana, cafe.ws), TODAY + timedelta(days=days))

    assert response.status_code == 422, response.text
    assert response.json() == {"code": "open_out_of_window", "detail": say("en", "open_out_of_window")}
    assert _claims(new_session, cafe.ws) == [] and held.reads == []


@pytest.mark.parametrize("days", [-1, 0, 1])
def test_a_local_date_within_a_day_of_utc_is_served(cafe, new_session, days):
    response = _open(new_session, _ctx(cafe.ana, cafe.ws), TODAY + timedelta(days=days))

    assert response.status_code == 200, response.text
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY + timedelta(days=days))]


@pytest.mark.parametrize("zone", ["Mars/Olympus", "", "UTC+3", "../../etc/passwd", "Europe/" + "x" * 80])
def test_a_timezone_that_is_not_an_iana_name_is_422(cafe, new_session, zone):
    response = _open(new_session, _ctx(cafe.ana, cafe.ws), zone=zone)

    assert response.status_code == 422, response.text
    assert _claims(new_session, cafe.ws) == []


@pytest.mark.parametrize("body", [
    {"timezone": LISBON},
    {"local_date": "2026-13-01", "timezone": LISBON},
    {"local_date": TODAY.isoformat()},
    {"local_date": TODAY.isoformat(), "timezone": LISBON, "user_id": 1},
    {"local_date": TODAY.isoformat(), "timezone": LISBON, "chat_id": str(uuid.uuid4())},
    {"local_date": TODAY.isoformat(), "timezone": LISBON, "workspace_id": str(uuid.uuid4())},
], ids=["no-date", "not-a-date", "no-timezone", "a-person", "a-chat", "a-workspace"])
def test_a_body_that_is_not_the_callers_day_is_422(cafe, new_session, body):
    response = _client(new_session, _ctx(cafe.ana, cafe.ws)).post(PATH, json=body)

    assert response.status_code == 422, response.text
    assert _claims(new_session, cafe.ws) == []


# --------------------------------------------------------------------------- who gets one, and what it says


def test_a_person_with_no_phone_row_gets_a_brief(cafe, new_session):
    with new_session() as s:
        assert s.execute(text("SELECT count(*) FROM mobile_devices WHERE user_id = :u"),
                         {"u": cafe.bo.pk}).scalar() == 0

    response = _open(new_session, _ctx(cafe.bo, cafe.ws))

    assert response.status_code == 200 and response.json()["posted"] is True
    (brief,) = _briefs(new_session, cafe.ws)
    assert brief.user_id == cafe.bo.pk and _card(brief)["date"] == TODAY.isoformat()


def test_a_viewer_gets_a_brief_without_question_or_approval_rows(cafe, new_session):
    plan = _plan(new_session, cafe.ws, "Launch the autumn blend")
    viewer = _open(new_session, _ctx(cafe.cy, cafe.ws))
    owner = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert viewer.status_code == owner.status_code == 200
    cards = {row.user_id: _card(row) for row in _briefs(new_session, cafe.ws)}
    # Not even the Mission's plan: a viewer can't approve it (P268W2-RVW-2), though the badge counts it.
    assert _asks(cards[cafe.cy.pk]) == [] and "Launch the autumn blend" not in cards[cafe.cy.pk]["spoken"]
    assert {f"question:{cafe.question}", f"approval:{cafe.approval}", f"approval:{plan}"} <= set(
        _asks(cards[cafe.ana.pk]))
    ids = [row["id"] for row in cards[cafe.cy.pk]["sections"]["needs_you"]]
    assert f"review:{cafe.review}" in ids  # the ticket waiting for review is the viewer's to see


def test_the_brief_is_in_the_bodys_language_else_accept_language_else_english(cafe, new_session):
    ana = _ctx(cafe.ana, cafe.ws)
    portuguese, english = say("pt-BR", "section_needs_you"), say("en", "section_needs_you")

    _open(new_session, ana, TODAY - timedelta(days=1), locale="pt-BR", headers={"Accept-Language": "en"})
    _open(new_session, ana, TODAY, headers={"Accept-Language": "pt-BR,en;q=0.5"})
    _open(new_session, ana, TODAY + timedelta(days=1))
    _open(new_session, _ctx(cafe.bo, cafe.ws), locale="pt_BR", headers={"Accept-Language": "en-GB"})

    labels = {(row.user_id, row.source["local_date"]): _card(row)["labels"]["sections"]["needs_you"]
              for row in _briefs(new_session, cafe.ws)}
    assert labels == {
        (cafe.ana.pk, (TODAY - timedelta(days=1)).isoformat()): portuguese,
        (cafe.ana.pk, TODAY.isoformat()): portuguese,
        (cafe.ana.pk, (TODAY + timedelta(days=1)).isoformat()): english,
        (cafe.bo.pk, TODAY.isoformat()): english,  # pt_BR is not a tag, so the header chooses
    }


def test_a_refusal_is_in_the_brief_language(cafe, new_session):
    response = _open(new_session, _ctx(cafe.ana, cafe.ws), TODAY + timedelta(days=3), locale="pt-BR")

    assert response.json() == {"code": "open_out_of_window", "detail": say("pt-BR", "open_out_of_window")}


def test_the_per_person_block_makes_no_model_call_and_a_cached_read_none(cafe, model, new_session, monkeypatch):
    import services.digest_service as digest_service

    first = _open(new_session, _ctx(cafe.ana, cafe.ws))
    again = _open(new_session, _ctx(cafe.ana, cafe.ws))
    assert (first.json()["posted"], again.json()["posted"]) == (True, False)
    assert model == [cafe.ws] and _usage_rows(new_session, cafe.ws) == 1  # Auto's Read's own, once

    cached = {"text": "From the cache.", "state_hash": SNAPSHOT["state_hash"], "generated_at": NOW.isoformat()}
    monkeypatch.setattr(digest_service, "_default_redis", lambda: NS(get=lambda key: json.dumps(cached)))
    from_cache = _open(new_session, _ctx(cafe.bo, cafe.ws))

    assert from_cache.json()["posted"] is True
    bo_s = next(row for row in _briefs(new_session, cafe.ws) if row.user_id == cafe.bo.pk)
    assert _card(bo_s)["prose"] == "From the cache."
    assert model == [cafe.ws] and _usage_rows(new_session, cafe.ws) == 1


# --------------------------------------------------------------------------- tenancy and who may ask


def test_another_workspaces_message_and_another_persons_are_never_answered(cafe, new_session):
    s = new_session()
    for ws, user_pk in ((cafe.other, cafe.ana.pk), (cafe.ws, cafe.bo.pk)):
        person = DuePerson(UUID(ws), user_pk, TODAY)
        assert post_brief(s, person, digest.brief_of(s, person, dict(READ)), VIA_OPEN) is not None
    assert claim_brief(s, cafe.ws, cafe.ana.pk, TODAY) is True  # Ana's day here: taken, no message yet

    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.json() == {"posted": False, "pending": True}


def test_a_stranger_and_a_caller_with_no_person_are_403_and_claim_nothing(cafe, new_session):
    api_key = RequestContext(workspace_id=UUID(cafe.ws), user=UserContext(id="api_key", system_role="admin"),
                             auth_type="api_key")
    nobody = NS(clerk=f"user_prd268w2bo_nobody_{uuid.uuid4().hex[:10]}")  # no users row behind it

    assert _open(new_session, _ctx(cafe.kim, cafe.ws)).status_code == 403  # the gate: no role here
    assert _open(new_session, api_key).status_code == 403  # the gate: no person, no role
    past_the_gate = _open(new_session, _ctx(nobody, cafe.ws, "super_admin"))
    assert (past_the_gate.status_code, past_the_gate.json()["code"]) == (403, "open_no_person")
    assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []


def test_a_super_admin_outside_the_workspace_is_403_and_one_with_no_workspace_404(cafe, new_session):
    outside = _open(new_session, _ctx(cafe.kim, cafe.ws, "super_admin"))
    nowhere = _open(new_session, _ctx(cafe.kim, None, "super_admin"))

    assert (outside.status_code, outside.json()["code"]) == (403, "open_not_a_member")
    assert (nowhere.status_code, nowhere.json()["code"]) == (404, "open_no_workspace")
    assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []


# --------------------------------------------------------------------------- a failure gives the claim back


@pytest.mark.parametrize("failure", ["build", "post"])
def test_a_failed_build_or_post_is_503_gives_the_claim_back_and_the_next_open_posts(
        cafe, new_session, monkeypatch, caplog, failure):
    def broken_build(*args, **kwargs):
        raise RuntimeError(f"[parameters: {PROSE}]")

    with monkeypatch.context() as patched:
        if failure == "build":
            patched.setattr(digest, "build_brief", broken_build)
        else:
            patched.setattr(digest, "deliver_background_message", lambda db, **kwargs: None)
        with caplog.at_level(logging.INFO):
            refused = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert refused.status_code == 503
    assert refused.json() == {"code": "open_unavailable", "detail": say("en", "open_unavailable")}
    assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []
    assert PROSE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records if record.name == on_open.__name__)

    again = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert again.json()["posted"] is True and _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


# --------------------------------------------------------------------------- the rollout gate (P268W2-RVW-3)


def _switch(new_session, ws, settings):
    with new_session() as s:
        s.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:w AS uuid)"),
                  {"s": json.dumps(settings), "w": ws})
        s.commit()


@pytest.mark.parametrize("platform,settings", [
    ("off", SWITCHED_ON), ("live", {}), ("shadow", {"proactive": {"enabled": False}}),
    (RuntimeError("system_settings unreadable"), SWITCHED_ON),
], ids=["platform-off", "workspace-never-switched", "workspace-switched-off", "mode-unreadable"])
def test_where_the_proactive_mode_is_off_an_open_is_404_and_claims_reads_and_builds_nothing(
        cafe, held, new_session, monkeypatch, platform, settings):
    built = []
    monkeypatch.setattr(digest, "build_brief", lambda *args, **kwargs: built.append(args))
    held.mode = platform
    _switch(new_session, cafe.ws, settings)

    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.status_code == 404, response.text
    assert response.json() == {"code": "brief_off", "detail": say("en", "brief_off")}
    assert _claims(new_session, cafe.ws) == [] and _briefs(new_session, cafe.ws) == []
    assert built == [] and held.reads == []  # never built: no Auto's Read, no model call


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_in_shadow_and_live_an_open_posts_the_brief_as_before(cafe, held, new_session, mode):
    held.mode = mode

    first = _open(new_session, _ctx(cafe.ana, cafe.ws))
    again = _open(new_session, _ctx(cafe.ana, cafe.ws))

    (brief,) = _briefs(new_session, cafe.ws)
    assert first.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert again.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": False}
    assert brief.source["via"] == VIA_OPEN and _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


# --------------------------------------------------------------------------- the rules, the route, discovery


def test_the_window_is_one_day_either_side_of_the_utc_date():
    late = datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc)
    kiritimati = late.astimezone(timezone(timedelta(hours=14)))  # already 26 Oct there

    assert on_open.DATE_WINDOW_DAYS == 1
    assert [on_open.in_window(late.date() + timedelta(days=d), late) for d in (-2, -1, 0, 1, 2)] == [
        False, True, True, True, False]
    assert on_open.in_window(kiritimati.date() - timedelta(days=1), kiritimati)  # read against the UTC date


def test_an_open_answers_its_message_or_pending():
    assert on_open.Opened(posted=False).to_dict() == {"posted": False, "pending": True}
    assert on_open.Opened(posted=True, chat_id="c", message_id="m").to_dict() == {
        "chat_id": "c", "message_id": "m", "posted": True}


def test_the_route_is_a_plain_def_gated_registered_in_the_manifest_and_listed_in_discovery():
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from modules.policy.roles import WorkspaceRole, workspace_has_permission
    from router_manifest import MANIFEST_ROUTERS, RouterSpec
    from services.mobile import discovery

    assert not inspect.iscoroutinefunction(proactive_brief.open_brief)
    assert RouterSpec("api.proactive_brief") in MANIFEST_ROUTERS
    (route,) = [r for r in proactive_brief.router.routes if r.path == PATH]
    assert route.methods == {"POST"}
    gates = [getattr(d.call, PERMISSION_MARKER_ATTR, None) for d in route.dependant.dependencies]
    assert gates.count(proactive_brief.BRIEF_PERMISSION) == 1  # the authz sweep's class (a)
    assert all(workspace_has_permission(role.value, proactive_brief.BRIEF_PERMISSION) for role in WorkspaceRole)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": PATH} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    assert discovery.CAPABILITY_ROUTES["brief"] == ("POST", PATH)
    assert "brief" not in discovery.CONFIGURED_CAPABILITIES  # listed whenever this build runs


# --------------------------------------------------------------------------- a stranded claim (P268W2-RVW-1)


def _stale_claim(new_session, ws, user_pk, age=None):
    """A claim on the person's day, ``age`` seconds old (past the setting unless given), with no
    message: its claimant died between the claim and the post. Its ``posted_at``."""
    stale = int(config.BRIEF_CLAIM_STALE_SECONDS)
    with new_session() as s:
        held = s.execute(text(
            "INSERT INTO mobile_digest_posts (workspace_id, user_id, local_date, posted_at) "
            "VALUES (CAST(:w AS uuid), :u, :d, now() - make_interval(secs => :age)) RETURNING posted_at"),
            {"w": ws, "u": user_pk, "d": TODAY, "age": stale + 60 if age is None else age}).scalar()
        s.commit()
    return held


def test_an_open_over_a_stale_claim_with_no_message_posts_one_brief(cafe, held, new_session):
    _stale_claim(new_session, cafe.ws, cafe.ana.pk)

    first = _open(new_session, _ctx(cafe.ana, cafe.ws))
    again = _open(new_session, _ctx(cafe.ana, cafe.ws))

    (brief,) = _briefs(new_session, cafe.ws)
    assert first.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert again.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": False}
    assert brief.source["via"] == VIA_OPEN and brief.user_id == cafe.ana.pk
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)] and held.notices == []


def test_an_open_over_a_claim_younger_than_the_setting_answers_pending_and_posts_nothing(cafe, held, new_session):
    _stale_claim(new_session, cafe.ws, cafe.ana.pk, age=int(config.BRIEF_CLAIM_STALE_SECONDS) - 60)

    response = _open(new_session, _ctx(cafe.ana, cafe.ws))

    assert response.status_code == 200, response.text
    assert response.json() == {"posted": False, "pending": True}
    assert _briefs(new_session, cafe.ws) == [] and held.reads == []
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


def test_three_openers_released_together_against_one_stale_claim_post_exactly_one_message(cafe, new_session):
    _stale_claim(new_session, cafe.ws, cafe.ana.pk)
    ctx = _ctx(cafe.ana, cafe.ws)

    answers = _all_at_once([lambda: _open(new_session, ctx).json() for _ in range(3)])

    (brief,) = _briefs(new_session, cafe.ws)
    assert [answer.get("posted") for answer in answers].count(True) == 1
    for answer in answers:
        assert answer in ({"chat_id": brief.chat_id, "message_id": brief.id, "posted": answer.get("posted")},
                          {"posted": False, "pending": True})
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


def _taken_over_while_building(cafe, new_session, monkeypatch, failure=None):
    """The winner's open, whose claim goes stale at once: while it builds, another open takes the
    day over and posts. ``failure`` is raised in the winner's build after that. The winner's
    response and the taker's answer."""
    monkeypatch.setattr(config, "BRIEF_CLAIM_STALE_SECONDS", 0)
    builds, taker = [], []

    def read_digest(db, workspace_id):
        builds.append(str(workspace_id))
        if len(builds) == 1:  # the winner's build; the taker's own build is the second
            taker.append(on_open.open_brief(new_session(), DuePerson(UUID(cafe.ws), cafe.ana.pk, TODAY, LISBON, "en")))
            if failure is not None:
                raise failure
        return dict(READ)

    monkeypatch.setattr(build, "read_digest", read_digest)
    return _open(new_session, _ctx(cafe.ana, cafe.ws)), taker[0]


def test_a_winner_whose_day_was_taken_over_while_it_built_posts_nothing(cafe, new_session, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        winner, taker = _taken_over_while_building(cafe, new_session, monkeypatch)

    (brief,) = _briefs(new_session, cafe.ws)
    assert taker.to_dict() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": True}
    assert winner.status_code == 200, winner.text
    assert winner.json() == {"chat_id": brief.chat_id, "message_id": brief.id, "posted": False}
    assert "LeaseLost" in caplog.text and PROSE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records if record.name == on_open.__name__)
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]


def test_a_winner_whose_day_was_taken_over_and_then_fails_gives_back_nothing(cafe, new_session, monkeypatch):
    winner, taker = _taken_over_while_building(cafe, new_session, monkeypatch, RuntimeError(f"[parameters: {PROSE}]"))

    (brief,) = _briefs(new_session, cafe.ws)
    assert winner.status_code == 503
    assert winner.json() == {"code": "open_unavailable", "detail": say("en", "open_unavailable")}
    assert taker.to_dict()["message_id"] == brief.id
    assert _claims(new_session, cafe.ws) == [(cafe.ana.pk, TODAY)]  # the taker's claim, never deleted

