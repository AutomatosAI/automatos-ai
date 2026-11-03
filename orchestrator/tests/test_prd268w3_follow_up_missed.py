"""PRD-268 P268W3-RVW-1 (§6.3, K9, K10, K11) — a follow-up the scheduler never ran stops being pending.

``ScheduledTaskService.execute_task`` returns before it counts the run when only the trial
can pay (``scheduled_task_steps.skipped_for_credits``), and a one-shot whose time passes
the misfire grace while no scheduler runs it is dropped as missed. Either way the row stays
'active' for ever. Proven here, against Postgres:

* ``platform_schedule_follow_up`` refuses with ``no_background``, and writes nothing, when
  ``services.credits.routing.background_allowed`` says no, and when it cannot be read
  (logged with ``logger.exception``): a follow-up that cannot fire is never promised;
* an 'active' follow-up further past its time than ``ONE_SHOT_MISFIRE_GRACE_SECONDS`` lists
  as 'missed' through the tool and the route (``status=all``), and ``status=pending``
  leaves it out;
* five missed follow-ups do not stop a sixth under a pending cap of five, and a missed one
  still counts toward the 24-hour caps (it was scheduled);
* cancelling a missed one is refused (the route's 409 ``follow_up_missed``, the tool's
  ``already_missed``) and leaves the row as it was;
* one inside its grace is about to fire: it stays 'pending', and counts as pending.

``agent_scheduled_tasks`` has no model, so CI's create_all database has none: the module
makes it, in the shape the migrations give it, when it is missing, and drops it after.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
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

import api.proactive_follow_ups as routes  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.tools.discovery import handlers_follow_ups  # noqa: E402
from services.proactive import follow_ups  # noqa: E402
from services.proactive.decision_text import STRINGS  # noqa: E402
from services.proactive.follow_up_strings import say  # noqa: E402
from services.scheduled_task_service import ONE_SHOT_MISFIRE_GRACE_SECONDS  # noqa: E402

SCHEDULE, LIST, CANCEL = "platform_schedule_follow_up", "platform_list_follow_ups", "platform_cancel_follow_up"
ROUTE_LIST, ROUTE_CANCEL = "/api/proactive/follow-ups", "/api/proactive/follow-ups/{}/cancel"
REASON = "Check whether the Lisbon card moved"
# The scheduler's misfire grace: a one-shot further past its time than this is never run.
GRACE = timedelta(seconds=ONE_SHOT_MISFIRE_GRACE_SECONDS)
SCHEDULED_TASKS = """
    CREATE TABLE IF NOT EXISTS agent_scheduled_tasks (
        id SERIAL PRIMARY KEY, workspace_id UUID NOT NULL, created_by_agent_id INTEGER,
        target_agent_id INTEGER, task_type VARCHAR(20) NOT NULL DEFAULT 'one_shot',
        description TEXT NOT NULL, schedule VARCHAR(100) NOT NULL, max_runs INTEGER,
        run_count INTEGER NOT NULL DEFAULT 0, status VARCHAR(20) NOT NULL DEFAULT 'active',
        last_run_at TIMESTAMPTZ, next_run_at TIMESTAMPTZ, last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_by_user_id VARCHAR(255), deliver_as VARCHAR(20) NOT NULL DEFAULT 'chat', payload JSONB,
        origin_chat_id UUID, for_user_id INTEGER,
        CONSTRAINT ck_scheduled_task_deliver_as CHECK (deliver_as IN ('chat', 'board_task', 'follow_up')))
"""


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT paused FROM proactive_person_settings LIMIT 1"))
            c.execute(text("SELECT actor_type FROM audit_logs LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the missed follow-up tests need Postgres with the PRD-205 and PRD-268 tables: {exc}")
    with eng.begin() as c:
        built = not c.execute(text("SELECT to_regclass('agent_scheduled_tasks') IS NOT NULL")).scalar()
        if built:
            c.execute(text(SCHEDULED_TASKS))
    yield eng
    if built:
        with eng.begin() as c:
            c.execute(text("SET LOCAL lock_timeout = '5s'"))
            c.execute(text("DROP TABLE IF EXISTS agent_scheduled_tasks"))
    eng.dispose()


def _setup(s):
    """Ana, her workspace (proactive on), its Auto and her chat."""
    clerk = f"user_prd268w3mi_ana_{uuid.uuid4().hex[:10]}"
    ana = int(s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar())
    ws, chat = str(uuid.uuid4()), str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) VALUES (CAST(:id AS uuid), "
                   "'prd268w3 missed', :o, CAST(:settings AS jsonb))"),
              {"id": ws, "o": ana, "settings": json.dumps({"proactive": {"enabled": True}})})
    auto = int(s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent, configuration, slug) "
        "VALUES ('Auto', 'custom', CAST(:w AS uuid), 'active', true, CAST('{}' AS json), :slug) RETURNING id"),
        {"w": ws, "slug": f"auto-{ws}"}).scalar())
    s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind) "
                   "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), 'Lisbon', 'private', 'user')"),
              {"id": chat, "u": ana, "w": ws})
    return NS(clerk=clerk, ana=ana, ws=ws, auto=auto, chat=chat)


def _sweep(new_session, ws, clerk):
    s = new_session.sweep()
    for table in ("audit_logs", "agent_scheduled_tasks", "proactive_person_settings", "chats", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = :w"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = :w"), {"w": ws})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = :c"), {"c": clerk})
    s.commit()


@pytest.fixture
def team(engine, new_session, monkeypatch):
    """Ana's workspace, live, whose background work more than the trial pays for (``team.background``
    says otherwise, or raises when it holds an exception). The scheduler's calls are recorded."""
    from core.llm import manager
    from services.credits import routing
    from services.scheduled_task_service import ScheduledTaskService

    s = new_session()
    team = _setup(s)
    s.commit()
    team.new_session, team.background, team.registered, team.unscheduled = new_session, {"value": True}, [], []

    def background_allowed(db, workspace_id):
        if isinstance(team.background["value"], Exception):
            raise team.background["value"]
        return team.background["value"]

    monkeypatch.setattr(routing, "background_allowed", background_allowed)
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: "live" if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(ScheduledTaskService, "_register_with_scheduler",
                        lambda self, *args: team.registered.append(args))
    monkeypatch.setattr(ScheduledTaskService, "_reschedule", lambda self, *args: team.unscheduled.append(args))
    yield team
    _sweep(new_session, team.ws, team.clerk)


def _seed(team, *, due, created=None, status="active"):
    """One of Ana's follow-ups in her chat, due ``due`` from now (negative: past), created ``created`` from now."""
    now = datetime.now(timezone.utc)
    at = now + due
    s = team.new_session()
    row = s.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, task_type, "
        "description, schedule, next_run_at, origin_chat_id, deliver_as, for_user_id, status, created_at) VALUES "
        "(CAST(:w AS uuid), :a, :a, 'one_shot', :r, :at_text, :at, CAST(:c AS uuid), 'follow_up', :u, :s, :made) "
        "RETURNING id"), {"w": team.ws, "a": team.auto, "r": REASON, "at_text": at.isoformat(), "at": at,
                          "c": team.chat, "u": team.ana, "s": status, "made": now + (created or timedelta(0))}).scalar()
    s.commit()
    return int(row)


def _tool(team, action, **params):
    from modules.tools.discovery.follow_up_context import follow_up_params

    context = {"driving_user_id": str(team.ana), "conversation_id": team.chat, "user_id": "user_clerk"}
    sent = follow_up_params(action, {"_agent_id": team.auto, **params}, context)
    handler = {SCHEDULE: handlers_follow_ups.schedule_follow_up, LIST: handlers_follow_ups.list_follow_ups,
               CANCEL: handlers_follow_ups.cancel_follow_up}[action]
    return asyncio.run(handler(team.new_session(), team.ws, sent))


def _client(team) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    user = UserContext(id=team.clerk, clerk_user_id=team.clerk, system_role="user")
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=UUID(team.ws), user=user, auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: team.new_session()
    return TestClient(app)


def _route_statuses(team, **params):
    listed = _client(team).get(ROUTE_LIST, params=params)
    assert listed.status_code == 200, listed.text
    return {item["id"]: item["status"] for item in listed.json()["follow_ups"]}


def _tool_statuses(team, **params):
    listed = _tool(team, LIST, **params)
    assert listed["success"] is True, listed
    return {item["id"]: item["status"] for item in listed["follow_ups"]}


def _rows(team):
    s = team.new_session()
    return s.execute(text("SELECT id, status, updated_at FROM agent_scheduled_tasks "
                          "WHERE workspace_id = CAST(:w AS uuid) ORDER BY id"), {"w": team.ws}).fetchall()


def _refused(reply, code):
    return reply.get("success") is False and reply.get("error_code") == code


# --------------------------------------------------------------------------- never promised


def test_schedule_is_refused_with_nothing_written_when_only_the_trial_can_pay(team):
    team.background["value"] = False

    reply = _tool(team, SCHEDULE, reason=REASON, in_minutes=30)

    assert _refused(reply, "no_background") and reply["error"] == say("en", "no_background")
    assert _rows(team) == [] and team.registered == []
    assert say("pt-BR", "no_background") != say("en", "no_background")


def test_schedule_is_refused_with_nothing_written_when_the_credit_check_cannot_be_read(team, caplog):
    team.background["value"] = RuntimeError("balance read failed")

    with caplog.at_level(logging.ERROR, logger=handlers_follow_ups.__name__):
        reply = _tool(team, SCHEDULE, reason=REASON, in_minutes=30)

    assert _refused(reply, "no_background")
    assert _rows(team) == [] and team.registered == []
    (record,) = [r for r in caplog.records if r.name == handlers_follow_ups.__name__ and r.levelno == logging.ERROR]
    assert isinstance(record.exc_info[1], RuntimeError)


def test_schedule_goes_ahead_when_more_than_the_trial_pays(team):
    reply = _tool(team, SCHEDULE, reason=REASON, in_minutes=30)

    assert reply["success"] is True and len(_rows(team)) == 1


# --------------------------------------------------------------------------- missed, not pending


def test_a_follow_up_past_its_grace_lists_as_missed_and_pending_leaves_it_out(team):
    missed = _seed(team, due=-(GRACE + timedelta(minutes=5)), created=-timedelta(hours=1))
    pending = _seed(team, due=timedelta(minutes=40))

    assert _tool_statuses(team, status="all") == {pending: "pending", missed: "missed"}
    assert _route_statuses(team, status="all") == {pending: "pending", missed: "missed"}
    assert _tool_statuses(team) == {pending: "pending"}
    assert _route_statuses(team) == {pending: "pending"}
    assert _route_statuses(team, status="all", chat_id=team.chat) == {pending: "pending", missed: "missed"}


def test_the_pending_come_first_and_a_missed_one_sorts_with_those_that_came(team):
    missed = _seed(team, due=-timedelta(hours=2), created=-timedelta(hours=3))
    fired = _seed(team, due=-timedelta(hours=1), created=-timedelta(hours=3), status="completed")
    pending = _seed(team, due=timedelta(minutes=40))

    listed = _client(team).get(ROUTE_LIST, params={"status": "all"}).json()["follow_ups"]

    assert [(item["id"], item["status"]) for item in listed] == [
        (pending, "pending"), (fired, "fired"), (missed, "missed")]


def test_one_inside_its_grace_is_about_to_fire_and_stays_pending(team):
    due = _seed(team, due=-timedelta(seconds=60))

    assert GRACE > timedelta(seconds=60)
    assert _tool_statuses(team) == {due: "pending"}
    assert _route_statuses(team, status="all") == {due: "pending"}


def test_five_missed_follow_ups_do_not_stop_a_sixth_under_a_pending_cap_of_five(team):
    assert follow_ups.platform_caps().max_pending == 5
    for hours in range(5):
        _seed(team, due=-timedelta(hours=26 + hours), created=-timedelta(hours=30 + hours))

    reply = _tool(team, SCHEDULE, reason=REASON, in_minutes=30)

    assert reply["success"] is True, reply
    assert len(_rows(team)) == 6 and len(team.registered) == 1


def test_a_missed_follow_up_is_not_pending_but_still_counts_toward_the_days_caps(team):
    from modules.proactive.follow_up_caps import CapBroken, FollowUpCaps

    _seed(team, due=-timedelta(hours=1), created=-timedelta(hours=2))
    now = datetime.now(timezone.utc)
    request = follow_ups.FollowUpRequest(workspace_id=UUID(team.ws), user_id=team.ana, chat_id=UUID(team.chat),
                                         agent_id=team.auto, at=now + timedelta(hours=1), reason=REASON)

    pending_cap = follow_ups.schedule_follow_up(team.new_session(), request, FollowUpCaps(max_pending=1), now)

    assert isinstance(pending_cap, int)  # the missed one is not pending
    assert len(_rows(team)) == 2
    day_cap = follow_ups.schedule_follow_up(team.new_session(), request, FollowUpCaps(max_per_day=2), now)
    assert day_cap == CapBroken("max_per_day", 2)  # the missed one and the one just scheduled
    assert len(_rows(team)) == 2


def test_one_inside_its_grace_counts_as_pending(team):
    from modules.proactive.follow_up_caps import CapBroken, FollowUpCaps

    _seed(team, due=-timedelta(seconds=60))
    now = datetime.now(timezone.utc)
    request = follow_ups.FollowUpRequest(workspace_id=UUID(team.ws), user_id=team.ana, chat_id=UUID(team.chat),
                                         agent_id=team.auto, at=now + timedelta(hours=1), reason=REASON)

    assert follow_ups.schedule_follow_up(team.new_session(), request, FollowUpCaps(max_pending=1), now) == CapBroken(
        "max_pending", 1)


# --------------------------------------------------------------------------- nothing to cancel


def test_cancelling_a_missed_follow_up_on_the_route_is_409_and_leaves_it_unchanged(team):
    missed = _seed(team, due=-(GRACE + timedelta(minutes=5)), created=-timedelta(hours=1))
    before = _rows(team)

    refused = _client(team).post(ROUTE_CANCEL.format(missed))
    pt = _client(team).post(ROUTE_CANCEL.format(missed), headers={"Accept-Language": "pt-BR"})

    assert refused.status_code == 409, refused.text
    assert refused.json() == {"code": routes.MISSED, "detail": STRINGS["en"][routes.MISSED]}
    assert pt.status_code == 409 and pt.json()["detail"] == STRINGS["pt-BR"][routes.MISSED]
    assert routes.MISSED == "follow_up_missed"
    assert _rows(team) == before and team.unscheduled == []


def test_cancelling_a_missed_follow_up_with_the_tool_is_refused_and_leaves_it_unchanged(team):
    missed = _seed(team, due=-(GRACE + timedelta(minutes=5)), created=-timedelta(hours=1))
    before = _rows(team)

    reply = _tool(team, CANCEL, follow_up_id=missed)

    assert _refused(reply, "already_missed")
    assert reply["error"] == say("en", "already_missed", follow_up_id=missed)
    assert _rows(team) == before and team.unscheduled == []


def test_one_inside_its_grace_can_still_be_cancelled(team):
    due = _seed(team, due=-timedelta(seconds=60))

    reply = _client(team).post(ROUTE_CANCEL.format(due))

    assert reply.status_code == 200, reply.text
    assert reply.json()["follow_up"]["status"] == "cancelled"
    assert team.unscheduled == [(due, "cancelled")]


def test_missed_is_a_status_the_person_sees_and_its_words_are_in_both_languages():
    assert follow_ups.MISSED in follow_ups.STATUSES and follow_ups.MISSED == "missed"
    assert follow_ups.MISSED_AFTER == GRACE  # the scheduler's own grace, no new number
    assert handlers_follow_ups.ALREADY[follow_ups.MISSED] == "already_missed"
    assert routes.NOT_CANCELLED[follow_ups.MISSED] == (routes.MISSED, routes.HTTP_CONFLICT)
    for locale in ("en", "pt-BR"):
        assert routes.MISSED in STRINGS[locale]
        assert say(locale, "already_missed", follow_up_id=7) and say(locale, "no_background")
