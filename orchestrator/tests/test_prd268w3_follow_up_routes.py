"""PRD-268 US-C11e (§6.3 'In the thread', K8, K22) — the person's follow-up routes, against Postgres.

``GET /api/proactive/follow-ups`` and ``POST /api/proactive/follow-ups/{follow_up_id}/cancel``
(``api/proactive_follow_ups.py``) over ``services/proactive/follow_ups.py``. Proven here:

* the list is the caller's own follow-ups in this workspace, pending by default and soonest
  first, ``all`` adding those that fired or were cancelled; ``chat_id`` keeps one chat's; each
  is ``{id, at, reason, status, chat_id}``;
* another person's follow-up, another workspace's (even the caller's own there) and an
  ordinary scheduled task are never listed, and cancelling any of them is 404, never 403,
  with nothing changed;
* a cancel cancels the caller's own pending one (the job removed, one audit row with the
  person as its actor and no reason in it); a second cancel is 409 ``follow_up_cancelled``,
  and one that already fired 409 ``follow_up_fired``;
* a caller with no person behind it is 403, a call with no workspace 404, and a database
  that fails 503 with nothing of the failure in the body; a refusal speaks pt-BR when asked;
* a viewer reads their own list but cannot cancel (``agents:execute``, K22);
* both routes are plain ``def``s, gated by ``require_workspace_permission``, registered
  through ``RouterSpec`` and in the route manifest.

``agent_scheduled_tasks`` has no model, so CI's create_all database has none: the module
makes it, in the shape the migrations give it, when it is missing, and drops it after. The
rows commit (each request runs on its own session), and each test sweeps its own.
"""
from __future__ import annotations

import inspect
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
from sqlalchemy.exc import OperationalError  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.proactive_follow_ups as routes  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.proactive import follow_ups  # noqa: E402
from services.proactive.decision_text import STRINGS  # noqa: E402

LIST = "/api/proactive/follow-ups"
CANCEL = "/api/proactive/follow-ups/{}/cancel"
MANIFEST = ORCH / "reports" / "route-manifest.json"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
REASON = "Check whether the Lisbon card moved; Ana said the landlord may pull out"
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
            c.execute(text("SELECT actor_type FROM audit_logs LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the follow-up route tests need Postgres with the PRD-205 tables: {exc}")
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


def _person(s, tag):
    clerk = f"user_prd268w3fr_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(clerk=clerk, pk=int(pk))


def _workspace(s, owner):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) "
                   "VALUES (CAST(:id AS uuid), 'prd268w3 routes', :o, CAST('{}' AS jsonb))"), {"id": ws, "o": owner})
    return ws


def _member(s, ws, user, role):
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, :r, true)"), {"w": ws, "u": user, "r": role})


def _auto(s, ws):
    return int(s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent, configuration, slug) "
        "VALUES ('Auto', 'custom', CAST(:w AS uuid), 'active', true, CAST('{}' AS json), :slug) RETURNING id"),
        {"w": ws, "slug": f"auto-{ws}"}).scalar())


def _chat(s, ws, user):
    chat = str(uuid.uuid4())
    s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind) "
                   "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), 'Lisbon', 'private', 'user')"),
              {"id": chat, "u": user, "w": ws})
    return chat


def _row(s, ws, *, user, chat, agent, minutes, status="active", deliver_as="follow_up", reason=REASON):
    """One scheduled row: a follow-up of ``user``'s in ``chat`` by default, due in ``minutes``."""
    return int(s.execute(text("""
        INSERT INTO agent_scheduled_tasks
            (workspace_id, created_by_agent_id, target_agent_id, task_type, description, schedule,
             next_run_at, origin_chat_id, deliver_as, for_user_id, status)
        VALUES (CAST(:w AS uuid), :a, :a, 'one_shot', :r, :sched, :at, CAST(:c AS uuid), :d, :u, :st)
        RETURNING id"""), {
        "w": ws, "a": agent, "r": reason, "sched": (NOW + timedelta(minutes=minutes)).isoformat(),
        "at": NOW + timedelta(minutes=minutes), "c": chat, "d": deliver_as, "u": user, "st": status}).scalar())


def _sweep(new_session, workspaces, clerks):
    s = new_session.sweep()
    for table in ("audit_logs", "agent_scheduled_tasks", "chats", "workspace_members", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": clerks})
    s.commit()


@pytest.fixture
def team(engine, new_session, monkeypatch):
    """Ana owns the workspace and has follow-ups in two chats: two pending in Lisbon, one that
    fired there and one she cancelled; one pending in Porto and one there whose run failed. Ben (an editor) and Vic (a viewer) each have
    one; Kim owns another workspace, where she has one of her own, and is an editor here too.
    An ordinary scheduled task sits beside them. The scheduler's job removal is recorded."""
    from services.scheduled_task_service import ScheduledTaskService

    s = new_session()
    ana, ben, vic, kim = _person(s, "ana"), _person(s, "ben"), _person(s, "vic"), _person(s, "kim")
    ws, other_ws = _workspace(s, ana.pk), _workspace(s, kim.pk)
    _member(s, ws, ben.pk, "editor")
    _member(s, ws, vic.pk, "viewer")
    _member(s, ws, kim.pk, "editor")
    auto, other_auto = _auto(s, ws), _auto(s, other_ws)
    lisbon, porto = _chat(s, ws, ana.pk), _chat(s, ws, ana.pk)
    ben_chat, vic_chat, kim_chat = _chat(s, ws, ben.pk), _chat(s, ws, vic.pk), _chat(s, other_ws, kim.pk)
    t = NS(ws=ws, other_ws=other_ws, ana=ana, ben=ben, vic=vic, kim=kim, lisbon=lisbon, porto=porto,
           new_session=new_session, unscheduled=[])
    t.later = _row(s, ws, user=ana.pk, chat=lisbon, agent=auto, minutes=90)
    t.soon = _row(s, ws, user=ana.pk, chat=lisbon, agent=auto, minutes=20)
    t.in_porto = _row(s, ws, user=ana.pk, chat=porto, agent=auto, minutes=45)
    t.fired = _row(s, ws, user=ana.pk, chat=lisbon, agent=auto, minutes=-60, status="completed")
    t.cancelled = _row(s, ws, user=ana.pk, chat=lisbon, agent=auto, minutes=30, status="cancelled")
    t.failed = _row(s, ws, user=ana.pk, chat=porto, agent=auto, minutes=-120, status="failed")
    t.bens = _row(s, ws, user=ben.pk, chat=ben_chat, agent=auto, minutes=25, reason="Ben's own reason")
    t.vics = _row(s, ws, user=vic.pk, chat=vic_chat, agent=auto, minutes=35, reason="Vic's own reason")
    t.kims = _row(s, other_ws, user=kim.pk, chat=kim_chat, agent=other_auto, minutes=15, reason="Kim's own reason")
    t.ordinary = _row(s, ws, user=None, chat=lisbon, agent=auto, minutes=40, deliver_as="chat")
    s.commit()
    monkeypatch.setattr(ScheduledTaskService, "_reschedule", lambda self, *args: t.unscheduled.append(args))
    yield t
    _sweep(new_session, [ws, other_ws], [ana.clerk, ben.clerk, vic.clerk, kim.clerk])


def _ctx(person, ws, role="user") -> RequestContext:
    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role=role)
    return RequestContext(workspace_id=UUID(ws) if ws else None, user=user, auth_type="clerk")


def _client(new_session, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: new_session()
    return TestClient(app)


def _list(team, person, ws=None, **params):
    return _client(team.new_session, _ctx(person, ws or team.ws)).get(LIST, params=params)


def _cancel(team, person, follow_up_id, ws=None, headers=None):
    return _client(team.new_session, _ctx(person, ws or team.ws)).post(CANCEL.format(follow_up_id),
                                                                       headers=headers or {})


def _ids(response):
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["follow_ups"]]


def _status(team, row_id):
    with team.new_session() as s:
        return s.execute(text("SELECT status FROM agent_scheduled_tasks WHERE id = :id"), {"id": row_id}).scalar()


def _audits(team, ws=None):
    with team.new_session() as s:
        return s.execute(text(
            "SELECT action, user_id, actor_type, resource_type, resource_id, details FROM audit_logs "
            "WHERE workspace_id = CAST(:w AS uuid) AND resource_type = 'follow_up' ORDER BY created_at"),
            {"w": ws or team.ws}).fetchall()


def _refusal(response, status_code, code, locale="en"):
    assert response.status_code == status_code, response.text
    assert response.json() == {"code": code, "detail": STRINGS[locale][code]}


# --------------------------------------------------------------------------- the list


def test_the_list_is_the_callers_own_pending_follow_ups_soonest_first(team):
    listed = _list(team, team.ana)

    assert _ids(listed) == [team.soon, team.in_porto, team.later]
    first = listed.json()["follow_ups"][0]
    assert first == {"id": team.soon, "at": (NOW + timedelta(minutes=20)).isoformat(), "reason": REASON,
                     "status": "pending", "chat_id": team.lisbon}


def test_the_list_keeps_one_chat_and_all_adds_the_fired_and_the_cancelled(team):
    in_lisbon = _list(team, team.ana, chat_id=team.lisbon)
    everything_in_lisbon = _list(team, team.ana, chat_id=team.lisbon, status="all")

    assert _ids(in_lisbon) == [team.soon, team.later]
    assert _ids(everything_in_lisbon) == [team.soon, team.later, team.cancelled, team.fired]
    statuses = {item["id"]: item["status"] for item in everything_in_lisbon.json()["follow_ups"]}
    assert statuses == {team.soon: "pending", team.later: "pending", team.cancelled: "cancelled",
                        team.fired: "fired"}
    in_porto = _list(team, team.ana, chat_id=team.porto, status="all").json()["follow_ups"]
    assert [(item["id"], item["status"]) for item in in_porto] == [(team.in_porto, "pending"), (team.failed, "fired")]


def test_another_persons_another_workspaces_and_an_ordinary_task_are_never_listed(team):
    anas = set(_ids(_list(team, team.ana, status="all")))
    bens = _ids(_list(team, team.ben, status="all"))
    kims_here = _ids(_list(team, team.kim, status="all"))
    kims_there = _ids(_list(team, team.kim, team.other_ws, status="all"))

    assert anas == {team.soon, team.in_porto, team.later, team.fired, team.cancelled, team.failed}
    assert not anas & {team.bens, team.vics, team.kims, team.ordinary}
    assert bens == [team.bens]
    assert kims_here == []  # her own follow-up is in her own workspace, not this one
    assert kims_there == [team.kims]


def test_a_status_the_list_does_not_know_and_a_chat_that_is_not_an_id_are_422(team):
    assert _list(team, team.ana, status="fired").status_code == 422
    assert _list(team, team.ana, chat_id="lisbon").status_code == 422


# --------------------------------------------------------------------------- the cancel


def test_a_cancel_cancels_the_callers_own_and_a_second_is_409(team):
    first = _cancel(team, team.ana, team.soon)
    second = _cancel(team, team.ana, team.soon)

    assert first.status_code == 200, first.text
    assert first.json() == {"follow_up": {"id": team.soon, "at": (NOW + timedelta(minutes=20)).isoformat(),
                                          "reason": REASON, "status": "cancelled", "chat_id": team.lisbon}}
    assert _status(team, team.soon) == "cancelled"
    assert team.unscheduled == [(team.soon, "cancelled")]
    _refusal(second, 409, "follow_up_cancelled")
    assert team.unscheduled == [(team.soon, "cancelled")]
    assert _ids(_list(team, team.ana)) == [team.in_porto, team.later]


def test_a_follow_up_that_already_fired_is_409(team):
    _refusal(_cancel(team, team.ana, team.fired), 409, "follow_up_fired")
    _refusal(_cancel(team, team.ana, team.failed), 409, "follow_up_fired")  # it came; its run failed
    _refusal(_cancel(team, team.ana, team.cancelled), 409, "follow_up_cancelled")

    assert [_status(team, row) for row in (team.fired, team.failed, team.cancelled)] == [
        "completed", "failed", "cancelled"]
    assert team.unscheduled == []


def test_another_persons_follow_up_and_an_ordinary_task_are_404_and_unchanged(team):
    _refusal(_cancel(team, team.ben, team.soon), 404, "follow_up_not_found")
    _refusal(_cancel(team, team.ana, team.bens), 404, "follow_up_not_found")
    _refusal(_cancel(team, team.ana, team.ordinary), 404, "follow_up_not_found")
    # Another person's that already fired is not found either: its state is not theirs to learn.
    _refusal(_cancel(team, team.ben, team.fired), 404, "follow_up_not_found")

    assert [_status(team, row) for row in (team.soon, team.bens, team.ordinary)] == ["active"] * 3
    assert team.unscheduled == []
    assert _audits(team) == []


def test_another_workspaces_follow_up_is_404_even_to_its_own_person(team):
    _refusal(_cancel(team, team.kim, team.kims), 404, "follow_up_not_found")  # Kim, calling in Ana's workspace
    _refusal(_cancel(team, team.ana, team.kims), 404, "follow_up_not_found")
    _refusal(_cancel(team, team.ana, 2_147_483_000), 404, "follow_up_not_found")

    assert _status(team, team.kims) == "active"
    assert _audits(team, team.other_ws) == []


def test_the_cancel_writes_one_audit_row_with_the_person_as_actor_and_no_reason(team):
    assert _cancel(team, team.ana, team.later).status_code == 200

    (audit,) = _audits(team)
    assert (audit.action, audit.user_id, audit.actor_type, audit.resource_type, audit.resource_id) == (
        follow_ups.CANCELLED_ACTION, team.ana.pk, "user", "follow_up", str(team.later))
    assert set(audit.details) == {"at"}
    assert datetime.fromisoformat(audit.details["at"]) == NOW + timedelta(minutes=90)
    assert REASON not in json.dumps(audit.details)


def test_a_follow_up_id_that_is_not_a_positive_number_is_422(team):
    for bad in ("0", "-3", "lisbon", str(routes.MAX_FOLLOW_UP_ID)):
        assert _cancel(team, team.ana, bad).status_code == 422, bad
    assert team.unscheduled == []


# --------------------------------------------------------------------------- who may


def test_a_viewer_reads_their_own_list_but_cannot_cancel(team):
    assert _ids(_list(team, team.vic)) == [team.vics]

    refused = _cancel(team, team.vic, team.vics)

    assert refused.status_code == 403, refused.text
    assert _status(team, team.vics) == "active"


def test_a_caller_with_no_person_behind_it_is_403(team):
    # Past the gate (a super admin bypasses it) with no users row behind the caller.
    nobody = _client(team.new_session, _ctx(NS(clerk=f"user_prd268w3fr_nobody_{uuid.uuid4().hex[:10]}"), team.ws,
                                            "super_admin"))
    # An API key has no person either, and the gate refuses it first.
    key = _client(team.new_session, RequestContext(
        workspace_id=UUID(team.ws), user=UserContext(id="api_key", system_role="admin"), auth_type="api_key"))

    _refusal(nobody.get(LIST), 403, "follow_up_no_person")
    _refusal(nobody.post(CANCEL.format(team.soon)), 403, "follow_up_no_person")
    assert key.get(LIST).status_code == 403
    assert key.post(CANCEL.format(team.soon)).status_code == 403
    assert _status(team, team.soon) == "active"
    assert team.unscheduled == []


def test_a_caller_with_no_workspace_is_404(team):
    nowhere = _ctx(team.ana, None, "super_admin")

    _refusal(_client(team.new_session, nowhere).get(LIST), 404, "no_workspace")
    _refusal(_client(team.new_session, nowhere).post(CANCEL.format(team.soon)), 404, "no_workspace")
    assert _status(team, team.soon) == "active"


def test_a_refusal_speaks_the_callers_language(team):
    refused = _cancel(team, team.ana, team.fired, headers={"Accept-Language": "pt-BR,pt;q=0.9"})

    _refusal(refused, 409, "follow_up_fired", "pt-BR")
    codes = {routes.NO_WORKSPACE, routes.NO_PERSON, routes.NOT_FOUND, routes.FIRED, routes.CANCELLED,
             routes.UNREADABLE, routes.UNAVAILABLE}
    assert all(code in STRINGS["en"] and code in STRINGS["pt-BR"] for code in codes)


def test_a_database_that_fails_is_503_and_the_body_says_nothing_of_it(team, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise OperationalError("SELECT secret_column", {}, Exception("connection to 10.0.0.7 lost"))

    monkeypatch.setattr(follow_ups, "list_follow_ups", broken)
    monkeypatch.setattr(follow_ups, "cancel_follow_up", broken)

    with caplog.at_level(logging.ERROR, logger=routes.__name__):
        listed, cancelled = _list(team, team.ana), _cancel(team, team.ana, team.soon)

    _refusal(listed, 503, "follow_ups_unreadable")
    _refusal(cancelled, 503, "follow_up_unavailable")
    said = listed.text + cancelled.text
    assert "10.0.0.7" not in said and "secret_column" not in said and "OperationalError" not in said
    records = [r for r in caplog.records if r.name == routes.__name__ and r.levelno == logging.ERROR]
    assert len(records) == 2 and all(isinstance(r.exc_info[1], OperationalError) for r in records)
    assert _status(team, team.soon) == "active"


# --------------------------------------------------------------------------- the routes


def test_the_routes_are_plain_defs_gated_registered_and_in_the_manifest():
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(routes.list_my_follow_ups)
    assert not inspect.iscoroutinefunction(routes.cancel_my_follow_up)
    assert RouterSpec("api.proactive_follow_ups") in MANIFEST_ROUTERS
    gates = {}
    for route in routes.router.routes:
        (method,) = route.methods
        gates[(method, route.path)] = [getattr(d.call, PERMISSION_MARKER_ATTR, None)
                                       for d in route.dependant.dependencies]
    assert gates[("GET", LIST)].count(routes.READ_PERMISSION) == 1
    assert gates[("POST", CANCEL.format("{follow_up_id}"))].count("agents:execute") == 1  # the sweep's class (a)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": LIST} in manifest["routes"]
    assert {"method": "POST", "path": CANCEL.format("{follow_up_id}")} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
