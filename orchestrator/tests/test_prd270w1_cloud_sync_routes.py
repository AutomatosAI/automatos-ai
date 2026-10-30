"""PRD-270 US-270-03b (§5.1, §5.5, §8-Q1, K17) — the Sync with Automatos Cloud routes.

``POST /api/marketplace/cloud-sync`` and ``GET /api/marketplace/cloud-sync/status``
(``api/marketplace_cloud_sync.py``) run on a mini FastAPI app. The gates need no database: in ``gates``
every override records that it ran, so a refusal shows what was looked at before it. The Sync itself runs
for real on Postgres through ``tests/test_prd270w1_cloud_pull.py``'s ``cloud`` fixture (committing
sessions, a fake S3 that must stay empty, a test keys file, Automatos Cloud as an ``httpx.MockTransport``,
the status row swept and put back). Proven:

* an owner, an admin and the local operator get 200 from both routes; an editor and a viewer get 403,
  and nothing runs behind the refusal;
* on SaaS every caller gets 404 from both routes, before the caller or the database is looked at;
* with ``AUTOMATOS_CLOUD_URL`` empty the status answers enabled false and the Sync is 409 with a plain
  sentence, no request sent and nothing recorded;
* a failing Sync is a 502 with the plain error it recorded, which the status then shows as lastError, the
  previous success kept;
* a good Sync answers its counts and the status; a second one is unchanged;
* a Sync while another runs is 409 (already running) and records nothing;
* both handlers are plain ``def``s, the edition check is their first dependency and the
  ``workspace:manage`` gate comes after it.
"""
from __future__ import annotations

import inspect
import time
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import api.marketplace_cloud_sync as sync_api
import core.auth.workspace_permission as permission_mod
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE
from core.database.database import get_db
from services.marketplace_cloud import fetch, pull
from tests.test_prd270w1_cloud_pull import CLOUD, GENERATED, _catalog, _item, _pkg
from tests.test_prd270w1_cloud_pull import cloud, engine  # noqa: F401  (the Sync tests' fixtures)

WS = uuid.uuid4()
SYNC = "/api/marketplace/cloud-sync"
STATUS = SYNC + "/status"
ROUTES = (("GET", STATUS), ("POST", SYNC))
MANAGERS = ("owner", "admin", "operator")
NOT_MANAGERS = ("editor", "viewer")
OPERATOR = "operator"  # the local edition's anonymous caller: the workspace owner (PRD-175)
NO_DATABASE = object()  # what ``gates`` hands a route for a session: any use of it fails
EMPTY = {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}


def _caller(role):
    """A signed-in member of the workspace, or the local operator for ``OPERATOR``."""
    if role == OPERATOR:
        user = UserContext(id="operator@local", email="operator@local", system_role="user",
                           raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": 1})
        return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")
    user = UserContext(id="clerk-9", clerk_user_id="clerk-9", system_role="user",
                       raw_claims={"sub": "clerk-9", "iat": int(time.time())})
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _app(state, database, factory=None):
    app = FastAPI()
    app.include_router(sync_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.seen("caller") or _caller(state.role)
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[sync_api.get_cloud_client_factory] = lambda: factory
    return TestClient(app)


def _call(state, method, path):
    return state.client.request(method, path)


def _never(*args, **kwargs):
    raise AssertionError("nothing may run behind a refused gate")


@pytest.fixture
def gates(monkeypatch):
    """The router with no database, in the local edition with a Cloud URL; ``calls`` lists what ran."""
    state = SimpleNamespace(role="owner", calls=[])
    state.seen = lambda name: state.calls.append(name)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.seen("role") or state.role)
    state.client = _app(state, lambda: state.seen("database") or NO_DATABASE)
    return state


@pytest.fixture
def env(cloud, monkeypatch):  # noqa: F811  (cloud: the Sync tests' fixture)
    """The router over Postgres and the mock Automatos Cloud, an owner calling."""
    state = SimpleNamespace(cloud=cloud, role="owner", calls=[])
    state.seen = lambda name: state.calls.append(name)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)

    def database():
        session = cloud.new()
        try:
            yield session
        finally:
            session.close()

    state.client = _app(state, database, cloud.mock.factory)
    return state


# --------------------------------------------------------------------------- the gates


@pytest.mark.parametrize("role", MANAGERS + NOT_MANAGERS)
@pytest.mark.parametrize("method, path", ROUTES)
def test_on_saas_every_caller_gets_404_before_the_caller_or_the_database_is_looked_at(gates, monkeypatch, role,
                                                                                         method, path):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(pull, "sync_catalog", _never)
    monkeypatch.setattr(pull, "sync_status", _never)
    gates.role = role
    answer = _call(gates, method, path)
    assert answer.status_code == 404 and answer.json()["detail"] == sync_api.HOSTED_NO_SYNC
    assert gates.calls == []


@pytest.mark.parametrize("role", NOT_MANAGERS)
@pytest.mark.parametrize("method, path", ROUTES)
def test_a_member_who_is_not_an_owner_or_admin_gets_403_and_nothing_runs(gates, monkeypatch, role, method, path):
    monkeypatch.setattr(pull, "sync_catalog", _never)
    monkeypatch.setattr(pull, "sync_status", _never)
    gates.role = role
    answer = _call(gates, method, path)
    assert answer.status_code == 403 and answer.json()["detail"] == "Permission denied: workspace:manage"
    assert gates.calls[0] == "caller" and "role" in gates.calls


def test_with_the_url_empty_the_sync_is_409_and_never_runs(gates, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "  ")
    monkeypatch.setattr(pull, "sync_catalog", _never)
    answer = _call(gates, "POST", SYNC)
    assert answer.status_code == 409 and answer.json()["detail"] == fetch.CLOUD_OFF


def test_a_sync_while_another_runs_is_409_already_running_and_records_nothing(gates):
    assert pull._SYNC_LOCK.acquire(blocking=False)
    try:
        answer = _call(gates, "POST", SYNC)
    finally:
        pull._SYNC_LOCK.release()
    assert answer.status_code == 409 and answer.json()["detail"] == pull.ALREADY_RUNNING
    # the session was handed over, never used: NO_DATABASE fails on any use


def _routes():
    return {(method, route.path): route for route in sync_api.router.routes if isinstance(route, APIRoute)
            for method in route.methods - {"HEAD", "OPTIONS"}}


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def test_both_routes_are_plain_defs_edition_first_then_workspace_manage():
    routes = _routes()
    assert set(routes) == set(ROUTES)
    for key, route in routes.items():
        assert not inspect.iscoroutinefunction(route.endpoint), f"{key} must be a plain def"
        calls = _calls(route.dependant)
        assert calls[0] is sync_api.require_local_edition, key
        marked = [n for n, call in enumerate(calls) if getattr(call, permission_mod.PERMISSION_MARKER_ATTR, None)]
        assert [getattr(calls[n], permission_mod.PERMISSION_MARKER_ATTR) for n in marked] == ["workspace:manage"], key
        assert calls.index(get_request_context_hybrid) > 0, f"{key}: the caller is read after the edition check"


# --------------------------------------------------------------------------- the Sync (Postgres)


@pytest.mark.parametrize("role", MANAGERS)
def test_an_owner_an_admin_and_the_operator_get_200_from_both_routes(env, role):
    env.role = role
    env.cloud.mock.body = _catalog()
    status = env.client.get(STATUS)
    assert status.status_code == 200 and status.json()["enabled"] is True
    synced = env.client.post(SYNC)
    assert synced.status_code == 200 and synced.json()["counts"] == EMPTY


def test_with_the_url_empty_the_status_answers_enabled_false_and_the_sync_records_nothing(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "")
    status = env.client.get(STATUS)
    assert status.status_code == 200
    assert status.json()["enabled"] is False and status.json()["cloudUrl"] is None
    refused = env.client.post(SYNC)
    assert refused.status_code == 409 and refused.json()["detail"] == fetch.CLOUD_OFF
    assert env.cloud.mock.requests == [] and env.client.get(STATUS).json()["lastAttemptAt"] is None


def test_a_good_sync_answers_its_counts_and_the_status(env):
    arcade, ink = _pkg(env.cloud.theme("arcade")), _pkg(env.cloud.theme("ink"))
    env.cloud.mock.body = _catalog(_item(arcade), _item(ink))
    answer = env.client.post(SYNC)
    assert answer.status_code == 200
    body = answer.json()
    assert set(body) == {"counts", "status"}
    assert body["counts"] == {**EMPTY, "new": 2}
    status = body["status"]
    assert status["enabled"] is True and status["cloudUrl"] == CLOUD and status["lastError"] is None
    assert status["counts"] == body["counts"] and status["generatedAt"] == GENERATED
    assert status["lastSuccessAt"] == status["lastAttemptAt"] and status["stale"] is False
    assert env.client.get(STATUS).json() == status
    again = env.client.post(SYNC)
    assert again.status_code == 200 and again.json()["counts"] == {**EMPTY, "unchanged": 2}
    assert env.cloud.s3.objects == {}  # a Sync stores no package byte


def test_a_failing_sync_is_502_with_the_error_it_recorded_and_the_status_shows_it(env):
    env.cloud.mock.body = _catalog()
    good = env.client.post(SYNC).json()["status"]
    env.cloud.mock.status = 404
    answer = env.client.post(SYNC)
    assert answer.status_code == 502
    error = answer.json()["detail"]
    assert error == fetch.BAD_STATUS.format(status=404)
    status = env.client.get(STATUS).json()
    assert status["lastError"] == error
    assert status["lastSuccessAt"] == good["lastSuccessAt"] and status["counts"] == good["counts"]
    assert datetime.fromisoformat(status["lastAttemptAt"]) > datetime.fromisoformat(good["lastAttemptAt"])
