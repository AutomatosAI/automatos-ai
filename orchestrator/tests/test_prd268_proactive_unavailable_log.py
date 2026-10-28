"""PRD-268 P268W1-RVW-4 (AGENTS.md "Errors and inputs") — a proactive route's 503 logs what failed.

``api/proactive.py`` answers a database failure on a suggestion or settings route with 503
and a refusal that carries no internals; the server log carries the failure itself. No
database: each route's service is made to raise, and the route's own gate and the session
are stand-ins. Proven here, for every route that answers 503 (the suggestion list, Start,
Hide, the runs, the workspace settings read and write, the caller's own read and write):

* the answer is 503 with the same refusal body as before (the code and its sentence, nothing
  of the failure);
* the session is rolled back;
* the ``api.proactive`` ERROR record carries ``exc_info`` (the raised exception and its
  stack), so the failure's message reaches the log.
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

WORKSPACE = uuid.uuid4()
SUGGESTION = uuid.uuid4()
USER_PK = 7
WHY = "the database went away"
PREFIX = "/api/proactive"
ROUTES = [
    NS(name="list", method="GET", template="/suggestions", service=("listing", "suggestions"),
       body=None, code="unreadable"),
    NS(name="start", method="POST", template="/suggestions/{suggestion_id}/start", service=("decisions", "start"),
       body=None, code="unavailable"),
    NS(name="hide", method="POST", template="/suggestions/{suggestion_id}/hide", service=("decisions", "hide"),
       body=None, code="unavailable"),
    NS(name="runs", method="GET", template="/runs", service=("listing", "runs"), body=None, code="unreadable"),
    NS(name="settings_read", method="GET", template="/settings", service=("settings", "read_workspace"),
       body=None, code="settings_unreadable"),
    NS(name="settings_write", method="PUT", template="/settings", service=("settings", "change_workspace"),
       body={"enabled": False}, code="settings_unavailable"),
    NS(name="me_read", method="GET", template="/me", service=("settings", "read_person"),
       body=None, code="settings_unreadable"),
    NS(name="me_write", method="PUT", template="/me", service=("settings", "change_person"),
       body={"paused": True}, code="settings_unavailable"),
]


class _Session:
    """A stand-in session: it only counts its rollbacks."""

    def __init__(self) -> None:
        self.rollbacks = 0

    def rollback(self) -> None:
        self.rollbacks += 1


def _down(*args, **kwargs):
    raise OperationalError("SELECT ... FROM proactive_suggestions", {}, Exception(WHY))


def _ctx():
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id="user_prd268_rvw4", clerk_user_id="user_prd268_rvw4", system_role="user")
    return RequestContext(workspace_id=WORKSPACE, user=user, auth_type="clerk")


def _gate(router, template: str, method: str):
    """The dependency that answers a route's ``ctx``: the gate the route was built with."""
    route = next(r for r in router.routes if r.path == template and method in r.methods)
    return next(dep.call for dep in route.dependant.dependencies if dep.name == "ctx")


def _app(proactive, route: NS, session: _Session) -> TestClient:
    from core.database.database import get_db

    ctx = _ctx()
    app = FastAPI()
    app.include_router(proactive.router)
    app.dependency_overrides[_gate(proactive.router, PREFIX + route.template, route.method)] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def _the_service_fails(proactive, route: NS, monkeypatch) -> None:
    """The route's service raises; who the caller is resolves without a database."""
    import importlib

    module = importlib.import_module(f"services.proactive.{route.service[0]}")
    monkeypatch.setattr(module, route.service[1], _down)
    monkeypatch.setattr(proactive, "_viewer", lambda db, ctx: proactive._Viewer(manages=True, user_pk=USER_PK))
    monkeypatch.setattr(proactive, "_manages", lambda db, ctx: True)
    monkeypatch.setattr(proactive, "resolve_user_pk", lambda db, ctx: USER_PK)


@pytest.mark.parametrize("route", ROUTES, ids=[route.name for route in ROUTES])
def test_a_503_answers_the_same_refusal_and_logs_the_failure_with_its_stack(route, monkeypatch, caplog):
    import api.proactive as proactive
    from services.proactive.decision_text import say

    _the_service_fails(proactive, route, monkeypatch)
    session = _Session()
    url = PREFIX + route.template.format(suggestion_id=SUGGESTION)
    with caplog.at_level(logging.ERROR, logger=proactive.__name__):
        response = _app(proactive, route, session).request(route.method, url, json=route.body)
    assert response.status_code == 503
    assert response.json() == {"code": route.code, "detail": say("en", route.code)}
    assert WHY not in response.text and "OperationalError" not in response.text
    assert session.rollbacks == 1
    records = [r for r in caplog.records if r.name == proactive.__name__ and r.levelno == logging.ERROR]
    assert len(records) == 1 and records[0].exc_info is not None
    assert isinstance(records[0].exc_info[1], OperationalError) and records[0].exc_info[2] is not None
    assert WHY in caplog.text


def test_every_route_that_answers_503_is_covered():
    """Each ``_unavailable`` call in the router is one of the routes above."""
    import inspect

    import api.proactive as proactive

    calls = inspect.getsource(proactive).count("return _unavailable(")
    assert calls == len(ROUTES)
    assert {(route.method, PREFIX + route.template) for route in ROUTES} <= {
        (method, r.path) for r in proactive.router.routes for method in r.methods}
