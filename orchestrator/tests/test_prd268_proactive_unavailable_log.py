"""PRD-268 P268W1-RVW-4 (AGENTS.md "Errors and inputs") — a proactive route's 503 logs what failed.

``api/proactive.py`` answers a database failure on a suggestion or settings route with 503
and a refusal that carries no internals; the server log carries the failure itself. No
database: each route's service is made to raise, and the route's own gate and the session
are stand-ins. Proven here, for the suggestion list, Start and the settings write:

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


def _list(proactive, monkeypatch) -> NS:
    from services.proactive import listing

    monkeypatch.setattr(proactive, "_viewer", lambda db, ctx: proactive._Viewer(manages=True, user_pk=USER_PK))
    monkeypatch.setattr(listing, "suggestions", _down)
    template = "/api/proactive/suggestions"
    return NS(method="GET", template=template, url=template, body=None, code=proactive.UNREADABLE)


def _start(proactive, monkeypatch) -> NS:
    from services.proactive import decisions

    monkeypatch.setattr(decisions, "start", _down)
    return NS(method="POST", template="/api/proactive/suggestions/{suggestion_id}/start",
              url=f"/api/proactive/suggestions/{SUGGESTION}/start", body=None, code=proactive.UNAVAILABLE)


def _settings_write(proactive, monkeypatch) -> NS:
    from services.proactive import settings

    monkeypatch.setattr(settings, "change_workspace", _down)
    template = "/api/proactive/settings"
    return NS(method="PUT", template=template, url=template, body={"enabled": False},
              code=proactive.SETTINGS_UNAVAILABLE)


@pytest.mark.parametrize("arrange", [_list, _start, _settings_write], ids=["list", "start", "settings_write"])
def test_a_503_answers_the_same_refusal_and_logs_the_failure_with_its_stack(arrange, monkeypatch, caplog):
    import api.proactive as proactive
    from core.database.database import get_db
    from services.proactive.decision_text import say

    call = arrange(proactive, monkeypatch)
    monkeypatch.setattr(proactive, "resolve_user_pk", lambda db, ctx: USER_PK)
    session, ctx = _Session(), _ctx()
    app = FastAPI()
    app.include_router(proactive.router)
    app.dependency_overrides[_gate(proactive.router, call.template, call.method)] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: session
    with caplog.at_level(logging.ERROR, logger=proactive.__name__):
        response = TestClient(app).request(call.method, call.url, json=call.body)
    assert response.status_code == 503
    assert response.json() == {"code": call.code, "detail": say("en", call.code)}
    assert WHY not in response.text and "OperationalError" not in response.text
    assert session.rollbacks == 1
    records = [r for r in caplog.records if r.name == proactive.__name__ and r.levelno == logging.ERROR]
    assert len(records) == 1 and records[0].exc_info is not None
    assert isinstance(records[0].exc_info[1], OperationalError) and records[0].exc_info[2] is not None
    assert WHY in caplog.text
