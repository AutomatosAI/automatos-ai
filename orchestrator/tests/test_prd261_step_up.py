"""PRD-261 A-07c: step-up on sensitive routes (core/auth/step_up.py).

A route behind ``require_step_up(N)`` passes in SaaS only for a Clerk session
whose first factor was verified less than N minutes ago (the token's ``fva``
claim plus the token's age). Anything else in SaaS gets 403: a Clerk session
gets Clerk's reverification body with ``clerk_error`` at the TOP level (so the
frontend's ``useReverification`` prompts), a key or anonymous context a plain
403. The local edition passes, whatever the context: the gate is the edition,
never the role. The routes are mounted on a test app (lane rule: main.py is the
wire story's).
"""
from __future__ import annotations

import os
import time
from uuid import UUID

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (the pattern of
# test_f353_d_a_thumbnail_is_served_only_to_its_workspace.py). No database here.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import APIRouter, Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402

WORKSPACE = UUID("00000000-0000-0000-0000-0000000000a7")
MAX_AGE = 10
CLERK_BODY = {
    "clerk_error": {
        "type": "forbidden",
        "reason": "reverification-error",
        "metadata": {"reverification": {"level": "first_factor", "afterMinutes": MAX_AGE}},
    }
}


def _router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/sensitive", dependencies=[Depends(step_up.require_step_up(MAX_AGE))])
    def sensitive():
        return {"ok": True}

    return router


def _client(ctx: RequestContext, *, with_handler: bool = True) -> TestClient:
    app = FastAPI()
    if with_handler:
        step_up.install_step_up_handler(app)
    app.include_router(_router())
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app, raise_server_exceptions=False)


def _clerk(claims, system_role: str = "user") -> RequestContext:
    user = UserContext(id="user_a07c", email="a07c@example.com", system_role=system_role,
                       clerk_user_id="user_a07c", raw_claims=claims)
    return RequestContext(workspace_id=WORKSPACE, user=user, auth_type="clerk")


def _claims(fva, *, issued_minutes_ago: float = 0.0) -> dict:
    claims = {"sub": "user_a07c", "iat": int(time.time() - issued_minutes_ago * 60)}
    if fva is not None:
        claims["fva"] = fva
    return claims


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")


def _assert_reverification(resp) -> None:
    assert resp.status_code == 403
    body = resp.json()
    assert body["clerk_error"] == CLERK_BODY["clerk_error"], "clerk_error must sit at the top level"
    assert isinstance(body["detail"], str) and "10 minutes" in body["detail"]
    assert body["code"] == "step_up_required"


def test_a_fresh_first_factor_passes(saas):
    resp = _client(_clerk(_claims([2, -1]))).post("/api/sensitive")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_a_stale_first_factor_gets_the_top_level_reverification_body(saas):
    _assert_reverification(_client(_clerk(_claims([11, -1]))).post("/api/sensitive"))


def test_exactly_the_max_age_is_stale_as_clerk_decides(saas):
    # Clerk: afterMinutes > factorAge; 10 > 10 is false.
    _assert_reverification(_client(_clerk(_claims([MAX_AGE, -1]))).post("/api/sensitive"))


def test_the_token_age_counts_toward_the_factor_age(saas):
    claims = _claims([8, -1], issued_minutes_ago=3)
    _assert_reverification(_client(_clerk(claims)).post("/api/sensitive"))


@pytest.mark.parametrize("fva", [[-1, -1], [-1, 0], None, [], [3], [3, 1, 1], "3,1", [True, 0], [3.5, 0], [-4, 0]])
def test_never_verified_missing_or_malformed_fva_is_refused(saas, fva):
    _assert_reverification(_client(_clerk(_claims(fva))).post("/api/sensitive"))


def test_a_clerk_context_without_claims_is_refused(saas):
    _assert_reverification(_client(_clerk(None)).post("/api/sensitive"))


def test_a_super_admin_still_needs_step_up_in_saas(saas):
    ctx = _clerk(_claims(None), system_role="super_admin")
    _assert_reverification(_client(ctx).post("/api/sensitive"))


@pytest.mark.parametrize("auth_type", ["api_key", "sdk_key", "anonymous"])
def test_a_context_with_no_person_is_refused_in_saas(saas, auth_type):
    ctx = RequestContext(workspace_id=WORKSPACE, user=UserContext(id="api_key", system_role="admin"),
                         auth_type=auth_type)
    resp = _client(ctx).post("/api/sensitive")
    assert resp.status_code == 403
    body = resp.json()
    assert "clerk_error" not in body, "a key cannot reverify, so no Clerk prompt hint"
    assert body["detail"]["code"] == "step_up_unavailable"


def test_the_local_edition_passes_the_anonymous_owner(local):
    ctx = RequestContext(workspace_id=WORKSPACE, user=UserContext(email="op@local", system_role="super_admin"),
                         auth_type="anonymous")
    assert _client(ctx).post("/api/sensitive").status_code == 200


def test_the_local_edition_gate_is_the_edition_not_the_role(local):
    ctx = _clerk(_claims(None), system_role="user")
    assert _client(ctx).post("/api/sensitive").status_code == 200


def test_an_unknown_edition_value_enforces_step_up(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "something-else")
    _assert_reverification(_client(_clerk(_claims(None))).post("/api/sensitive"))


def test_without_the_handler_the_refusal_is_still_a_403(saas):
    resp = _client(_clerk(_claims([30, -1])), with_handler=False).post("/api/sensitive")
    assert resp.status_code == 403
    assert "minutes" in resp.json()["detail"]


def test_the_body_is_clerks_reverification_error_shape():
    assert step_up.reverification_error(MAX_AGE) == CLERK_BODY
    body = step_up.step_up_body(MAX_AGE)
    assert {k: body[k] for k in CLERK_BODY} == CLERK_BODY


def test_the_default_max_age_is_ten_minutes():
    # Gerard D1-a/e, 18 Oct 2026.
    assert step_up.DEFAULT_MAX_AGE_MINUTES == 10


def test_the_dependency_carries_its_marker():
    dep = step_up.require_step_up(MAX_AGE)
    assert getattr(dep, step_up.STEP_UP_MARKER_ATTR) == MAX_AGE
    assert getattr(step_up.require_step_up(), step_up.STEP_UP_MARKER_ATTR) == step_up.DEFAULT_MAX_AGE_MINUTES


@pytest.mark.parametrize("bad", [0, -5, True, 2.5, "10", None])
def test_a_bad_max_age_is_refused_when_the_route_is_built(bad):
    with pytest.raises(ValueError, match="max age"):
        step_up.require_step_up(bad)


def test_freshness_is_a_pure_decision():
    now = 1_000_000.0
    assert step_up.first_factor_is_fresh({"fva": [0, -1], "iat": now}, 1, now) is True
    assert step_up.first_factor_is_fresh({"fva": [0, -1], "iat": now - 60}, 1, now) is False
    # A future iat (clock skew) never makes a factor younger than its claim.
    assert step_up.first_factor_is_fresh({"fva": [9, -1], "iat": now + 600}, MAX_AGE, now) is True
    assert step_up.first_factor_is_fresh({"fva": [10, -1], "iat": now + 600}, MAX_AGE, now) is False
    # No iat: the claim alone decides.
    assert step_up.first_factor_is_fresh({"fva": [9, -1]}, MAX_AGE, now) is True
