"""PRD-261 A-07f: step-up on the routes that grant a person or a key rights (Gerard D1-d, #1157).

The workspace role change (``members:change_role``), the invite (``members:invite``) and SDK key
creation (``workspace:manage``) each need, in SaaS, a sign-in from the last
``STEP_UP_MAX_AGE_MINUTES`` minutes: theme code runs in the page as the signed-in user, and this
stops it promoting, inviting or minting a key silently. The real routers run on a mini app with
the step-up handler installed, the REAL permission gate (only the caller's workspace role is
stubbed) and the REAL step-up dependency. Pins:

* SaaS, no fresh first factor (missing, never verified, too old): 403 with Clerk's
  reverification body at the top level, and no handler runs (nothing written, nobody invited,
  no key minted);
* a caller the permission gate refuses gets the plain 403, never a reverification prompt;
* a key context in SaaS cannot reverify: ``step_up_unavailable``, and no key is minted;
* SaaS with a fresh first factor: each route runs as before;
* local edition: each route runs with no step-up (one anonymous owner, nobody to re-sign-in).

The route table facts (which routes carry the dependency, in which order) are
``test_prd261_b2_wire.py``. The chat tools ``platform_set_member_role`` and
``platform_invite_member`` are NOT covered: their handlers receive no verified token (see the
A-07f notes).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.api_keys as api_keys_api  # noqa: E402
import api.team as team_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS = UUID("00000000-0000-0000-0000-00000000a07f")
NOW = datetime(2026, 10, 19, 12, 0, 0)

ROLE_CHANGE = ("patch", f"/api/workspaces/{WS}/team/members/5/role", {"json": {"role": "admin"}})
INVITE = ("post", f"/api/workspaces/{WS}/team/invite", {"json": {"email": "new@example.com", "role": "admin"}})
KEY_CREATE = ("post", "/api/api-keys", {"json": {"name": "ci", "key_type": "server", "permissions": ["chat"]}})
CALLS = {"role change": ROLE_CHANGE, "invite": INVITE, "key creation": KEY_CREATE}
# What each handler does once it runs: the status it answers with.
RAN_STATUS = {"role change": 200, "invite": 200, "key creation": 201}


# ---------------------------------------------------------------------------
# Callers, the session and the handlers' side effects
# ---------------------------------------------------------------------------


def _clerk(fva=None, auth_type="clerk"):
    claims = {"sub": "clerk-a07f", "iat": int(time.time())}
    if fva is not None:
        claims["fva"] = fva
    user = UserContext(id="clerk-a07f", clerk_user_id="clerk-a07f", email="owner@example.com",
                       system_role="user", raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type=auth_type)


def _local_operator():
    return RequestContext(workspace_id=WS, auth_type="anonymous",
                          user=UserContext(id="operator@local", email="operator@local", system_role="user"))


class _AccountDb:
    """Just enough session for ``update_member_role``: one editor row, commits counted."""

    def __init__(self):
        self.member = SimpleNamespace(id=5, role="editor")
        self.commits = 0

    def query(self, model):
        return self

    def filter(self, *conditions):
        return self

    def first(self):
        return self.member

    def commit(self):
        self.commits += 1


def _invitation(**fields):
    return SimpleNamespace(id=41, email=fields["email"], role=fields["role"],
                           expires_at=NOW + timedelta(days=7), created_at=NOW)


def _minted_key(**fields):
    return {"id": "key-1", "name": fields["name"], "key": "minted-in-the-test", "key_type": fields["key_type"],
            "permissions": fields["permissions"]}


@pytest.fixture
def account(monkeypatch):
    """The real routers with the role stubbed to ``owner`` and every side effect recorded."""
    effects = SimpleNamespace(invited=[], minted=[], audits=[], db=_AccountDb(), role="owner")

    async def invite(**fields):
        effects.invited.append(fields)
        return _invitation(**fields)

    def mint(**fields):
        effects.minted.append(fields)
        return _minted_key(**fields)

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: effects.role)
    monkeypatch.setattr(team_api, "_resolve_internal_user_id", lambda db, ctx: 1)
    monkeypatch.setattr(team_api, "invite_member_to_workspace", invite)
    monkeypatch.setattr(team_api, "AuditService", lambda db: SimpleNamespace(log=lambda **e: effects.audits.append(e)))
    monkeypatch.setattr(api_keys_api, "ApiKeyService", SimpleNamespace(create_api_key=mint))
    return effects


def _call(effects, ctx, name):
    app = FastAPI()
    step_up.install_step_up_handler(app)
    app.include_router(team_api.router)
    app.include_router(api_keys_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: effects.db
    method, path, kwargs = CALLS[name]
    return getattr(TestClient(app, raise_server_exceptions=False), method)(path, **kwargs)


def _nothing_ran(effects):
    return (effects.db.commits == 0 and effects.db.member.role == "editor"
            and effects.invited == [] and effects.minted == [] and effects.audits == [])


def _ran(effects, name):
    if name == "role change":
        return effects.db.member.role == "admin" and effects.audits[0]["action"] == "member:role_changed"
    if name == "invite":
        return [(i["email"], i["role"]) for i in effects.invited] == [("new@example.com", "admin")]
    return [m["name"] for m in effects.minted] == ["ci"]


# ---------------------------------------------------------------------------
# SaaS refusals
# ---------------------------------------------------------------------------


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.mark.parametrize("fva", [None, [-1, -1], [config.STEP_UP_MAX_AGE_MINUTES + 5, -1]])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_without_a_fresh_step_up_saas_refuses_with_the_reverification_body(saas, account, fva, name):
    resp = _call(account, _clerk(fva), name)
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["clerk_error"]["reason"] == "reverification-error"  # top level, never under detail
    level = body["clerk_error"]["metadata"]["reverification"]
    assert level == {"level": "first_factor", "afterMinutes": config.STEP_UP_MAX_AGE_MINUTES}
    assert body["code"] == step_up.STEP_UP_CODE
    assert _nothing_ran(account)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_viewer_is_refused_by_the_gate_before_any_reverification_prompt(saas, account, name):
    account.role = "viewer"
    resp = _call(account, _clerk(None), name)
    assert resp.status_code == 403 and "clerk_error" not in resp.json()
    assert _nothing_ran(account)


def test_a_key_cannot_mint_a_key_in_saas_because_it_cannot_reverify(saas, account):
    resp = _call(account, _clerk(None, auth_type="api_key"), "key creation")
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["code"] == step_up.STEP_UP_UNAVAILABLE_CODE
    assert _nothing_ran(account)


# ---------------------------------------------------------------------------
# Allowed: a fresh step-up in SaaS, and the local edition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CALLS))
def test_with_a_fresh_step_up_saas_runs_the_route(saas, account, name):
    resp = _call(account, _clerk([1, -1]), name)
    assert resp.status_code == RAN_STATUS[name], resp.text
    assert _ran(account, name)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_the_local_edition_runs_the_route_with_no_step_up(monkeypatch, account, name):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    resp = _call(account, _local_operator(), name)
    assert resp.status_code == RAN_STATUS[name], resp.text
    assert _ran(account, name)
