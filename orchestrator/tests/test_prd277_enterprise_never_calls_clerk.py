"""PRD-277 US-103: an enterprise install never calls Clerk, even with Clerk's secret left set.

Clerk sends every invitation's email, and accepting one verifies a Clerk token, so the
enterprise edition has no invitations (PRD-277 §4). These tests patch the edition past the
start-up refusal (``test_prd277_enterprise_start_up.py``) and pin:

* every method of the Clerk client that calls Clerk's API refuses where the install does not
  sign people in with Clerk: no request leaves (the HTTP client is never built), and each
  fails as it does with no secret set (``ClerkNotUsed``, a ``ValueError``; or ``None``, ``[]``,
  ``False``);
* the control: on saas the same calls reach the HTTP client;
* the invitation routes (invite, list, revoke, accept, info) answer 404 before the caller is
  looked at; local and saas still reach them;
* removing a member still works in enterprise and never asks Clerk; saas still does;
* Auto's invite tool refuses in enterprise without storing anything.

No Clerk webhook receiver exists in this codebase, so there is none to refuse.

DB-free: the session and the permission seam are stubbed.
"""
from __future__ import annotations

import asyncio
import os
import sys
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

import api.team as team_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.workspaces.invitations as invitations_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import clerk as clerk_mod  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import User  # noqa: E402
from core.workspaces.models import WorkspaceMember  # noqa: E402
from modules.tools.discovery import handlers_members  # noqa: E402

WS = UUID("00000000-0000-0000-0000-0000000277cc")
# A stand-in for Clerk's secret, built from parts so no literal looks like a credential.
STAND_IN_CLERK_SECRET = "prd277" + "-" + "stand-in"
ORG = "org" + "_" + "prd277"
CLERK_USER = "user" + "_" + "prd277"


USER_LIST_PATH = "/v1/users"


class _Response:
    """Success, with an empty body of the shape the call reads (the user list is an array)."""

    status_code = 200
    text = ""

    def __init__(self, url):
        self.url = url

    def json(self):
        if self.url.endswith(USER_LIST_PATH):
            return []
        return {"id": "inv" + "_" + "prd277", "data": []}


class _SyncClient:
    """httpx's sync client as the Clerk client uses it: a context manager that records requests."""

    def __init__(self, sent):
        self.sent = sent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _send(self, method, url):
        self.sent.append((method, url))
        return _Response(url)

    def get(self, url, **kwargs):
        return self._send("GET", url)


class _AsyncClient(_SyncClient):
    """httpx's async client as the Clerk client uses it."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        return self._send("GET", url)

    async def post(self, url, **kwargs):
        return self._send("POST", url)

    async def delete(self, url, **kwargs):
        return self._send("DELETE", url)


class _Http:
    """Stands in for ``httpx`` inside the Clerk client: counts clients built and requests sent."""

    def __init__(self):
        self.built = 0
        self.sent = []

    def Client(self, **kwargs):  # noqa: N802 (httpx's name)
        self.built += 1
        return _SyncClient(self.sent)

    def AsyncClient(self, **kwargs):  # noqa: N802 (httpx's name)
        self.built += 1
        return _AsyncClient(self.sent)


@pytest.fixture
def http(monkeypatch):
    stand_in = _Http()
    monkeypatch.setattr(clerk_mod, "httpx", stand_in)
    return stand_in


@pytest.fixture
def clerk(monkeypatch, http):
    """A real Clerk client with a secret set, the one the routes get."""
    client = clerk_mod.ClerkAuth()
    client.clerk_secret_key = STAND_IN_CLERK_SECRET
    monkeypatch.setattr(clerk_mod, "get_clerk_auth", lambda: client)
    monkeypatch.setattr(team_api, "get_clerk_auth", lambda: client)
    return client


@pytest.fixture
def edition(monkeypatch):
    def use(name):
        monkeypatch.setattr(config, "AUTH_EDITION", name)
    return use


def _run(result):
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


# Every method of the Clerk client that calls Clerk's API, and what it gives back with no secret.
REFUSING_CALLS = {
    "_fetch_email_from_clerk_api": ((CLERK_USER,), None),
    "get_org_members": ((ORG,), []),
    "revoke_user_invitation": (("inv" + "_" + "prd277",), False),
}
RAISING_CALLS = {
    "invite_to_org": ((ORG, "new@example.com", "admin", CLERK_USER), None),
    "create_user_invitation": (("new@example.com", "https://automatos.example.test/accept-invitation"), None),
    "remove_from_org": ((ORG, CLERK_USER), None),
    "delete_user": ((CLERK_USER,), None),
    "list_users_page": ((10, 0), None),
}


# --------------------------------------------------------------------------- the Clerk client


def test_the_edition_does_not_use_clerk(edition):
    edition("enterprise")
    assert clerk_mod.clerk_in_use() is False
    for name in ("local", "saas"):
        edition(name)
        assert clerk_mod.clerk_in_use() is True


@pytest.mark.parametrize("method", sorted(REFUSING_CALLS))
def test_a_lookup_answers_as_with_no_secret_and_sends_nothing(edition, clerk, http, method):
    edition("enterprise")
    args, no_secret_answer = REFUSING_CALLS[method]

    assert _run(getattr(clerk, method)(*args)) == no_secret_answer
    assert http.built == 0 and http.sent == []


@pytest.mark.parametrize("method", sorted(RAISING_CALLS))
def test_a_change_refuses_as_not_set_up_and_sends_nothing(edition, clerk, http, method):
    edition("enterprise")
    args, _ = RAISING_CALLS[method]

    with pytest.raises(clerk_mod.ClerkNotUsed) as refused:
        _run(getattr(clerk, method)(*args))

    assert isinstance(refused.value, ValueError)  # handled wherever an unset secret is
    assert str(refused.value) == clerk_mod.CLERK_NOT_USED
    assert STAND_IN_CLERK_SECRET not in str(refused.value)
    assert http.built == 0 and http.sent == []


def test_no_email_is_looked_up_while_reading_claims(edition, clerk, http):
    edition("enterprise")
    info = clerk.extract_user_info({"sub": CLERK_USER})

    assert info["email"].endswith("@clerk.placeholder") and http.sent == []


@pytest.mark.parametrize("method", sorted({**REFUSING_CALLS, **RAISING_CALLS}))
def test_saas_still_calls_clerk(edition, clerk, http, method):
    """The control: the same stand-in is called on saas, so the enterprise tests are not vacuous."""
    edition("saas")
    args, _ = {**REFUSING_CALLS, **RAISING_CALLS}[method]

    _run(getattr(clerk, method)(*args))

    assert http.built == 1 and len(http.sent) == 1


# --------------------------------------------------------------------------- the team routes


class _Query:
    def __init__(self, row):
        self.row = row

    def filter(self, *criteria):
        return self

    def order_by(self, *columns):
        return self

    def first(self):
        return self.row

    def all(self):
        return []


class _Db:
    """One member (an editor with a Clerk id) and their person; nothing else."""

    def __init__(self):
        self.member = SimpleNamespace(id=5, user_id=9, role="editor", is_active=True)
        self.user = SimpleNamespace(id=9, clerk_user_id=CLERK_USER)
        self.commits = 0

    def query(self, model):
        if model is WorkspaceMember:
            return _Query(self.member)
        if model is User:
            return _Query(self.user)
        return _Query(None)

    def commit(self):
        self.commits += 1


@pytest.fixture
def team(monkeypatch, clerk):
    effects = SimpleNamespace(db=_Db(), invited=[], audits=[])

    async def invite(**fields):
        effects.invited.append(fields)
        raise AssertionError("the invite flow must not run")

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(team_api, "_resolve_internal_user_id", lambda db, ctx: 1)
    monkeypatch.setattr(team_api, "invite_member_to_workspace", invite)
    monkeypatch.setattr(team_api, "AuditService", lambda db: SimpleNamespace(log=lambda **e: effects.audits.append(e)))
    return effects


def _owner():
    user = UserContext(id=CLERK_USER, clerk_user_id=CLERK_USER, email="owner@example.com", org_id=ORG)
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _client(effects):
    app = FastAPI()
    app.include_router(team_api.router)
    app.include_router(team_api.public_router)
    app.dependency_overrides[get_request_context_hybrid] = _owner
    app.dependency_overrides[get_db] = lambda: effects.db
    return TestClient(app, raise_server_exceptions=False)


INVITATION_ROUTES = {
    "invite": ("post", f"/api/workspaces/{WS}/team/invite", {"json": {"email": "new@example.com", "role": "admin"}}),
    "list": ("get", f"/api/workspaces/{WS}/team/invitations", {}),
    "revoke": ("delete", f"/api/workspaces/{WS}/team/invitations/41", {}),
    "accept": ("post", "/api/team/accept-invitation", {"json": {"token": "prd277"}}),
    "info": ("get", "/api/team/invitation-info", {"params": {"token": "prd277"}}),
}


@pytest.mark.parametrize("name", sorted(INVITATION_ROUTES))
def test_every_invitation_route_is_404_in_enterprise(edition, team, http, name):
    edition("enterprise")
    method, path, kwargs = INVITATION_ROUTES[name]

    answer = getattr(_client(team), method)(path, **kwargs)

    assert answer.status_code == 404
    assert answer.json() == {"detail": invitations_mod.INVITATIONS_OFF}
    assert team.invited == [] and team.audits == [] and team.db.commits == 0
    assert http.built == 0


@pytest.mark.parametrize("name", ["local", "saas"])
def test_local_and_saas_still_reach_the_invitation_routes(edition, team, name):
    """The control: the same request is answered by the route itself, not the edition."""
    edition(name)
    answer = _client(team).get(f"/api/workspaces/{WS}/team/invitations")

    assert answer.status_code == 200 and answer.json() == []


def test_removing_a_member_in_enterprise_never_asks_clerk(edition, team, http):
    edition("enterprise")
    answer = _client(team).delete(f"/api/workspaces/{WS}/team/members/5")

    assert answer.status_code == 200 and answer.json() == {"status": "success"}
    assert team.db.member.is_active is False and team.audits[0]["action"] == "member:removed"
    assert http.built == 0 and http.sent == []


def test_removing_a_member_in_saas_still_removes_them_from_the_clerk_organisation(edition, team, http):
    edition("saas")
    answer = _client(team).delete(f"/api/workspaces/{WS}/team/members/5")

    assert answer.status_code == 200 and team.db.member.is_active is False
    assert [method for method, _ in http.sent] == ["DELETE"]
    assert http.sent[0][1].endswith(f"/v1/organizations/{ORG}/memberships/{CLERK_USER}")


# --------------------------------------------------------------------------- Auto's invite tool


def test_autos_invite_tool_refuses_in_enterprise_and_stores_nothing(edition, monkeypatch, http):
    edition("enterprise")
    flows = []
    monkeypatch.setattr(invitations_mod, "invite_member_to_workspace", lambda **fields: flows.append(fields))

    answer = asyncio.run(handlers_members.invite_member(None, WS, {"email": "new@example.com", "role": "admin"}))

    assert answer == {"success": False, "error": invitations_mod.INVITATIONS_OFF}
    assert flows == [] and http.built == 0
