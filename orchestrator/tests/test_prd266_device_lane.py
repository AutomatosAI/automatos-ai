"""PRD-266 US-004a — the device lane: a paired phone's ``Bearer amd_…``.

``core/auth/bearer_registry.py`` maps the ``amd_`` prefix to
``core.auth.device.resolve_device_context`` and decorates ``get_request_context_hybrid``
so a registered bearer is resolved BEFORE the Clerk lane. Proven here, against Postgres:

* local: ``amd_`` resolves as ``auth_type="device"``, in the row's workspace, as the row's
  person (id = the email; the operator ``super_admin``, anyone else ``user``) with
  ``raw_claims {source: "mobile_device", device_id, user_id}``;
* saas: 401, nothing looked up, and Clerk's ``verify_token`` is never called;
* an unknown token is 401; a revoked row is ``401 {"code": "device_revoked"}``; a
  deactivated person is 401; a person dropped from the device's workspace (still active
  elsewhere) is 401 too — the local operator is exempt, having no membership row at all;
* after a rotation the previous token works strictly before ``prev_token_valid_until`` and
  is 401 from then on (a frozen clock);
* a request naming another workspace (either header or the query: every input hybrid
  reads) is 403; the same one or none resolves to the bound workspace; hybrid's workspace
  rules apply (deleted 400, paused 403);
* ``last_seen_at`` is written at most every five minutes, and a failed write is logged by
  type and never costs the request; the session is always closed and no token reaches the
  log;
* a non-``amd_`` bearer reaches the wrapped function unchanged; the decorated dependency
  keeps FastAPI's view (``Depends`` and ``dependency_overrides``) and its ``__wrapped__``
  is hybrid's own function, whose ``_get_bearer_token`` now lives in the registry;
* ``resolve_user_pk`` reads a ``mobile_device`` claim and ignores an unknown source.
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from config import config  # noqa: E402
from core.auth import bearer_registry, device, hybrid  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.auth.principal import resolve_user_pk  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402

T0 = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
TOKEN_BYTES = 32


# --------------------------------------------------------------------------- fixtures


class _BorrowedSession:
    """The test's transactional session, lent to the lane: ``close()`` is counted, not done."""

    def __init__(self, session):
        self._session = session
        self.closes = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        self.closes += 1


class _Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _ClerkSpy:
    """Stands in for Clerk: records every token it is asked to verify and accepts none."""

    def __init__(self):
        self.verified = []

    def verify_token(self, token):
        self.verified.append(token)
        return None


@pytest.fixture
def clock(monkeypatch):
    frozen = _Clock(T0)
    monkeypatch.setattr(device, "_now", frozen)
    return frozen


@pytest.fixture
def lane_db(db_session, monkeypatch):
    borrowed = _BorrowedSession(db_session)
    monkeypatch.setattr(device, "SessionLocal", lambda: borrowed)
    return borrowed


@pytest.fixture
def local(monkeypatch, lane_db, clock):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return config


@pytest.fixture
def clerk_spy(monkeypatch):
    spy = _ClerkSpy()
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: spy)
    return spy


def _seed_user(db) -> tuple:
    tag = uuid.uuid4().hex[:12]
    email = f"prd266-lane-{tag}@t.test"
    user_pk = db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-lane-{tag}", "e": email},
    ).scalar()
    return user_pk, email


def _member(db, workspace_id, user_pk, *, active=True) -> None:
    db.execute(
        text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
             "VALUES (CAST(:ws AS uuid), :uid, 'member', :active)"),
        {"ws": str(workspace_id), "uid": user_pk, "active": active},
    )
    db.flush()


@pytest.fixture
def person(db_session, seed_workspace):
    """A workspace and its person, an active member of it: ``(workspace_id, user_pk, email)``."""
    user_pk, email = _seed_user(db_session)
    workspace_id = uuid.UUID(seed_workspace())
    _member(db_session, workspace_id, user_pk)
    return workspace_id, user_pk, email


def _paired(db, person, *, status="active", last_seen_at=None):
    """An active paired row for ``person``; returns ``(row, token)``."""
    token, digest = mint_token(bearer_registry.DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    row = MobileDevice(
        workspace_id=person[0], user_id=person[1], kind="paired", status=status,
        install_id=f"prd266-lane-{uuid.uuid4().hex[:12]}", platform="ios",
        token_hash=digest, paired_at=T0, last_seen_at=last_seen_at, events=[],
    )
    db.add(row)
    db.flush()
    return row, token


def _rotate(db, row, valid_until: datetime) -> str:
    """Rotate ``row``'s token: the old hash becomes the previous one until ``valid_until``."""
    token, digest = mint_token(bearer_registry.DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    row.prev_token_hash, row.prev_token_valid_until = row.token_hash, valid_until
    row.token_hash, row.token_rotated_at = digest, T0
    db.flush()
    return token


def _request(token=None, headers=None, query=None):
    sent = {"authorization": f"Bearer {token}"} if token else {}
    sent.update(headers or {})
    return SimpleNamespace(method="GET", headers=sent, query_params=dict(query or {}), state=SimpleNamespace())


async def _resolve(token, **kwargs):
    return await hybrid.get_request_context_hybrid(_request(token, **kwargs))


async def _refused(token, **kwargs) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await _resolve(token, **kwargs)
    return refused.value


def _last_seen(db, row):
    db.expire_all()
    return db.get(MobileDevice, row.id).last_seen_at


# --------------------------------------------------------------------------- the lane resolves


@pytest.mark.asyncio
async def test_a_device_token_resolves_as_its_person_in_its_workspace(db_session, person, local, lane_db):
    row, token = _paired(db_session, person)

    ctx = await _resolve(token)

    assert ctx.auth_type == "device" and ctx.workspace_id == person[0]
    assert (ctx.user.id, ctx.user.email, ctx.user.system_role) == (person[2], person[2], "user")
    assert ctx.user.clerk_user_id is None
    assert ctx.user.raw_claims == {"source": "mobile_device", "device_id": str(row.id), "user_id": person[1]}
    assert lane_db.closes == 1


@pytest.mark.asyncio
async def test_the_local_operators_device_is_super_admin(db_session, person, local, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", person[2])
    _, token = _paired(db_session, person)

    ctx = await _resolve(token)

    assert ctx.user.system_role == "super_admin" and ctx.user.id == person[2]


@pytest.mark.asyncio
async def test_saas_refuses_an_amd_bearer_without_a_lookup_or_clerk(clerk_spy, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(device, "SessionLocal", lambda: pytest.fail("saas must not look a device up"))

    refused = await _refused("amd_" + "x" * 43)

    assert refused.status_code == 401
    assert clerk_spy.verified == []


@pytest.mark.asyncio
async def test_an_unknown_token_is_401_and_never_reaches_clerk(local, lane_db, clerk_spy, caplog):
    token, _ = mint_token(bearer_registry.DEVICE_TOKEN_PREFIX, TOKEN_BYTES)

    with caplog.at_level(logging.DEBUG):
        refused = await _refused(token)

    assert refused.status_code == 401 and refused.detail != {"code": "device_revoked"}
    assert clerk_spy.verified == []
    assert lane_db.closes == 1
    assert token not in caplog.text


@pytest.mark.asyncio
async def test_a_revoked_device_is_401_device_revoked(db_session, person, local, lane_db):
    _, token = _paired(db_session, person, status="revoked")

    refused = await _refused(token)

    assert (refused.status_code, refused.detail) == (401, {"code": "device_revoked"})
    assert lane_db.closes == 1


@pytest.mark.asyncio
async def test_a_deactivated_person_is_401(db_session, person, local):
    _, token = _paired(db_session, person)
    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :id"), {"id": person[1]})

    refused = await _refused(token)

    assert refused.status_code == 401 and refused.detail != {"code": "device_revoked"}


@pytest.mark.asyncio
async def test_a_person_dropped_from_the_workspace_is_401(db_session, person, local):
    """A device paired while its person was a member keeps no access once they're removed."""
    _, token = _paired(db_session, person)
    db_session.execute(
        text("UPDATE workspace_members SET is_active = false WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid"),
        {"ws": str(person[0]), "uid": person[1]},
    )

    refused = await _refused(token)

    assert refused.status_code == 401 and refused.detail != {"code": "device_revoked"}


@pytest.mark.asyncio
async def test_the_local_operator_needs_no_membership_row(db_session, person, local, monkeypatch):
    """The operator binds by email (hybrid's local lane), never by workspace_members."""
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", person[2])
    db_session.execute(
        text("DELETE FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :uid"),
        {"ws": str(person[0]), "uid": person[1]},
    )
    _, token = _paired(db_session, person)

    ctx = await _resolve(token)

    assert ctx.user.system_role == "super_admin"


@pytest.mark.asyncio
async def test_the_previous_token_works_until_the_overlap_ends(db_session, person, local, clock):
    row, old_token = _paired(db_session, person)
    new_token = _rotate(db_session, row, valid_until=T0 + timedelta(minutes=10))

    clock.now = T0 + timedelta(minutes=10) - timedelta(seconds=1)
    assert (await _resolve(old_token)).user.raw_claims["device_id"] == str(row.id)
    assert (await _resolve(new_token)).workspace_id == person[0]

    clock.now = T0 + timedelta(minutes=10)
    assert (await _refused(old_token)).status_code == 401
    assert (await _resolve(new_token)).auth_type == "device"


# --------------------------------------------------------------------------- the workspace is the row's


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["x-workspace-id", "x-workspace", "query"])
async def test_naming_another_workspace_is_403(db_session, person, local, seed_workspace, where):
    _, token = _paired(db_session, person)
    other = seed_workspace()
    named = {"query": {"workspace_id": other}} if where == "query" else {"headers": {where: other}}

    refused = await _refused(token, **named)

    assert refused.status_code == 403


def _inputs_read(function, source: str) -> set:
    """The literal names ``function`` passes to ``request.<source>.get(...)``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == source
        and node.args and isinstance(node.args[0], ast.Constant)
    }


def test_the_lane_checks_every_workspace_input_hybrid_reads():
    reader = hybrid._get_workspace_id_from_request

    assert _inputs_read(reader, "headers") == set(device.WORKSPACE_HEADERS)
    assert _inputs_read(reader, "query_params") == {device.WORKSPACE_QUERY_PARAM}


@pytest.mark.asyncio
async def test_an_unparsable_workspace_header_is_403(db_session, person, local):
    _, token = _paired(db_session, person)

    assert (await _refused(token, headers={"x-workspace-id": "__all__"})).status_code == 403


@pytest.mark.asyncio
async def test_the_same_workspace_or_none_resolves_to_the_bound_one(db_session, person, local, monkeypatch):
    other_default = str(uuid.uuid4())
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", other_default)
    _, token = _paired(db_session, person)

    assert (await _resolve(token)).workspace_id == person[0]
    assert (await _resolve(token, headers={"x-workspace-id": str(person[0])})).workspace_id == person[0]


@pytest.mark.asyncio
async def test_hybrids_workspace_rules_apply(db_session, person, local):
    _, token = _paired(db_session, person)
    db_session.execute(text("UPDATE workspaces SET paused_at = now() WHERE id = :id"), {"id": str(person[0])})
    assert (await _refused(token)).status_code == 403

    db_session.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = :id"), {"id": str(person[0])})
    assert (await _refused(token)).status_code == 400


# --------------------------------------------------------------------------- last seen


@pytest.mark.asyncio
async def test_last_seen_is_written_at_most_every_five_minutes(db_session, person, local, clock):
    row, token = _paired(db_session, person)
    assert device.LAST_SEEN_WRITE_INTERVAL == timedelta(minutes=5)

    await _resolve(token)
    assert _last_seen(db_session, row) == T0

    clock.now = T0 + timedelta(minutes=4, seconds=59)
    await _resolve(token)
    assert _last_seen(db_session, row) == T0

    clock.now = T0 + timedelta(minutes=5)
    await _resolve(token)
    assert _last_seen(db_session, row) == T0 + timedelta(minutes=5)


@pytest.mark.asyncio
async def test_a_failed_last_seen_write_is_logged_and_never_costs_the_request(db_session, person, local, lane_db,
                                                                               caplog):
    _, token = _paired(db_session, person)
    rollbacks = []

    def _database_gone():
        raise OperationalError("UPDATE mobile_devices", {}, Exception("server closed the connection"))

    lane_db.commit = _database_gone
    lane_db.rollback = lambda: rollbacks.append(True)
    with caplog.at_level(logging.WARNING, logger="core.auth.device"):
        ctx = await _resolve(token)

    assert ctx.auth_type == "device" and rollbacks == [True] and lane_db.closes == 1
    assert "OperationalError" in caplog.text
    assert token not in caplog.text and "server closed" not in caplog.text


@pytest.mark.asyncio
async def test_a_refused_request_writes_no_last_seen(db_session, person, local, seed_workspace):
    row, token = _paired(db_session, person)

    await _refused(token, headers={"x-workspace-id": seed_workspace()})

    assert _last_seen(db_session, row) is None


# --------------------------------------------------------------------------- the decorator


@pytest.mark.asyncio
async def test_a_non_amd_bearer_reaches_the_wrapped_function_unchanged():
    calls = []

    async def wrapped(request):
        calls.append(request)
        return "wrapped"

    decorated = bearer_registry.resolve_registered_bearers(wrapped)
    for request in (_request("eyJhbGciOi.jwt.sig"), _request(), _request("AMD_upper")):
        assert await decorated(request) == "wrapped"
        assert calls[-1] is request
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_a_clerk_bearer_still_goes_to_clerk(clerk_spy, monkeypatch):
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))

    refused = await _refused("eyJhbGciOi.jwt.sig")

    assert refused.status_code == 401
    assert clerk_spy.verified == ["eyJhbGciOi.jwt.sig"]


def test_get_request_context_hybrid_is_the_decorated_function():
    decorated = hybrid.get_request_context_hybrid
    original = decorated.__wrapped__

    assert decorated is not original and inspect.iscoroutinefunction(decorated)
    assert original.__module__ == "core.auth.hybrid" and original.__name__ == "get_request_context_hybrid"
    decorator, definition = inspect.getsource(original).splitlines()[:2]
    assert decorator.strip() == "@resolve_registered_bearers"
    assert definition.startswith("async def get_request_context_hybrid(request: Request)")
    assert list(inspect.signature(decorated).parameters) == ["request"]
    assert bearer_registry.BEARER_RESOLVERS["amd_"] is device.resolve_device_context


def test_hybrid_reads_the_bearer_through_the_registry():
    assert hybrid._get_bearer_token is bearer_registry.bearer_token
    assert "def _get_bearer_token" not in inspect.getsource(hybrid)
    assert bearer_registry.bearer_token(_request("amd_abc")) == "amd_abc"
    assert bearer_registry.bearer_token(_request(headers={"authorization": "Basic x"})) is None


def _whoami_app() -> FastAPI:
    app = FastAPI()

    @app.get("/whoami")
    def whoami(ctx: RequestContext = Depends(hybrid.get_request_context_hybrid)):
        return {"auth_type": ctx.auth_type, "workspace_id": str(ctx.workspace_id), "email": ctx.user.email}

    return app


def test_fastapi_still_injects_the_request_into_the_decorated_dependency(db_session, person, local):
    row, token = _paired(db_session, person)
    client = TestClient(_whoami_app())

    response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200, response.text
    assert response.json() == {"auth_type": "device", "workspace_id": str(person[0]), "email": person[2]}
    row.status = "revoked"
    db_session.flush()
    revoked = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert (revoked.status_code, revoked.json()) == (401, {"detail": {"code": "device_revoked"}})


def test_dependency_overrides_still_key_on_the_decorated_function():
    app = _whoami_app()
    ctx = RequestContext(workspace_id=uuid.uuid4(), user=UserContext(email="override@t.test"), auth_type="device")
    app.dependency_overrides[hybrid.get_request_context_hybrid] = lambda: ctx

    response = TestClient(app).get("/whoami")

    assert response.json()["email"] == "override@t.test"


# --------------------------------------------------------------------------- resolve_user_pk


def test_resolve_user_pk_reads_a_mobile_device_claim_without_a_query():
    user = UserContext(id="phone@t.test", email="phone@t.test",
                       raw_claims={"source": "mobile_device", "device_id": "d", "user_id": 42})

    assert resolve_user_pk(None, RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="device")) == 42


@pytest.mark.parametrize("claims", [
    {"source": "jwt", "user_id": 42},
    {"user_id": 42},
    {"source": "mobile_device", "user_id": True},
    {"source": "mobile_device", "user_id": "42"},
])
def test_resolve_user_pk_ignores_an_unknown_source_or_a_non_integer(claims):
    user = UserContext(raw_claims=claims)

    assert resolve_user_pk(None, RequestContext(workspace_id=uuid.uuid4(), user=user)) is None
