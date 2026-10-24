"""PRD-266 US-009 (§4.3, §4.4, K14) — phones register for push on W1's device routes.

``POST /api/v1/mobile/devices`` over ``services/mobile/devices.py``, on Postgres. Proven here:

* A registration that carries an Expo token turns push on and clears the reason it was off,
  for a Clerk caller and for a paired phone through the real device lane; one without a token
  leaves push as it was; ``push_key`` is returned only when minted; a clerk row never holds a
  code or token hash (§8).
* The token handover (K14): a token another row holds moves only from the same install (in
  another workspace) or the same person (a new install), and that row loses it
  (``token_moved``, a ``push_token_moved`` event); a different person's install is
  ``409 push_token_in_use``, decided by the service before any write (never by the unique
  index), and the holder keeps it.
* ``PATCH push_enabled: true`` without a registered token is ``409 push_token_missing``; false
  keeps the token.
* Sign-out (W1's ``DELETE``): a question then reaches the person's other phone only
  (``fanout.deliver`` with a fake sender).
* The phone's name: stored and listed, 121 characters is 422, a blank name is not written, a
  registration without one keeps it, and it replaces the name a paired phone was given.
* Discovery lists ``push`` on the registration route.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

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

from config import config  # noqa: E402
from core.models.mobile_devices import NAME_LENGTH, MobileDevice  # noqa: E402
from services.mobile import devices  # noqa: E402
from services.push import fanout  # noqa: E402
from services.push.fanout import PushEvent  # noqa: E402
from services.push.sender import DEVICE_NOT_REGISTERED, TICKET_OK, PushTicket  # noqa: E402

DEVICES = "/api/v1/mobile/devices"
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
T0 = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
TOKEN_BYTES = 32
PHONE_NAME = "iPhone 16 Pro"


def _one(device_id) -> str:
    return f"{DEVICES}/{device_id}"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _install() -> str:
    return f"prd266-push-reg-{uuid.uuid4().hex[:12]}"


def _expo(tag: str) -> str:
    return f"ExponentPushToken[{tag}]"


# An override that leaves the field out of the body (not the same as sending null).
_LEFT_OUT = object()


def _body(install_id: str, /, **overrides) -> dict:
    # ``install_id`` is positional-only, so an override may drop or replace any field.
    fields = {"install_id": install_id, "platform": "ios", "app_version": "1.0.0", "locale": "en-IE",
              "timezone": "Europe/Dublin", "expo_push_token": _expo(install_id), **overrides}
    return {key: value for key, value in fields.items() if value is not _LEFT_OUT}


# --------------------------------------------------------------------------- fixtures


class _BorrowedSession:
    """The test's transactional session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeSender:
    """Accepts every message with a fresh ticket id, and remembers who it was sent to."""

    def __init__(self):
        self.sent = []

    def send(self, messages, /):
        self.sent.extend(messages)
        return [PushTicket(to=m.to, status=TICKET_OK, ticket_id=str(uuid.uuid4())) for m in messages]

    def receipts(self, ticket_ids, /):
        return {}


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    A 409 rolls the route's work back and ``deliver`` commits; on this session either reaches
    the last savepoint only, and every seed commits (``_settle``), so neither takes the seed.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    from core.auth import device

    monkeypatch.setattr(device, "_now", lambda: T0)
    monkeypatch.setattr(devices, "_now", lambda: T0)
    monkeypatch.setattr(fanout, "_now", lambda: T0)


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    return config


@pytest.fixture
def lane(monkeypatch, db_session):
    """The local edition's real device lane, on the test's session."""
    from core.auth import device, hybrid

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(device, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return config


@pytest.fixture
def make_client(db_session):
    """A client on W1's router: ``ctx`` stands in for the request context, or (None) the real lane."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    def _make(ctx=None):
        app = FastAPI()
        app.include_router(mobile_devices.router)
        app.dependency_overrides[get_db] = lambda: db_session
        if ctx is not None:
            app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    _settle(db_session)
    return workspace_id


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture
def preferences_table(db_session):
    """``notification_preferences`` as production has it: CI's test database lacks it (prd128's DDL)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


# --------------------------------------------------------------------------- seeds


def _settle(db) -> None:
    """Release the seed's savepoint (``db_session``): a route's rollback never reaches it."""
    db.commit()


def _person(db) -> tuple:
    """A users row with a Clerk id: ``(pk, email, clerk_user_id)``."""
    tag = uuid.uuid4().hex[:12]
    email, clerk_id = f"prd266-push-reg-{tag}@t.test", f"user_prd266pr{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, clerk_user_id) "
             "VALUES (:u, :e, true, :c) RETURNING id"),
        {"u": f"prd266-push-reg-{tag}", "e": email, "c": clerk_id},
    ).scalar()
    _settle(db)
    return pk, email, clerk_id


def _member(db, workspace_id, user_pk, role="editor") -> None:
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, TRUE) ON CONFLICT DO NOTHING"),
               {"ws": str(workspace_id), "user": user_pk, "role": role})
    _settle(db)


def _clerk_ctx(workspace_id, person):
    from core.auth.dependencies import RequestContext, UserContext

    _pk, email, clerk_id = person
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk")


def _clerk(db, workspace_id, person=None, role="editor"):
    """A Clerk caller who is a member of ``workspace_id``: ``(ctx, pk)``."""
    person = person or _person(db)
    _member(db, workspace_id, person[0], role)
    return _clerk_ctx(workspace_id, person), person[0]


def _paired(db, workspace_id, user_pk, **fields):
    """An active paired row (a member's) and its device token: ``(row, token)``."""
    from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX
    from core.auth.pairing import mint_token

    _member(db, workspace_id, user_pk, "member")
    token, digest = mint_token(DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    row = MobileDevice(workspace_id=workspace_id, user_id=user_pk, kind="paired", status="active",
                       install_id=_install(), platform="ios", token_hash=digest, paired_at=T0,
                       events=[], **fields)
    db.add(row)
    _settle(db)
    return row, token


def _row(db, device_id) -> MobileDevice:
    db.expire_all()
    return db.get(MobileDevice, uuid.UUID(str(device_id)))


def _register(client, install_id, **overrides) -> dict:
    response = client.post(DEVICES, json=_body(install_id, **overrides))
    assert response.status_code == 200, response.text
    return response.json()


def _rows_of(db, user_pk) -> list:
    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.user_id == user_pk).all()


# --------------------------------------------------------------------------- push on


def test_a_clerk_registration_with_a_token_turns_push_on_and_returns_push_key_once(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client, install_id = make_client(ctx), _install()

    first = _register(client, install_id)
    second = _register(client, install_id, app_version="1.1.0")

    assert first["push_key"] and second["push_key"] is None
    row = _row(db_session, first["device_id"])
    assert (row.kind, row.push_enabled, row.push_disabled_reason) == ("clerk", True, None)
    assert row.expo_push_token == _expo(install_id) and row.push_key == first["push_key"]
    assert (row.token_hash, row.prev_token_hash, row.pairing_code_hash) == (None, None, None)


def test_a_registration_without_a_token_leaves_push_as_it_was(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client, install_id = make_client(ctx), _install()
    device_id = _register(client, install_id, expo_push_token=_LEFT_OUT)["device_id"]
    assert _row(db_session, device_id).push_enabled is False
    _register(client, install_id)

    _register(client, install_id, expo_push_token=_LEFT_OUT)
    assert _row(db_session, device_id).push_enabled is True
    assert client.patch(_one(device_id), json={"push_enabled": False}).status_code == 200
    _register(client, install_id, expo_push_token=_LEFT_OUT)

    row = _row(db_session, device_id)
    assert (row.push_enabled, row.expo_push_token) == (False, _expo(install_id))


def test_a_paired_phone_registering_its_token_again_turns_push_back_on(db_session, ws, lane, make_client):
    pk = _person(db_session)[0]
    row, token = _paired(db_session, ws, pk, name="Kitchen phone", expo_push_token=_expo("paired"),
                         push_enabled=False, push_disabled_reason=DEVICE_NOT_REGISTERED)

    response = make_client().post(DEVICES, json=_body(row.install_id, expo_push_token=_expo("paired"),
                                                      name=PHONE_NAME), headers=_bearer(token))

    assert response.status_code == 200, response.text
    assert response.json()["push_key"]
    stored = _row(db_session, row.id)
    assert (stored.push_enabled, stored.push_disabled_reason) == (True, None)
    assert (stored.expo_push_token, stored.name) == (_expo("paired"), PHONE_NAME)


# --------------------------------------------------------------------------- the handover (K14)


def test_the_same_install_in_another_workspace_takes_its_token(db_session, ws, seed_workspace, make_client):
    other_ws = uuid.UUID(seed_workspace())
    _settle(db_session)
    install_id, token = _install(), _expo(uuid.uuid4().hex)
    first, _ = _clerk(db_session, ws)
    second, _ = _clerk(db_session, other_ws)  # another account, signed in on the same phone
    old = _register(make_client(first), install_id, expo_push_token=token)

    new = _register(make_client(second), install_id, expo_push_token=token)

    moved = _row(db_session, old["device_id"])
    assert (moved.expo_push_token, moved.push_enabled, moved.push_disabled_reason) == (None, False, "token_moved")
    assert moved.events[-1]["event"] == "push_token_moved" and moved.status == "active"
    taker = _row(db_session, new["device_id"])
    assert (taker.workspace_id, taker.expo_push_token, taker.push_enabled) == (other_ws, token, True)


def test_the_same_persons_new_install_takes_its_token(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client, token = make_client(ctx), _expo(uuid.uuid4().hex)
    old = _register(client, _install(), expo_push_token=token)

    new = _register(client, _install(), expo_push_token=token)  # the app reinstalled: a new install_id

    assert new["device_id"] != old["device_id"]
    moved = _row(db_session, old["device_id"])
    assert (moved.expo_push_token, moved.push_disabled_reason) == (None, "token_moved")
    assert _row(db_session, new["device_id"]).expo_push_token == token


def test_a_different_persons_install_is_409_and_the_holder_keeps_it(db_session, ws, seed_workspace, make_client):
    other_ws = uuid.UUID(seed_workspace())
    _settle(db_session)
    token = _expo(uuid.uuid4().hex)
    holder, _ = _clerk(db_session, ws)
    taker, taker_pk = _clerk(db_session, other_ws)
    held = _register(make_client(holder), _install(), expo_push_token=token)

    clash = make_client(taker).post(DEVICES, json=_body(_install(), expo_push_token=token))

    assert (clash.status_code, clash.json()) == (409, {"detail": {"code": "push_token_in_use"}})
    kept = _row(db_session, held["device_id"])
    assert (kept.expo_push_token, kept.push_enabled, kept.push_disabled_reason) == (token, True, None)
    assert [event["event"] for event in kept.events] == ["enrolled"]
    assert _rows_of(db_session, taker_pk) == []


def test_the_service_refuses_another_persons_token_before_any_write(db_session, ws, make_client):
    token = _expo(uuid.uuid4().hex)
    holder, _ = _clerk(db_session, ws)
    _register(make_client(holder), _install(), expo_push_token=token)
    taker_pk = _clerk(db_session, ws)[1]
    caller = devices.DeviceCaller(workspace_id=ws, user_pk=taker_pk)

    with pytest.raises(devices.DeviceConflict) as refused:
        devices.register_device(db_session, caller, _install(), {"expo_push_token": token})

    assert refused.value.code == "push_token_in_use"
    db_session.rollback()
    assert _rows_of(db_session, taker_pk) == []


# --------------------------------------------------------------------------- PATCH push_enabled


def test_patch_push_on_without_a_token_is_409_push_token_missing(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client = make_client(ctx)
    device_id = _register(client, _install(), expo_push_token=_LEFT_OUT)["device_id"]

    refused = client.patch(_one(device_id), json={"push_enabled": True})

    assert (refused.status_code, refused.json()) == (409, {"detail": {"code": "push_token_missing"}})
    assert _row(db_session, device_id).push_enabled is False


def test_patch_push_off_keeps_the_token(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client, install_id = make_client(ctx), _install()
    device_id = _register(client, install_id)["device_id"]

    response = client.patch(_one(device_id), json={"push_enabled": False})

    assert response.status_code == 200, response.text
    assert (response.json()["push_enabled"], response.json()["push_registered"]) == (False, True)
    assert _row(db_session, device_id).expo_push_token == _expo(install_id)


# --------------------------------------------------------------------------- sign-out


def test_after_sign_out_a_question_reaches_the_persons_other_phone_only(db_session, ws, saas, make_client,
                                                                      preferences_table):
    ctx, _pk = _clerk(db_session, ws, role="editor")
    client = make_client(ctx)
    signed_out = _register(client, _install())
    other_install = _install()
    _register(client, other_install)
    assert client.delete(_one(signed_out["device_id"])).status_code == 200
    sender = _FakeSender()

    counts = fanout.deliver(PushEvent(workspace_id=str(ws), event_type="question_pending", link_type="question",
                                      link_id="0"), sender=sender, session_factory=lambda: _BorrowedSession(db_session))

    assert {message.to for message in sender.sent} == {_expo(other_install)}
    assert counts["devices"] == 1
    revoked = _row(db_session, signed_out["device_id"])
    assert (revoked.status, revoked.expo_push_token, revoked.push_key) == ("revoked", None, None)


# --------------------------------------------------------------------------- the phone's name


def test_a_clerk_registration_stores_the_phones_name_and_lists_it(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client = make_client(ctx)
    device_id = _register(client, _install(), name=PHONE_NAME)["device_id"]

    listed = client.get(DEVICES)

    assert listed.status_code == 200, listed.text
    assert [device["name"] for device in listed.json()["devices"] if device["id"] == device_id] == [PHONE_NAME]


def test_a_name_over_the_limit_is_422(db_session, ws, make_client):
    ctx, pk = _clerk(db_session, ws)

    response = make_client(ctx).post(DEVICES, json=_body(_install(), name="x" * (NAME_LENGTH + 1)))

    assert response.status_code == 422
    assert _rows_of(db_session, pk) == []


def test_a_blank_name_is_not_written_and_a_registration_without_one_keeps_it(db_session, ws, make_client):
    ctx, _pk = _clerk(db_session, ws)
    client, install_id = make_client(ctx), _install()
    device_id = _register(client, install_id, name=f"  {PHONE_NAME} ")["device_id"]
    assert _row(db_session, device_id).name == PHONE_NAME

    _register(client, install_id, name="   ")
    assert _row(db_session, device_id).name == PHONE_NAME
    _register(client, install_id)
    assert _row(db_session, device_id).name == PHONE_NAME


# --------------------------------------------------------------------------- discovery


def test_discovery_lists_push_on_the_registration_route(monkeypatch):
    import json

    from api import mobile_devices
    from config import HOSTED_PUBLIC_API_HOST
    from services.mobile import discovery
    from services.push import consent

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    # The operator has not answered "Send notifications through Expo?": the capability is listed anyway.
    monkeypatch.setattr(consent, "get_system_setting", lambda category, key, default=None: default)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())

    document = discovery.build_discovery("http://192.168.1.20:8000/")

    assert "push" in document["capabilities"]
    assert discovery.CAPABILITY_ROUTES["push"] == ("POST", DEVICES)
    assert {"method": "POST", "path": DEVICES} in manifest["routes"]
    assert any(route.path == DEVICES and "POST" in route.methods for route in mobile_devices.router.routes)
