"""PRD-266 US-010c (D21, K13) — the operator's answer to "Send notifications through Expo's push service?".

* ``services/push/consent.py``: ``expo_push_allowed()`` is True on SaaS whatever is stored; on a
  local install only when ``notifications.push_via_expo`` is ``'true'``, and False when the
  read fails (three ways). ``set_push_consent`` writes ``push_via_expo`` and
  ``push_via_expo_answered_at`` (ISO-8601, UTC), creates a missing row (and overwrites every
  duplicate), flushes and never commits. ``consent_state`` reads strictly: a failed read raises,
  and the GET is 503, never "not answered". Not an environment setting: nothing in
  ``config.py`` or ``config-surface.json``.
* ``GET`` / ``PUT /api/v1/mobile/push-consent`` (``api/mobile_push.py``): 404 on SaaS before any
  authentication; ``require_workspace_admin`` (the local operator passes as super_admin, and so
  does a phone bound to the operator through W1's real device lane); a plain member and a
  viewer's phone are 403 and write nothing; the body is ``{agreed: bool}`` only; a failed write
  is 503 and rolls back.
* Discovery's ``push.expo`` follows the answer on a local install.
* Plain ``def`` routes, registered by ``RouterSpec("api.mobile_push")``, in the route manifest.

The settings reads (``get_system_setting``, its own ``SessionLocal``) and the device lane are
lent the test's session, so they see what the routes wrote inside the test's transaction.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import bindparam, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

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

from config import HOSTED_PUBLIC_API_HOST, config  # noqa: E402
from core.auth import device, hybrid  # noqa: E402
from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from services.push import consent  # noqa: E402
from tests.helpers_unreadable_settings import UNREADABLE_MODES, settings_unreadable  # noqa: E402

PUSH_CONSENT = "/api/v1/mobile/push-consent"
DISCOVERY = "/.well-known/automatos.json"
T0 = datetime(2026, 10, 24, 9, 30, tzinfo=timezone.utc)
TOKEN_BYTES = 32
NOT_ANSWERED = {"agreed": False, "answered": False, "answered_at": None}
CONSENT_KEYS = (consent.KEY_PUSH_VIA_EXPO, consent.KEY_ANSWERED_AT)


class _BorrowedSession:
    """The test's session, lent to a reader that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The PUT commits, and its failure path rolls back; here that reaches the last savepoint
    only, and every seed commits (``_settle``), so no route can take the seed with it.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    """Release the seed's savepoint: a route's rollback never reaches it."""
    db.commit()


@pytest.fixture
def settings_here(db_session, monkeypatch):
    """No stored answer, and every settings read (and the device lane) on the test's session."""
    from core.database import database

    db_session.execute(
        text("DELETE FROM system_settings WHERE category = :category AND key IN :keys")
        .bindparams(bindparam("keys", expanding=True)),
        {"category": consent.CONSENT_CATEGORY, "keys": list(CONSENT_KEYS)},
    )
    _settle(db_session)
    monkeypatch.setattr(database, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(device, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(device, "_now", lambda: T0)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    monkeypatch.setattr(consent, "_now", lambda: T0)
    return db_session


@pytest.fixture
def local(monkeypatch, settings_here):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", None)
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    return config


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    _settle(db_session)
    return workspace_id


@pytest.fixture
def make_client(db_session):
    """A client on the push and discovery routers; ``ctx`` stands in for the request context."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_push, well_known

    def _make(ctx=None):
        app = FastAPI()
        app.include_router(mobile_push.router)
        app.include_router(well_known.router)
        app.dependency_overrides[get_db] = lambda: db_session
        if ctx is not None:
            app.dependency_overrides[hybrid.get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


def _user(db, email=None) -> tuple:
    tag = uuid.uuid4().hex[:12]
    email = email or f"prd266-consent-{tag}@t.test"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-consent-{tag}", "e": email},
    ).scalar()
    _settle(db)
    return pk, email


@pytest.fixture
def operator(db_session, ws):
    """The local operator: super_admin, its users PK on raw_claims (hybrid's local lane)."""
    user_pk, email = _user(db_session)
    user = UserContext(id=email, email=email, system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": user_pk})
    return RequestContext(workspace_id=ws, user=user)


def _paired_phone(db, workspace_id, user_pk) -> dict:
    """An active paired row for the person, and its token's ``Authorization`` header."""
    token, digest = mint_token(DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    db.add(MobileDevice(workspace_id=workspace_id, user_id=user_pk, kind="paired", status="active",
                        install_id=f"prd266-consent-{uuid.uuid4().hex[:12]}", platform="ios",
                        token_hash=digest, paired_at=T0, events=[]))
    _settle(db)
    return {"Authorization": f"Bearer {token}"}


def _rows(db) -> dict:
    found = db.query(SystemSetting).filter(SystemSetting.category == consent.CONSENT_CATEGORY,
                                           SystemSetting.key.in_(CONSENT_KEYS)).all()
    return {row.key: row for row in found}


def _push_expo(client) -> bool:
    response = client.get(DISCOVERY)
    assert response.status_code == 200, response.text
    return response.json()["push"]["expo"]


def _answered_at(body) -> datetime:
    parsed = datetime.fromisoformat(body["answered_at"])
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    return parsed


# --------------------------------------------------------------------------- SaaS


def test_saas_closes_both_routes_before_any_auth_and_always_pushes(make_client, settings_here, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")

    def _no_auth_on_saas():
        pytest.fail("the routes must 404 on SaaS before the request context is resolved")

    client = make_client()
    client.app.dependency_overrides[hybrid.get_request_context_hybrid] = _no_auth_on_saas

    assert client.get(PUSH_CONSENT).status_code == 404
    assert client.put(PUSH_CONSENT, json={"agreed": True}).status_code == 404
    assert _rows(settings_here) == {}
    consent.set_push_consent(settings_here, False)
    assert consent.expo_push_allowed() is True  # SaaS ignores a local answer


# --------------------------------------------------------------------------- the local answer


def test_a_local_install_has_not_answered_and_does_not_push_by_default(make_client, local, operator):
    client = make_client(operator)

    response = client.get(PUSH_CONSENT)

    assert response.status_code == 200
    assert response.json() == NOT_ANSWERED
    assert consent.expo_push_allowed() is False
    assert _push_expo(client) is False


def test_agreeing_is_stored_with_its_time_and_discovery_follows(make_client, local, operator, db_session):
    client = make_client(operator)

    put = client.put(PUSH_CONSENT, json={"agreed": True})

    assert put.status_code == 200, put.text
    assert put.json() == {"agreed": True, "answered": True, "answered_at": T0.isoformat()}
    assert client.get(PUSH_CONSENT).json() == put.json()
    assert _answered_at(put.json()) == T0
    assert consent.expo_push_allowed() is True
    assert _push_expo(client) is True
    rows = _rows(db_session)
    assert rows[consent.KEY_PUSH_VIA_EXPO].value == "true"
    assert rows[consent.KEY_ANSWERED_AT].value == T0.isoformat()


def test_declining_is_answered_but_not_agreed(make_client, local, operator, db_session, monkeypatch):
    client = make_client(operator)
    client.put(PUSH_CONSENT, json={"agreed": True})
    later = datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(consent, "_now", lambda: later)

    put = client.put(PUSH_CONSENT, json={"agreed": False})

    assert put.status_code == 200
    assert put.json() == {"agreed": False, "answered": True, "answered_at": later.isoformat()}
    assert client.get(PUSH_CONSENT).json() == put.json()
    assert consent.expo_push_allowed() is False
    assert _push_expo(client) is False
    # The second answer updates the two rows; it never adds a second pair.
    assert db_session.query(SystemSetting).filter(SystemSetting.category == consent.CONSENT_CATEGORY,
                                                  SystemSetting.key.in_(CONSENT_KEYS)).count() == 2
    assert _rows(db_session)[consent.KEY_PUSH_VIA_EXPO].value == "false"


@pytest.mark.parametrize("body", [{}, {"agreed": "yes"}, {"agreed": 1}, {"agreed": None},
                                  {"agreed": True, "push_via_expo": "true"}])
def test_the_body_is_one_boolean(make_client, local, operator, db_session, body):
    assert make_client(operator).put(PUSH_CONSENT, json=body).status_code == 422
    assert _rows(db_session) == {}


def test_a_failed_write_is_503_rolled_back_and_logged_by_type(make_client, local, operator, db_session,
                                                             monkeypatch, caplog):
    def _dropped(db, agreed):
        raise OperationalError("UPDATE system_settings SET value='true'", {}, Exception("connection lost"))

    monkeypatch.setattr(consent, "set_push_consent", _dropped)

    response = make_client(operator).put(PUSH_CONSENT, json={"agreed": True})

    assert response.status_code == 503
    assert "system_settings" not in response.text
    assert "OperationalError" in caplog.text and "connection lost" not in caplog.text
    assert _rows(db_session) == {}


# --------------------------------------------------------------------------- who may answer


def test_a_plain_member_is_403_and_writes_nothing(make_client, local, ws, db_session):
    member = RequestContext(workspace_id=ws,
                            user=UserContext(id="member@t.test", email="member@t.test", system_role="user"))
    client = make_client(member)

    assert client.get(PUSH_CONSENT).status_code == 403
    assert client.put(PUSH_CONSENT, json={"agreed": True}).status_code == 403
    assert _rows(db_session) == {}


def test_a_phone_bound_to_the_operator_answers_through_the_real_lane(make_client, local, ws, db_session,
                                                                   monkeypatch):
    user_pk, email = _user(db_session)
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", email)
    phone = _paired_phone(db_session, ws, user_pk)
    client = make_client()  # nothing stands in for the request context

    put = client.put(PUSH_CONSENT, json={"agreed": True}, headers=phone)

    assert put.status_code == 200, put.text
    assert put.json()["agreed"] is True
    assert client.get(PUSH_CONSENT, headers=phone).json() == put.json()
    assert _push_expo(client) is True


def test_a_viewers_phone_is_403_through_the_real_lane(make_client, local, ws, db_session):
    user_pk, _email = _user(db_session)
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :uid, 'viewer', TRUE)"),
                       {"ws": str(ws), "uid": user_pk})
    _settle(db_session)
    phone = _paired_phone(db_session, ws, user_pk)
    client = make_client()

    assert client.get(PUSH_CONSENT, headers=phone).status_code == 403
    assert client.put(PUSH_CONSENT, json={"agreed": True}, headers=phone).status_code == 403
    assert _rows(db_session) == {}


# --------------------------------------------------------------------------- the service


def test_a_missing_row_is_created_flushed_and_never_committed(local, db_session, monkeypatch):
    monkeypatch.setattr(db_session, "commit", lambda: pytest.fail("set_push_consent must never commit"))

    state = consent.set_push_consent(db_session, True)

    assert state == consent.ConsentState(agreed=True, answered=True, answered_at=T0.isoformat())
    rows = _rows(db_session)  # a fresh query sees them: the writer flushed
    assert set(rows) == set(CONSENT_KEYS)
    assert (rows[consent.KEY_PUSH_VIA_EXPO].value, rows[consent.KEY_PUSH_VIA_EXPO].value_type) == ("true", "boolean")
    assert rows[consent.KEY_ANSWERED_AT].value == T0.isoformat()
    assert all(row.category == "notifications" and not row.is_sensitive for row in rows.values())
    assert consent.consent_state().to_dict() == {"agreed": True, "answered": True, "answered_at": T0.isoformat()}


@pytest.mark.parametrize("mode", UNREADABLE_MODES)
def test_a_read_failure_is_not_agreed_and_never_shown_as_not_answered(make_client, local, operator,
                                                                     monkeypatch, caplog, mode):
    with settings_unreadable(mode, monkeypatch):
        assert consent.expo_push_allowed() is False
        with pytest.raises(SQLAlchemyError):
            consent.consent_state()
        response = make_client(operator).get(PUSH_CONSENT)

    assert response.status_code == 503
    assert "push consent read failed" in caplog.text


def test_an_answer_overwrites_every_duplicate_row(make_client, local, operator, db_session):
    for _ in range(2):
        db_session.add(SystemSetting(category=consent.CONSENT_CATEGORY, key=consent.KEY_PUSH_VIA_EXPO,
                                     value="true", value_type="boolean"))
    _settle(db_session)

    assert make_client(operator).put(PUSH_CONSENT, json={"agreed": False}).status_code == 200

    stored = db_session.query(SystemSetting.value).filter(SystemSetting.category == consent.CONSENT_CATEGORY,
                                                         SystemSetting.key == consent.KEY_PUSH_VIA_EXPO).all()
    assert [value for (value,) in stored] == ["false", "false"]
    assert consent.expo_push_allowed() is False


def test_only_the_word_true_agrees(local, monkeypatch):
    stored = {}
    monkeypatch.setattr(consent, "get_system_setting",
                        lambda category, key, default=None: stored.get((category, key), default))
    for value, allowed in (("true", True), (" TRUE ", True), ("false", False), ("yes", False), ("1", False)):
        stored[(consent.CONSENT_CATEGORY, consent.KEY_PUSH_VIA_EXPO)] = value
        assert consent.expo_push_allowed() is allowed, value


def test_the_answer_is_not_an_environment_setting():
    surface = (ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8")
    assert "PUSH_VIA_EXPO" not in surface.upper()
    assert not hasattr(config, "PUSH_VIA_EXPO")


# --------------------------------------------------------------------------- wiring


def test_the_routes_are_plain_def_registered_and_in_the_manifest():
    from api import mobile_push
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(mobile_push.get_push_consent)
    assert not inspect.iscoroutinefunction(mobile_push.put_push_consent)
    assert RouterSpec("api.mobile_push") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    routes = {(route["method"], route["path"]) for route in manifest["routes"]}
    assert {("GET", PUSH_CONSENT), ("PUT", PUSH_CONSENT)} <= routes
    assert manifest["route_count"] == len(manifest["routes"])
