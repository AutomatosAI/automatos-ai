"""PRD-266 W2 post-merge review — ``POST /api/v1/mobile/devices`` has limits: each row is a push target.

* Per client IP: the ``mobile_devices_ip`` limit of ``core.security.rate_limiter.DEFAULT_LIMITS``
  (30 registrations an IP in ten minutes), counted by the real ``RateLimitStore`` over a fake
  Redis, as ``/pair`` is. Over it is 429 with ``Retry-After``, before the body is read (a
  malformed body is 429 too, not 422) and before anything is written; another IP still
  registers. Redis unreachable lets the registration through and logs an ERROR naming this
  limiter.
* Per person: at most ``MAX_ACTIVE_DEVICES_PER_PERSON`` (10) active devices in a workspace.
  One more install is ``409 {"code": "device_limit_reached", "message": …}`` and writes nothing;
  a device already active still updates; another person, and the same person in another
  workspace, still register. Revoking one makes room, and a device its person signed out of
  counts again when it comes back.

On Postgres, a Clerk caller standing in for the request context.
"""
from __future__ import annotations

import inspect
import logging
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

DEVICES = "/api/v1/mobile/devices"
OPERATION = "mobile_devices_ip"
LIMIT = 2


def _register_body(install_id: str, /, **overrides) -> dict:
    return {"install_id": install_id, "platform": "ios", "app_version": "1.0.0", "locale": "en-IE",
            "timezone": "Europe/Dublin", "expo_push_token": f"ExponentPushToken[{install_id}]", **overrides}


def _install() -> str:
    return f"prd266-limits-{uuid.uuid4().hex[:12]}"


def _ip(address: str) -> dict:
    return {"X-Forwarded-For": address}


# --------------------------------------------------------------------------- fake Redis


class _FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._ops = []

    def __getattr__(self, name):
        def _queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self

        return _queue

    def execute(self):
        return [getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._ops]


class _FakeRedis:
    """The sorted-set commands ``RateLimitStore`` uses, on a dict."""

    def __init__(self):
        self.z = {}

    def zremrangebyscore(self, key, low, high):
        marks = self.z.get(key, {})
        doomed = [m for m, score in marks.items() if float(low) <= score <= float(high)]
        for member in doomed:
            marks.pop(member)
        return len(doomed)

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zrange(self, key, start, end, withscores=False):
        items = sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])[start:end + 1]
        return items if withscores else [m for m, _ in items]

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key, ttl):
        return True

    def pipeline(self):
        return _FakePipeline(self)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here): a
    route's rollback (a 409) reaches the last savepoint only, never the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def fake_redis(monkeypatch):
    from api import mobile_devices
    from api.widgets.rate_limit import RateLimitStore
    from core.security import rate_limiter

    redis = _FakeRedis()
    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setitem(rate_limiter.DEFAULT_LIMITS, OPERATION, (LIMIT, 600))
    return redis


@pytest.fixture
def no_ip_limit(monkeypatch):
    from api import mobile_devices

    monkeypatch.setattr(mobile_devices, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 30, 29, 600)))


@pytest.fixture
def make_client(db_session):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    def _make(ctx):
        app = FastAPI()
        app.include_router(mobile_devices.router)
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    db_session.commit()
    return workspace_id


def _clerk(db, workspace_id, clerk_id=None):
    """A Clerk caller in ``workspace_id`` (a new person unless ``clerk_id`` names one): ``(ctx, pk)``."""
    from core.auth.dependencies import RequestContext, UserContext

    if clerk_id is None:
        clerk_id = f"user_prd266lim{uuid.uuid4().hex[:12]}"
        db.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) VALUES (:u, :e, true, :c)"),
                   {"u": clerk_id, "e": f"{clerk_id}@t.test", "c": clerk_id})
        db.commit()
    pk = db.execute(text("SELECT id FROM users WHERE clerk_user_id = :c"), {"c": clerk_id}).scalar()
    user = UserContext(id=clerk_id, email=f"{clerk_id}@t.test", clerk_user_id=clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk"), pk


def _rows(db, workspace_id, user_pk, status=None) -> int:
    from core.models.mobile_devices import MobileDevice

    query = db.query(MobileDevice).filter(MobileDevice.workspace_id == workspace_id, MobileDevice.user_id == user_pk)
    if status is not None:
        query = query.filter(MobileDevice.status == status)
    return query.count()


def _fill(client, count: int) -> list:
    """``count`` new installs registered; their device ids."""
    ids = []
    for _ in range(count):
        response = client.post(DEVICES, json=_register_body(_install()))
        assert response.status_code == 200, response.text
        ids.append(response.json()["device_id"])
    return ids


# --------------------------------------------------------------------------- per IP


def test_the_limit_is_thirty_registrations_an_ip_in_ten_minutes_checked_off_the_loop():
    from api import mobile_devices
    from core.security import rate_limiter

    assert rate_limiter.DEFAULT_LIMITS[OPERATION] == (30, 600)
    assert mobile_devices.DEVICES_RATE_LIMIT_OPERATION == OPERATION
    assert not inspect.iscoroutinefunction(mobile_devices.limit_device_registrations_per_ip)  # Redis in the threadpool


def test_an_ip_over_its_registrations_is_429_before_the_body_is_read(db_session, ws, make_client, fake_redis):
    ctx, pk = _clerk(db_session, ws)
    client = make_client(ctx)

    statuses = [client.post(DEVICES, json=_register_body(_install()), headers=_ip("203.0.113.9")).status_code
                for _ in range(LIMIT)]
    limited = client.post(DEVICES, json=_register_body(_install()), headers=_ip("203.0.113.9"))
    malformed = client.post(DEVICES, json={"install_id": "short"}, headers=_ip("203.0.113.9"))

    assert statuses == [200] * LIMIT
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    assert limited.json() == {"detail": "Too many device registrations from this address. Try again later."}
    assert malformed.status_code == 429                         # counted before the body is read
    assert _rows(db_session, ws, pk) == LIMIT                   # the limited call wrote nothing
    assert client.post(DEVICES, json=_register_body(_install()), headers=_ip("198.51.100.4")).status_code == 200
    buckets = sorted(key.split(":mobile_devices:ip:", 1)[1] for key in fake_redis.z if ":mobile_devices:ip:" in key)
    assert buckets == ["198.51.100.4", "203.0.113.9"]


def test_a_redis_outage_lets_the_registration_through_and_names_this_limiter(db_session, ws, make_client,
                                                                            monkeypatch, caplog):
    from api import mobile_devices
    from api.widgets.rate_limit import RateLimitStore

    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: None))
    ctx, _pk = _clerk(db_session, ws)

    with caplog.at_level(logging.ERROR, logger="api.widgets.rate_limit"):
        response = make_client(ctx).post(DEVICES, json=_register_body(_install()), headers=_ip("203.0.113.9"))

    assert response.status_code == 200
    (line,) = [r.getMessage() for r in caplog.records if "FAIL-OPEN" in r.getMessage()]
    assert line.startswith(f"{mobile_devices.DEVICES_RATE_LIMITER_LABEL} FAIL-OPEN")


# --------------------------------------------------------------------------- per person


def test_a_person_holds_at_most_ten_active_devices_in_a_workspace(db_session, ws, seed_workspace, make_client,
                                                                 no_ip_limit):
    from services.mobile.devices import DEVICE_LIMIT_MESSAGE, MAX_ACTIVE_DEVICES_PER_PERSON

    ctx, pk = _clerk(db_session, ws)
    client = make_client(ctx)
    installs = [_install() for _ in range(MAX_ACTIVE_DEVICES_PER_PERSON)]
    for install_id in installs:
        assert client.post(DEVICES, json=_register_body(install_id)).status_code == 200

    one_more = client.post(DEVICES, json=_register_body(_install()))

    assert MAX_ACTIVE_DEVICES_PER_PERSON == 10
    assert (one_more.status_code, one_more.json()) == (
        409, {"detail": {"code": "device_limit_reached", "message": DEVICE_LIMIT_MESSAGE}})
    assert "10 devices" in DEVICE_LIMIT_MESSAGE and "Phones & devices" in DEVICE_LIMIT_MESSAGE
    assert _rows(db_session, ws, pk) == MAX_ACTIVE_DEVICES_PER_PERSON
    assert client.post(DEVICES, json=_register_body(installs[0], app_version="1.1.0")).status_code == 200
    other, _ = _clerk(db_session, ws)
    assert make_client(other).post(DEVICES, json=_register_body(_install())).status_code == 200
    elsewhere, _ = _clerk(db_session, uuid.UUID(seed_workspace()), clerk_id=ctx.user.clerk_user_id)
    db_session.commit()
    assert make_client(elsewhere).post(DEVICES, json=_register_body(_install())).status_code == 200


def test_revoking_one_makes_room_and_a_signed_out_device_counts_again(db_session, ws, make_client, no_ip_limit):
    from services.mobile.devices import MAX_ACTIVE_DEVICES_PER_PERSON

    ctx, pk = _clerk(db_session, ws)
    client = make_client(ctx)
    signed_out = _install()
    device_id = client.post(DEVICES, json=_register_body(signed_out)).json()["device_id"]
    _fill(client, MAX_ACTIVE_DEVICES_PER_PERSON - 1)

    assert client.delete(f"{DEVICES}/{device_id}").status_code == 200   # signing out of one
    assert client.post(DEVICES, json=_register_body(_install())).status_code == 200
    back = client.post(DEVICES, json=_register_body(signed_out))

    assert back.status_code == 409 and back.json()["detail"]["code"] == "device_limit_reached"
    assert _rows(db_session, ws, pk, status="active") == MAX_ACTIVE_DEVICES_PER_PERSON
    assert _rows(db_session, ws, pk, status="revoked") == 1
