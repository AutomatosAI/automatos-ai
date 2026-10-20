"""PRD-266 US-003b — the pairing routes: ``api/mobile_devices.py``.

* ``POST /api/v1/mobile/pairing-codes`` — a workspace owner/admin (the local operator
  passes as super_admin) mints ``{code, expires_at, qr_payload}``; a plain member is 403;
  404 in saas.
* ``POST /api/v1/mobile/pair`` — public (AGENTS.md exception 1); 404 unless the local
  edition runs ``LOCAL_REMOTE_ACCESS=paired``; a wrong code is ``401 {"code":
  "invalid_pairing_code"}`` and nothing more; the body is validated.
* Rate limit: the real ``RateLimitStore`` over a fake Redis, with the ``mobile_pair_ip``
  limit of ``core.security.rate_limiter.DEFAULT_LIMITS``. Each client IP has its own bucket,
  checked before the code; over it is 429 with ``Retry-After``. There is no instance bucket:
  many IPs guessing wrong never block the person pairing, and their guesses never spend the
  pending code. Redis unreachable lets the attempt through and logs an ERROR naming this
  limiter.
* A database error is 409 (a unique index) or 503 (anything else) and is logged by its type
  only, never its text (which carries the bound hashes).
* Both routes and the limiter are plain ``def``, registered by ``RouterSpec`` and listed in
  the route manifest; the answers that carry a secret are ``no-store``.
"""
from __future__ import annotations

import inspect
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
PAIRING_CODES = "/api/v1/mobile/pairing-codes"
PAIR = "/api/v1/mobile/pair"
LIMIT_OPERATION = "mobile_pair_ip"
LIMIT = 3
WRONG_CODE = "ABCD-EFGH"
PERSON_IP = "198.51.100.7"


def _body(code: str, **overrides) -> dict:
    return {"code": code, "name": "Studio test phone", "platform": "ios", "app_version": "1.0.0",
            "install_id": "prd266-route-install-01", **overrides}


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
def fake_redis(monkeypatch):
    from api import mobile_devices
    from api.widgets.rate_limit import RateLimitStore
    from core.security import rate_limiter

    redis = _FakeRedis()
    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setitem(rate_limiter.DEFAULT_LIMITS, LIMIT_OPERATION, (LIMIT, 600))
    return redis


@pytest.fixture
def paired_mode(monkeypatch):
    from config import HOSTED_PUBLIC_API_HOST, config

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(config, "FRONTEND_URL", "https://studio.prd266.test")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    return config


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-route-{tag}", "e": f"prd266-route-{tag}@t.test"},
    ).scalar()


@pytest.fixture
def operator(db_session, seed_workspace):
    """The local operator: super_admin, its users PK on raw_claims (hybrid's local lane)."""
    from core.auth.dependencies import RequestContext, UserContext

    user_pk = _seed_user(db_session)
    user = UserContext(id="operator@t.test", email="operator@t.test", system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": user_pk})
    return RequestContext(workspace_id=uuid.UUID(seed_workspace()), user=user)


@pytest.fixture
def make_client(db_session, fake_redis):
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


def _ip(address: str) -> dict:
    return {"X-Forwarded-For": address}


def _mint(make_client, operator) -> str:
    response = make_client(operator).post(PAIRING_CODES)
    assert response.status_code == 200, response.text
    return response.json()["code"]


def _limit_keys(fake_redis) -> list:
    return sorted(key for key in fake_redis.z if ":mobile_pair:ip:" in key)


def _bucket(fake_redis_key: str) -> str:
    return fake_redis_key.split(":mobile_pair:ip:", 1)[1]


# --------------------------------------------------------------------------- where the routes answer


def test_both_routes_are_404_in_saas_and_never_touch_the_limiter(make_client, operator, paired_mode, fake_redis,
                                                                 monkeypatch):
    monkeypatch.setattr(paired_mode, "AUTH_EDITION", "saas")
    client = make_client(operator)

    assert client.post(PAIRING_CODES).status_code == 404
    assert make_client().post(PAIRING_CODES).status_code == 404
    assert client.post(PAIR, json=_body(WRONG_CODE)).status_code == 404
    assert fake_redis.z == {}


@pytest.mark.parametrize("mode", ["off", "open"])
def test_pair_is_404_unless_the_install_runs_paired_mode(make_client, paired_mode, fake_redis, monkeypatch, mode):
    monkeypatch.setattr(paired_mode, "LOCAL_REMOTE_ACCESS", mode)

    response = make_client().post(PAIR, json=_body(WRONG_CODE))

    assert response.status_code == 404
    assert fake_redis.z == {}


# --------------------------------------------------------------------------- minting a code


def test_an_admin_mints_a_code_shown_once_and_never_cached(make_client, operator, paired_mode, db_session):
    from core.models.mobile_devices import MobileDevice

    response = make_client(operator).post(PAIRING_CODES)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"code", "expires_at", "qr_payload"}
    assert body["qr_payload"].startswith("https://studio.prd266.test/go/pair?api=http%3A%2F%2Ftestserver&code=")
    assert response.headers["Cache-Control"] == "no-store"
    rows = db_session.query(MobileDevice).filter(MobileDevice.workspace_id == operator.workspace_id).all()
    assert [(row.status, row.user_id) for row in rows] == [("pending", operator.user.raw_claims["user_id"])]


def test_a_code_can_be_minted_while_pair_is_closed_but_only_on_a_local_install(make_client, operator, paired_mode,
                                                                                monkeypatch):
    monkeypatch.setattr(paired_mode, "LOCAL_REMOTE_ACCESS", "off")

    assert make_client(operator).post(PAIRING_CODES).status_code == 200


def test_a_member_who_is_not_an_admin_is_refused(make_client, operator, paired_mode, db_session):
    from core.auth.dependencies import RequestContext, UserContext
    from core.models.mobile_devices import MobileDevice

    member = RequestContext(workspace_id=operator.workspace_id,
                            user=UserContext(id="member@t.test", email="member@t.test", system_role="user"))

    response = make_client(member).post(PAIRING_CODES)

    assert response.status_code == 403
    assert db_session.query(MobileDevice).filter(MobileDevice.workspace_id == operator.workspace_id).count() == 0


# --------------------------------------------------------------------------- pairing


def test_a_valid_code_pairs_once_and_answers_the_token_once(make_client, operator, paired_mode):
    code = _mint(make_client, operator)
    client = make_client()

    first = client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP))
    again = client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP))

    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"device_id", "workspace_id", "token", "rotate_after"}
    assert body["token"].startswith("amd_") and body["workspace_id"] == str(operator.workspace_id)
    assert first.headers["Cache-Control"] == "no-store"
    assert again.status_code == 401


def test_a_wrong_code_is_401_invalid_pairing_code_and_nothing_more(make_client, paired_mode):
    response = make_client().post(PAIR, json=_body(WRONG_CODE), headers=_ip(PERSON_IP))

    assert response.status_code == 401
    assert response.json() == {"detail": {"code": "invalid_pairing_code"}}


@pytest.mark.parametrize("overrides", [
    {"platform": "windows"},
    {"install_id": "short"},
    {"install_id": "has spaces in it"},
    {"install_id": "x" * 129},
    {"name": "n" * 121},
    {"code": ""},
])
def test_a_malformed_body_is_422(make_client, paired_mode, overrides):
    assert make_client().post(PAIR, json=_body(WRONG_CODE, **overrides)).status_code == 422


@pytest.mark.parametrize("platform", ["ios", "android", "web"])
def test_every_platform_is_accepted(make_client, operator, paired_mode, platform):
    code = _mint(make_client, operator)

    assert make_client().post(PAIR, json=_body(code, platform=platform)).status_code == 200


# --------------------------------------------------------------------------- the rate limit is per IP only


def test_an_ip_over_its_attempts_is_429_with_retry_after_before_the_code_is_read(make_client, operator,
                                                                                  paired_mode, fake_redis):
    code = _mint(make_client, operator)
    client = make_client()

    statuses = [client.post(PAIR, json=_body(WRONG_CODE), headers=_ip("203.0.113.5")).status_code
                for _ in range(LIMIT)]
    limited = client.post(PAIR, json=_body(code), headers=_ip("203.0.113.5"))

    assert statuses == [401] * LIMIT
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    assert [_bucket(key) for key in _limit_keys(fake_redis)] == ["203.0.113.5"]
    # The limited attempt carried the right code and still spent nothing.
    assert client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP)).status_code == 200


def test_many_ips_guessing_wrong_never_block_the_person_pairing(make_client, operator, paired_mode, fake_redis):
    code = _mint(make_client, operator)
    client = make_client()

    for n in range(1, 21):
        for _ in range(LIMIT):
            assert client.post(PAIR, json=_body(WRONG_CODE), headers=_ip(f"192.0.2.{n}")).status_code == 401

    assert client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP)).status_code == 200
    keys = _limit_keys(fake_redis)
    assert len(keys) == 21 and len(fake_redis.z) == 21
    assert not any("instance" in key for key in fake_redis.z)


def test_the_pending_code_still_pairs_after_another_callers_wrong_guesses(make_client, operator, paired_mode):
    code = _mint(make_client, operator)
    client = make_client()

    for _ in range(LIMIT):
        assert client.post(PAIR, json=_body(WRONG_CODE), headers=_ip("198.51.100.66")).status_code == 401
    assert client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP)).status_code == 200


def test_an_ipv6_caller_is_counted_per_64(make_client, paired_mode, fake_redis):
    client = make_client()

    for suffix in range(LIMIT):
        client.post(PAIR, json=_body(WRONG_CODE), headers=_ip(f"2001:db8:9:9::{suffix + 1}"))
    response = client.post(PAIR, json=_body(WRONG_CODE), headers=_ip("2001:db8:9:9:ffff::1"))

    assert response.status_code == 429
    assert [_bucket(key) for key in _limit_keys(fake_redis)] == ["2001:db8:9:9::/64"]


def test_the_limit_is_a_named_operation_with_no_instance_bucket():
    from core.security import rate_limiter

    assert LIMIT_OPERATION in rate_limiter.DEFAULT_LIMITS
    assert not [name for name in rate_limiter.DEFAULT_LIMITS if name.startswith("mobile_pair") and name != LIMIT_OPERATION]


# --------------------------------------------------------------------------- wiring


def test_the_routes_are_plain_def_mounted_and_listed():
    from api import mobile_devices
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(mobile_devices.create_pairing_code)
    assert not inspect.iscoroutinefunction(mobile_devices.pair)
    assert not inspect.iscoroutinefunction(mobile_devices.limit_pair_attempts_per_ip)
    assert RouterSpec("api.mobile_devices") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    for path in (PAIRING_CODES, PAIR):
        assert {"method": "POST", "path": path} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_settings_are_named_in_the_config_surface_with_their_defaults():
    from config import config

    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))
    for name in ("LOCAL_REMOTE_ACCESS", "MOBILE_TOKEN_ROTATE_DAYS", "MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS"):
        assert name in surface["settings"]
    assert config.LOCAL_REMOTE_ACCESS == "off"
    assert config.MOBILE_TOKEN_ROTATE_DAYS >= 1
    assert config.MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS == 600


def test_the_module_names_its_public_route_exception():
    from api import mobile_devices

    doc = mobile_devices.__doc__
    assert "PUBLIC" in doc and "AGENTS.md exception 1" in doc and "/pair" in doc


def test_no_log_line_carries_a_code_a_token_or_a_hash(make_client, operator, paired_mode, caplog):
    import logging

    from core.auth.pairing import hash_secret

    with caplog.at_level(logging.DEBUG):
        code = _mint(make_client, operator)
        client = make_client()
        client.post(PAIR, json=_body(WRONG_CODE), headers=_ip(PERSON_IP))
        token = client.post(PAIR, json=_body(code), headers=_ip(PERSON_IP)).json()["token"]

    ours = [r.getMessage() for r in caplog.records
            if r.name.startswith(("api.mobile_devices", "services.mobile", "core.security"))]
    for secret in (code, token, hash_secret(code), hash_secret(token)):
        assert not any(secret in line for line in ours)


def test_a_redis_outage_lets_the_attempt_through_and_names_this_limiter(make_client, paired_mode, monkeypatch,
                                                                      caplog):
    import logging

    from api import mobile_devices
    from api.widgets.rate_limit import RateLimitStore

    monkeypatch.setattr(mobile_devices, "_rate_store", RateLimitStore(redis_factory=lambda: None))

    with caplog.at_level(logging.ERROR, logger="api.widgets.rate_limit"):
        response = make_client().post(PAIR, json=_body(WRONG_CODE), headers=_ip(PERSON_IP))

    assert response.status_code == 401
    (line,) = [r.getMessage() for r in caplog.records if "FAIL-OPEN" in r.getMessage()]
    assert line.startswith(f"{mobile_devices.RATE_LIMITER_LABEL} FAIL-OPEN") and WRONG_CODE not in line


@pytest.mark.parametrize("error, status", [("integrity", 409), ("operational", 503)])
def test_a_database_error_is_answered_by_status_and_logged_by_type_only(make_client, paired_mode, monkeypatch,
                                                                        caplog, error, status):
    import logging

    from sqlalchemy.exc import IntegrityError, OperationalError

    from api import mobile_devices

    leaked = "pairing_code_hash=" + "f" * 64
    raised = {"integrity": IntegrityError, "operational": OperationalError}[error]

    def _failing(*args, **kwargs):
        raise raised("UPDATE mobile_devices SET ...", {"token_hash": leaked}, Exception(leaked))

    monkeypatch.setattr(mobile_devices.pairing, "pair_device", _failing)

    with caplog.at_level(logging.DEBUG, logger="api.mobile_devices"):
        response = make_client().post(PAIR, json=_body(WRONG_CODE), headers=_ip(PERSON_IP))

    assert response.status_code == status and leaked not in response.text
    assert caplog.records and not any(leaked in r.getMessage() for r in caplog.records)
