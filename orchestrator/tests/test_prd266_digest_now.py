"""PRD-266 US-013 — "Send me today's brief now": ``POST /api/v1/mobile/digest/now``.

``api/mobile_push.py`` (the route), ``services/mobile/digest_now.py`` (who, how often, the
launch) and ``services/mobile/digest.py`` ``post_now`` (the work), on Postgres. Proven here:

* The route: ``202 {"status": "queued"}`` for any member (a viewer too), the job launched for
  the caller's own ``users.id`` in the caller's workspace and nobody else, whatever the body
  says; a person outside the workspace and a caller with no person behind it are 403 and
  launch nothing; no workspace is 404. A plain ``def`` in the committed route manifest.
* How often: one ask per person every ``DIGEST_NOW_MIN_INTERVAL_SECONDS`` (a fake Redis with
  ``SET NX EX`` and ``TTL``): the second is ``429 {"code": "digest_rate_limited",
  "retry_after_seconds": N}`` with ``Retry-After``, a refused ask does not push the wait back,
  another person is not held, and the interval's end lets the person ask again. Redis
  unreachable lets the ask through.
* Can't start: a failed database read, or a job no event loop takes, is
  ``503 {"code": "digest_unavailable"}``, and the mark is given back.
* The work: ``post_now`` makes Auto's Read and posts it through ``deliver_background_message``
  (patched to record) into the caller's Auto thread with origin ``daily_digest`` and their
  local date (their phone's zone, else UTC); no ``mobile_digest_posts`` claim is read or
  written, so the scheduled digest still goes out after it; the ``daily_digest`` switch off
  still posts; a digest that cannot be made posts nothing, never raises, and is logged by its
  class only. ``launch`` runs it as a guarded task.
* Discovery lists ``digest_now`` on the route; the setting is declared.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import math
import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.mobile_push as mobile_push  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.mobile_push import MobileDigestPost  # noqa: E402
from services.chat_messenger import deliver_background_message  # noqa: E402
from services.mobile import digest, digest_now, discovery  # noqa: E402

DIGEST_NOW = "/api/v1/mobile/digest/now"
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
MANIFEST = ORCH / "reports" / "route-manifest.json"
DIGEST_TEXT = "Two tickets wait for review; the Lisbon deck shipped."
# Saturday 24 Oct 2026, 08:00 in Lisbon (WEST, +01:00): the scheduled digest's time there.
T_LISBON_8 = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
SAT = date(2026, 10, 24)
INTERVAL = 300


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeRedis:
    """``SET NX EX``, ``TTL`` and ``DEL`` on a clock the test moves."""

    def __init__(self):
        self.now = 1_000.0
        self.keys: dict = {}
        self.sets: list = []

    def _live(self, key):
        held = self.keys.get(key)
        return held if held is not None and held[1] > self.now else None

    def set(self, key, value, nx=False, ex=None):
        self.sets.append((key, value, nx, ex))
        if nx and self._live(key):
            return None
        self.keys[key] = (value, self.now + ex)
        return True

    def ttl(self, key):
        held = self._live(key)
        return math.ceil(held[1] - self.now) if held else -2

    def delete(self, key):
        self.keys.pop(key, None)


class _Recorder:
    """``generate_digest`` and ``deliver_background_message``, recorded."""

    def __init__(self):
        self.generated: list = []
        self.posts: list = []
        self.generate_raises = None

    async def generate(self, db, workspace_id, period="1d", **_):
        self.generated.append(workspace_id)
        if self.generate_raises is not None:
            raise self.generate_raises
        return {"text": DIGEST_TEXT, "state_hash": "h", "needs_attention_count": 0}

    def deliver(self, db, **kwargs):
        self.posts.append(kwargs)
        return object()


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The digest commits its post and the scheduled one each claim; here a commit reaches the
    last savepoint only, and every seed commits (``_settle``), so nothing escapes the test.
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
    db.commit()


@pytest.fixture(autouse=True)
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "DIGEST_DEFAULT_TIME", "08:00")
    monkeypatch.setattr(config, "DIGEST_NOW_MIN_INTERVAL_SECONDS", INTERVAL)


@pytest.fixture
def redis(monkeypatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(digest_now, "cooldown", digest_now.Cooldown(redis_factory=lambda: fake))
    return fake


@pytest.fixture
def launched(monkeypatch, redis) -> list:
    """Every job the route launches, as (workspace_id, user_id); none runs."""
    jobs: list = []
    monkeypatch.setattr(digest_now, "launch", lambda workspace_id, user_id: jobs.append((workspace_id, user_id)))
    return jobs


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    recorded = _Recorder()
    monkeypatch.setattr(digest, "generate_digest", recorded.generate)
    monkeypatch.setattr(digest, "deliver_background_message", recorded.deliver)
    return recorded


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
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
    """``notification_preferences`` as production has it (prd128's DDL; IF NOT EXISTS)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


# --------------------------------------------------------------------------- seeds


def _person(db) -> SimpleNamespace:
    tag = uuid.uuid4().hex[:12]
    clerk_id = f"user_prd266_now_{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, clerk_user_id, is_active) VALUES (:u, :e, :c, true) RETURNING id"),
        {"u": f"prd266-now-{tag}", "e": f"prd266-now-{tag}@t.test", "c": clerk_id},
    ).scalar()
    _settle(db)
    return SimpleNamespace(pk=pk, clerk_id=clerk_id)


def _owner(db, workspace_id) -> SimpleNamespace:
    person = _person(db)
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": person.pk, "ws": workspace_id})
    _settle(db)
    return person


def _member(db, workspace_id, role="editor") -> SimpleNamespace:
    person = _person(db)
    db.execute(text(
        "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
        "VALUES (CAST(:ws AS uuid), :uid, :role, true)"),
        {"ws": workspace_id, "uid": person.pk, "role": role})
    _settle(db)
    return person


def _phone(db, workspace_id, user_pk, *, zone="Europe/Lisbon") -> None:
    db.add(MobileDevice(
        workspace_id=uuid.UUID(str(workspace_id)), user_id=user_pk, kind="clerk", status="active",
        install_id=f"prd266-now-{uuid.uuid4().hex[:12]}", platform="ios", locale="en-IE",
        timezone=zone, digest_time="08:00", events=[],
    ))
    _settle(db)


def _claims(db, workspace_id) -> list:
    return sorted(
        (row.user_id, row.local_date) for row in
        db.query(MobileDigestPost).filter(MobileDigestPost.workspace_id == uuid.UUID(workspace_id)).all()
    )


def _ctx(person, workspace_id) -> RequestContext:
    user = UserContext(id=person.clerk_id, clerk_user_id=person.clerk_id, system_role="user")
    return RequestContext(workspace_id=uuid.UUID(workspace_id) if workspace_id else None, user=user,
                          auth_type="clerk")


def _client(db, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(mobile_push.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _ask(db, person, workspace_id, **kwargs):
    return _client(db, _ctx(person, workspace_id)).post(DIGEST_NOW, **kwargs)


def _post_now(db, workspace_id, person, at=T_LISBON_8) -> bool:
    return asyncio.run(digest.post_now(uuid.UUID(workspace_id), person.pk, now=at,
                                       session_factory=lambda: _BorrowedSession(db)))


# --------------------------------------------------------------------------- the route: who


@pytest.mark.parametrize("role", ["owner", "admin", "editor", "viewer"])
def test_any_member_gets_202_and_the_job_for_themselves(db_session, ws, launched, role):
    person = _owner(db_session, ws) if role == "owner" else _member(db_session, ws, role)

    response = _ask(db_session, person, ws)

    assert response.status_code == 202, response.text
    assert response.json() == {"status": "queued"}
    assert launched == [(uuid.UUID(ws), person.pk)]


def test_the_job_is_the_callers_whatever_the_body_names(db_session, ws, launched):
    caller = _member(db_session, ws)
    other = _member(db_session, ws)

    response = _ask(db_session, caller, ws, json={"user_id": other.pk, "workspace_id": str(uuid.uuid4())})

    assert response.status_code == 202, response.text
    assert launched == [(uuid.UUID(ws), caller.pk)]


def test_a_paired_device_asks_for_its_own_person(db_session, ws, launched, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    person = _member(db_session, ws)
    email = f"prd266-now-device-{uuid.uuid4().hex[:8]}@t.test"
    user = UserContext(id=email, email=email, system_role="user",
                       raw_claims={"source": "mobile_device", "device_id": str(uuid.uuid4()), "user_id": person.pk})
    ctx = RequestContext(workspace_id=uuid.UUID(ws), user=user, auth_type="device")

    response = _client(db_session, ctx).post(DIGEST_NOW)

    assert response.status_code == 202, response.text
    assert launched == [(uuid.UUID(ws), person.pk)]


def test_a_person_outside_the_workspace_is_403_and_launches_nothing(db_session, ws, seed_workspace, launched, redis):
    elsewhere = seed_workspace()
    _settle(db_session)
    stranger = _member(db_session, elsewhere)

    response = _ask(db_session, stranger, ws)

    assert response.status_code == 403, response.text
    assert launched == [] and redis.sets == []


def test_a_caller_with_no_person_behind_it_is_403(db_session, ws, launched):
    ctx = RequestContext(workspace_id=uuid.UUID(ws), user=UserContext(id="api_key", system_role="admin"),
                         auth_type="api_key")

    response = _client(db_session, ctx).post(DIGEST_NOW)

    assert response.status_code == 403, response.text
    assert launched == []


def test_a_caller_with_no_workspace_is_404(db_session, ws, launched):
    response = _ask(db_session, _member(db_session, ws), None)

    assert response.status_code == 404, response.text
    assert launched == []


def test_the_route_is_a_plain_def_in_the_route_manifest():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(mobile_push.send_digest_now)
    assert RouterSpec("api.mobile_push") in MANIFEST_ROUTERS
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": DIGEST_NOW} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


# --------------------------------------------------------------------------- the route: how often


def test_one_ask_per_person_per_interval(db_session, ws, launched, redis):
    person = _member(db_session, ws)

    first = _ask(db_session, person, ws)
    redis.now += 100
    second = _ask(db_session, person, ws)

    assert first.status_code == 202, first.text
    assert second.status_code == 429, second.text
    assert second.json() == {"code": "digest_rate_limited", "retry_after_seconds": INTERVAL - 100}
    assert second.headers["Retry-After"] == str(INTERVAL - 100)
    assert launched == [(uuid.UUID(ws), person.pk)]
    assert redis.sets[0] == (f"mobile:digest_now:{person.pk}", "1", True, INTERVAL)


def test_a_refused_ask_does_not_push_the_wait_back(db_session, ws, launched, redis):
    person = _member(db_session, ws)
    _ask(db_session, person, ws)
    redis.now += 100
    _ask(db_session, person, ws)

    redis.now += INTERVAL - 100
    again = _ask(db_session, person, ws)

    assert again.status_code == 202, again.text
    assert len(launched) == 2


def test_another_person_is_not_held_back(db_session, ws, launched, redis):
    first, second = _member(db_session, ws), _member(db_session, ws, "viewer")

    assert _ask(db_session, first, ws).status_code == 202
    assert _ask(db_session, second, ws).status_code == 202
    assert launched == [(uuid.UUID(ws), first.pk), (uuid.UUID(ws), second.pk)]


def test_the_interval_is_the_setting(db_session, ws, launched, redis, monkeypatch):
    monkeypatch.setattr(config, "DIGEST_NOW_MIN_INTERVAL_SECONDS", 120)
    person = _member(db_session, ws)

    _ask(db_session, person, ws)
    refused = _ask(db_session, person, ws)

    assert refused.json() == {"code": "digest_rate_limited", "retry_after_seconds": 120}


def test_redis_unreachable_lets_the_ask_through_and_says_so(caplog):
    cooldown = digest_now.Cooldown(redis_factory=lambda: None)

    with caplog.at_level(logging.ERROR, logger=digest_now.__name__):
        assert (cooldown.take(7), cooldown.take(7)) == (None, None)

    assert "FAIL-OPEN" in caplog.text


# --------------------------------------------------------------------------- the route: can't start


def test_a_failed_database_read_is_503_and_takes_no_mark(db_session, ws, launched, redis, monkeypatch):
    def broken(db, workspace_id, user_id):
        raise OperationalError("SELECT", {}, Exception("connection lost"))

    monkeypatch.setattr(digest, "belongs", broken)

    response = _ask(db_session, _member(db_session, ws), ws)

    assert response.status_code == 503, response.text
    assert response.json() == {"code": "digest_unavailable"}
    assert launched == [] and redis.sets == []


def test_a_job_that_cannot_start_is_503_and_gives_the_mark_back(db_session, ws, redis, monkeypatch):
    def no_loop(workspace_id, user_id):
        raise RuntimeError("no event loop")

    monkeypatch.setattr(digest_now, "launch", no_loop)
    person = _member(db_session, ws)

    response = _ask(db_session, person, ws)

    assert response.status_code == 503, response.text
    assert response.json() == {"code": "digest_unavailable"}
    assert redis.keys == {}


# --------------------------------------------------------------------------- the work


def test_post_now_posts_auto_s_read_into_the_callers_auto_thread(db_session, ws, recorder):
    caller = _member(db_session, ws)
    _member(db_session, ws)
    _phone(db_session, ws, caller.pk)

    assert _post_now(db_session, ws, caller) is True

    assert recorder.generated == [uuid.UUID(ws)]
    (post,) = recorder.posts
    assert post["workspace_id"] == uuid.UUID(ws)
    assert post["clerk_user_id"] == caller.clerk_id
    assert post["text"] == DIGEST_TEXT
    assert post["source"] == {"origin": "daily_digest", "local_date": "2026-10-24"}
    assert "chat_id" not in post  # the person's Auto thread


def test_the_post_goes_through_the_chat_messenger():
    """The scheduled digest's own delivery: it raises auto_message, so the phone's push."""
    delivered_by = digest.deliver_background_message
    assert (delivered_by.__module__, delivered_by.__name__) == (
        deliver_background_message.__module__, deliver_background_message.__name__)
    assert digest_now.digest is digest


def test_post_now_writes_no_claim_and_the_scheduled_digest_still_goes_out(db_session, ws, recorder,
                                                                         preferences_table):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person.pk)

    assert _post_now(db_session, ws, person, at=T_LISBON_8 - timedelta(hours=1)) is True
    assert _claims(db_session, ws) == []

    counts = asyncio.run(digest.run_digests(T_LISBON_8, session_factory=lambda: _BorrowedSession(db_session)))

    assert counts["posted"] == 1
    assert len(recorder.posts) == 2
    assert _claims(db_session, ws) == [(person.pk, SAT)]


def test_post_now_posts_even_when_todays_digest_is_claimed(db_session, ws, recorder):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person.pk)
    db_session.add(MobileDigestPost(workspace_id=uuid.UUID(ws), user_id=person.pk, local_date=SAT))
    _settle(db_session)

    assert _post_now(db_session, ws, person) is True
    assert _claims(db_session, ws) == [(person.pk, SAT)]


def test_post_now_posts_with_the_daily_digest_switch_off(db_session, ws, recorder, preferences_table):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person.pk)
    db_session.execute(text(
        "INSERT INTO notification_preferences (workspace_id, user_id, event_type, destination, enabled) "
        "VALUES (CAST(:ws AS uuid), :uid, 'daily_digest', 'push', false)"), {"ws": ws, "uid": person.pk})
    _settle(db_session)

    due = [p for p in digest.due_people(db_session, T_LISBON_8) if str(p.workspace_id) == ws]
    assert due == []  # the schedule skips them
    assert _post_now(db_session, ws, person) is True
    assert len(recorder.posts) == 1


@pytest.mark.parametrize("zone,at,local", [
    ("Europe/Lisbon", "2026-10-24T23:30", "2026-10-25"),  # 00:30 on Sunday in Lisbon
    ("America/New_York", "2026-10-25T02:00", "2026-10-24"),  # 22:00 on Saturday in New York
    (None, "2026-10-24T23:30", "2026-10-24"),  # no phone with a zone: UTC
])
def test_the_local_date_is_the_phones(db_session, ws, recorder, zone, at, local):
    person = _owner(db_session, ws)
    if zone:
        _phone(db_session, ws, person.pk, zone=zone)

    _post_now(db_session, ws, person, at=datetime.fromisoformat(at).replace(tzinfo=timezone.utc))

    assert recorder.posts[0]["source"]["local_date"] == local


def test_a_digest_that_cannot_be_made_posts_nothing_and_logs_its_class_only(db_session, ws, recorder, caplog):
    person = _owner(db_session, ws)
    recorder.generate_raises = RuntimeError("provider said: secret workspace words")

    with caplog.at_level(logging.INFO, logger=digest.__name__):
        assert _post_now(db_session, ws, person) is False

    assert recorder.posts == []
    assert "RuntimeError" in caplog.text
    assert "secret workspace words" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_the_digest_text_is_never_logged(db_session, ws, recorder, caplog):
    person = _owner(db_session, ws)

    with caplog.at_level(logging.DEBUG, logger=digest.__name__):
        _post_now(db_session, ws, person)

    assert DIGEST_TEXT not in caplog.text


def test_launch_runs_post_now_as_a_guarded_task(monkeypatch):
    calls: list = []
    workspace = uuid.uuid4()

    async def fake_post_now(workspace_id, user_id):
        calls.append((workspace_id, user_id))
        return True

    monkeypatch.setattr(digest, "post_now", fake_post_now)

    async def drive():
        digest_now.launch(str(workspace), "7")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(drive())

    assert calls == [(workspace, 7)]


# --------------------------------------------------------------------------- discovery and settings


def test_discovery_lists_digest_now_on_the_route(monkeypatch):
    from config import HOSTED_PUBLIC_API_HOST
    from services.push import consent

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(consent, "get_system_setting", lambda category, key, default=None: default)

    document = discovery.build_discovery("http://192.168.1.20:8000/")

    assert "digest_now" in document["capabilities"]
    assert discovery.CAPABILITY_ROUTES["digest_now"] == ("POST", DIGEST_NOW)
    assert any(route.path == DIGEST_NOW and "POST" in route.methods for route in mobile_push.router.routes)


def test_the_setting_is_declared():
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert "DIGEST_NOW_MIN_INTERVAL_SECONDS" in surface
    assert int(config.DIGEST_NOW_MIN_INTERVAL_SECONDS) >= 1
