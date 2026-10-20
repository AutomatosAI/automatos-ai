"""PRD-261 C-05h — the local edition's pull of the revocation feed, its sync status and Check now.

The real sync and install routers run on a mini FastAPI app over in-memory SQLite, in the local
edition, as its anonymous owner (a super admin). The pull's client is an ``httpx.MockTransport`` that
records each request, built with ``follow_redirects=True``: a 3xx answered once proves the pull
turns redirects off itself. Pins (plan §2.6 C-05h): rows by hosted id with ``origin`` 'feed', lifted
only by ``liftedAt``, absent rows kept, local rows never written; every failure (500, timeout, bad
JSON, schema, a bad row, oversize, 3xx, encoded, database) changes no row and records
``last_attempt_at``/``last_error``; no ``Authorization``, ``Cookie`` or ``X-Workspace-ID`` sent;
``http://`` only to loopback; ``revokedKeys`` stored; status (stale after 7 days) and Check now
admin-only, 404 hosted, 409 without a URL; a pulled disable makes the local ``active`` the base.
"""
from __future__ import annotations

import gzip
import inspect
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_theme_revocation_sync as sync_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeChoice, ThemeInstallation, ThemeLook, ThemeOptIn, ThemePackage, ThemeRevocation, ThemeRevocationSync,
    ThemeWorkspaceSettings,
)
from services.themes import revocation_feed, revocation_pull  # noqa: E402
from services.themes.revocation_feed import as_utc  # noqa: E402

WS = uuid.uuid4()
OPERATOR_PK = 1
SYNC = "/api/admin/themes/revocations/sync"
PULL = SYNC + "/pull"
FEED_URL = "https://feed.example.test/api/themes/revocations"
GENERATED = "2026-10-21T09:00:00+00:00"
LIFTED = "2026-10-21T08:00:00+00:00"
KEY_A, KEY_B = "ed25519:" + "a" * 16, "ed25519:" + "b" * 16
LONG_AGO = datetime(2020, 1, 1, tzinfo=timezone.utc)
DROP = object()  # a field the wire helpers leave out
COLUMNS = ("digest", "theme_id", "version_range", "level", "reason", "created_by", "created_at", "lifted_at", "origin")
TABLES = [m.__table__ for m in (ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn,
                                ThemeWorkspaceSettings, ThemeRevocation, ThemeRevocationSync)]


def _operator():
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK})
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


def _clerk(role):
    user = UserContext(id="clerk-8", clerk_user_id="clerk-8", system_role=role,
                       raw_claims={"sub": "clerk-8", "iat": int(time.time())})
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _row(**fields):
    row = {"id": str(uuid.uuid4()), "digest": "ab" * 32, "level": "disable", "reason": "leaks data",
           "createdAt": "2026-10-20T10:00:00+00:00", "liftedAt": None, **fields}
    return {key: value for key, value in row.items() if value is not DROP}


def _feed(rows, keys=(), **fields):
    feed = {"schema": 1, "generatedAt": GENERATED, "revocations": rows, "revokedKeys": list(keys), **fields}
    return json.dumps(feed).encode("utf-8")


def _answer(body, status=200, headers=None):
    return lambda request: httpx.Response(status, content=body, headers=headers or {})


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, email) VALUES (1, 'operator@local')"))
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, FEED_URL, raising=False)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: None)
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_operator(), seen=[], answer=_answer(_feed([])))

    def handle(request):
        state.seen.append(request)
        return state.answer(request)

    state.factory = lambda: httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True)
    app = FastAPI()
    app.include_router(sync_api.router)
    app.include_router(install_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[sync_api.get_pull_client_factory] = lambda: state.factory
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _pull(api, **headers):
    return api.client.post(PULL, headers=headers)


def _rows(db):
    db.expire_all()
    return {str(row.id): tuple(getattr(row, name) for name in COLUMNS) for row in db.query(ThemeRevocation).all()}


def _sync(db):
    db.expire_all()
    return db.get(ThemeRevocationSync, 1)


def _held(db, wire):
    db.expire_all()
    return db.get(ThemeRevocation, uuid.UUID(wire["id"]))


# --------------------------------------------------------------------------- feed to rows


def test_the_first_pull_inserts_the_feed_rows_under_their_hosted_ids(api):
    by_digest = _row()
    by_theme = _row(digest=DROP, themeId="acme/ink", versionRange="^1.0.0", level="warn", reason="slow")
    api.answer = _answer(_feed([by_digest, by_theme], keys=[KEY_A]))
    resp = _pull(api)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["rows"], body["lastError"], body["feedGeneratedAt"]) == (2, None, GENERATED)
    assert body["revokedKeys"] == [KEY_A]
    assert body["enabled"] is True and body["stale"] is False
    assert body["lastSuccessAt"] == body["lastAttemptAt"] is not None
    for wire in (by_digest, by_theme):
        row = _held(api.db, wire)
        assert (row.origin, row.lifted_at, row.created_by) == ("feed", None, None)
        assert (row.digest, row.theme_id, row.version_range, row.level, row.reason) == (
            wire.get("digest"), wire.get("themeId"), wire.get("versionRange"), wire["level"], wire["reason"],
        )
        assert as_utc(row.created_at) == datetime.fromisoformat(wire["createdAt"])
    sync = _sync(api.db)
    assert (sync.revoked_key_ids, sync.last_error) == ([KEY_A], None)
    assert as_utc(sync.feed_generated_at) == datetime.fromisoformat(GENERATED)


def test_a_later_lifted_at_lifts_and_an_absent_row_stays_live(api):
    first, second = _row(), _row(digest="cd" * 32)
    api.answer = _answer(_feed([first, second]))
    assert _pull(api).status_code == 200
    api.answer = _answer(_feed([dict(first, liftedAt=LIFTED)]))
    assert _pull(api).status_code == 200
    assert as_utc(_held(api.db, first).lifted_at) == datetime.fromisoformat(LIFTED)
    assert _held(api.db, second).lifted_at is None
    api.answer = _answer(_feed([first]))  # liftedAt null again: a lift is never undone by the feed
    assert _pull(api).status_code == 200
    assert _held(api.db, first).lifted_at is not None and len(_rows(api.db)) == 2


def test_local_rows_are_never_written(api):
    mine = ThemeRevocation(digest="ef" * 32, level="warn", reason="ours")
    clash = ThemeRevocation(digest="ab" * 32, level="warn", reason="ours too")
    api.db.add_all([mine, clash])
    api.db.commit()
    before = _rows(api.db)
    fresh = _row(digest="12" * 32)
    api.answer = _answer(_feed([_row(id=str(clash.id), reason="theirs", liftedAt=LIFTED), fresh]))
    resp = _pull(api)
    assert resp.status_code == 200 and resp.json()["rows"] == 1
    after = _rows(api.db)
    assert {key: after[key] for key in before} == before
    assert after[fresh["id"]][COLUMNS.index("origin")] == "feed" and len(after) == 3


def _raises(error):
    def answer(request):
        raise error("feed down", request=request)
    return answer


BASELINE = _row()
LIFT_BASELINE = dict(BASELINE, liftedAt=LIFTED)
OVER_ONE_KB = 2048
FAILURES = {
    "a 500": (_answer(b"oops", 500), revocation_pull.BAD_STATUS.format(status=500)),
    "a timeout": (_raises(httpx.ReadTimeout), revocation_pull.TIMED_OUT),
    "unreachable": (_raises(httpx.ConnectError), revocation_pull.UNREACHABLE.format(error="ConnectError")),
    "invalid JSON": (_answer(b"{not json"), revocation_feed.NOT_JSON),
    "a schema error": (_answer(_feed([LIFT_BASELINE], schema=2)), revocation_feed.NOT_SCHEMA),
    "a bad row": (_answer(_feed([LIFT_BASELINE, _row(digest="AB" * 32)])), revocation_feed.ROW_DIGEST.format(number=2)),
    "a declared oversize": (_answer(b" " * OVER_ONE_KB), revocation_feed.TOO_LARGE.format(kb=1)),
    # a list body is streamed (chunked, no Content-Length), never pre-read
    "a streamed oversize": (_answer([b" " * 700, b" " * 700]), revocation_feed.TOO_LARGE.format(kb=1)),
    "a 302": (_answer(b"", 302, {"location": "https://elsewhere.example.test/feed"}),
              revocation_pull.REDIRECTED.format(status=302)),
    "a 301 to http": (_answer(b"", 301, {"location": "http://feed.example.test/feed"}),
                      revocation_pull.REDIRECTED.format(status=301)),
    "an encoded body": (_answer(gzip.compress(_feed([LIFT_BASELINE])), 200, {"content-encoding": "gzip"}),
                        revocation_pull.ENCODED),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_failed_pull_changes_no_revocation_and_records_the_error(api, monkeypatch, name):
    answer, message = FAILURES[name]
    api.answer = _answer(_feed([BASELINE], keys=[KEY_A]))
    assert _pull(api).status_code == 200
    sync = _sync(api.db)
    success, generated = sync.last_success_at, sync.feed_generated_at
    sync.last_attempt_at = LONG_AGO
    api.db.commit()
    before = _rows(api.db)
    monkeypatch.setattr(config, revocation_feed.FEED_MAX_KB_SETTING, 1, raising=False)
    api.answer, api.seen[:] = answer, []
    resp = _pull(api)
    assert (resp.status_code, resp.json()["detail"]) == (502, message)
    assert _rows(api.db) == before and len(api.seen) == 1  # one request: no redirect followed
    sync = _sync(api.db)
    assert sync.last_error == message and as_utc(sync.last_attempt_at) > LONG_AGO
    assert (sync.last_success_at, sync.feed_generated_at, sync.revoked_key_ids) == (success, generated, [KEY_A])


def test_a_database_error_rolls_the_whole_pull_back(api, monkeypatch):
    real, calls = revocation_pull._upsert, []

    def second_write_fails(db, item):
        calls.append(item.id)
        if len(calls) == 2:
            raise SQLAlchemyError("disk full")
        return real(db, item)

    monkeypatch.setattr(revocation_pull, "_upsert", second_write_fails)
    api.answer = _answer(_feed([_row(), _row(digest="cd" * 32)], keys=[KEY_A]))
    resp = _pull(api)
    assert (resp.status_code, resp.json()["detail"]) == (502, revocation_pull.NOT_SAVED)
    sync = _sync(api.db)
    assert _rows(api.db) == {} and (sync.last_success_at, sync.revoked_key_ids) == (None, [])
    assert sync.last_error == revocation_pull.NOT_SAVED and sync.last_attempt_at is not None


# --------------------------------------------------------------------------- what the pull sends, and where to


def test_no_credentials_cookie_or_workspace_header_reaches_the_feed(api):
    api.answer = _answer(_feed([]), 200, {"set-cookie": "feed=1; Path=/"})
    for _ in range(2):
        resp = _pull(api, Authorization="Bearer test", Cookie="session=test", **{"X-Workspace-ID": str(WS)})
        assert resp.status_code == 200
    for request in api.seen:
        assert (request.method, str(request.url)) == ("GET", FEED_URL)
        assert not {"authorization", "cookie", "x-workspace-id"} & {name.lower() for name in request.headers}
        assert request.headers["user-agent"] == revocation_pull.USER_AGENT
        assert request.headers["accept-encoding"] == "identity"


def test_the_pull_client_is_its_own_and_never_follows_redirects():
    with revocation_pull._client() as client:
        assert client.follow_redirects is False and client.auth is None and not client.cookies
        assert client.timeout.read == revocation_pull.PULL_TIMEOUT_SECONDS


ALLOWED_URLS = ("https://feed.example.test/feed", "http://127.0.0.1:8000/feed", "http://localhost/feed",
                "http://[::1]:9000/feed")
REFUSED_URLS = ("http://feed.example.test/feed", "ftp://feed.example.test/feed", "https://someone@feed.example.test/f",
                "https://a:b@feed.example.test/f", "http://localhost.example.test/f", "https:///f", "file:///etc/hosts",
                "http://[::1/f", "")


@pytest.mark.parametrize("url", ALLOWED_URLS + REFUSED_URLS)
def test_only_https_or_http_to_loopback_without_credentials_is_fetched(url):
    if url in ALLOWED_URLS:
        assert revocation_pull.checked_url(url) == url
        return
    with pytest.raises(revocation_pull.PullFailed):
        revocation_pull.checked_url(url)


def test_an_http_feed_url_is_refused_before_any_request(api, monkeypatch):
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, "http://feed.example.test/feed", raising=False)
    resp = _pull(api)
    assert (resp.status_code, resp.json()["detail"]) == (502, revocation_pull.URL_NOT_HTTPS)
    assert api.seen == [] and _sync(api.db).last_error == revocation_pull.URL_NOT_HTTPS


def test_revoked_keys_are_stored_for_the_signature_check(api):
    api.answer = _answer(_feed([], keys=[KEY_B, KEY_A, KEY_B]))
    assert _pull(api).status_code == 200
    assert _sync(api.db).revoked_key_ids == [KEY_A, KEY_B] == revocation_pull.stored_revoked_keys(api.db)
    api.answer = _answer(_feed([], keys=[]))
    assert _pull(api).status_code == 200
    assert revocation_pull.stored_revoked_keys(api.db) == []


# --------------------------------------------------------------------------- status, gates, and end to end


def test_the_status_before_any_pull_and_when_stale(api):
    assert api.client.get(SYNC).json() == {
        "enabled": True, "lastAttemptAt": None, "lastSuccessAt": None, "lastError": None,
        "feedGeneratedAt": None, "rows": 0, "revokedKeys": [], "stale": True,
    }
    now = datetime.now(timezone.utc)
    api.db.add(ThemeRevocationSync(id=1, last_success_at=now - timedelta(days=8), revoked_key_ids=[]))
    api.db.commit()
    assert api.client.get(SYNC).json()["stale"] is True
    _sync(api.db).last_success_at = now - timedelta(days=1)
    api.db.commit()
    assert api.client.get(SYNC).json()["stale"] is False
    assert revocation_pull.is_stale(now - timedelta(days=6, hours=23), now) is False
    assert revocation_pull.is_stale(now - timedelta(days=7, seconds=1), now) is True


def test_both_routes_are_admin_only(api):
    api.ctx = _clerk("user")
    assert api.client.get(SYNC).status_code == 403 and _pull(api).status_code == 403
    assert api.seen == [] and _sync(api.db) is None


def test_the_hosted_edition_has_no_sync_and_a_non_admin_is_still_refused(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    api.ctx = _clerk("admin")
    assert api.client.get(SYNC).status_code == 404 and _pull(api).status_code == 404
    api.ctx = _clerk("user")
    assert api.client.get(SYNC).status_code == 403 and _pull(api).status_code == 403
    assert api.seen == []


def test_check_now_with_an_empty_feed_url_is_409(api, monkeypatch):
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, "", raising=False)
    resp = _pull(api)
    assert (resp.status_code, resp.json()["detail"]) == (409, revocation_pull.PULL_OFF)
    assert api.client.get(SYNC).json()["enabled"] is False and api.seen == []


def test_the_routes_are_plain_defs_that_check_the_admin_in_the_handler():
    seen = set()
    for route in sync_api.router.routes:
        assert not inspect.iscoroutinefunction(route.endpoint)
        assert "_require_admin(ctx" in inspect.getsource(route.endpoint)
        seen |= {(method, route.path) for method in route.methods}
    assert seen == {("GET", SYNC), ("POST", PULL)}


def test_a_pulled_disable_makes_the_local_active_answer_the_base(api):
    manifest = {"id": "acme/ink", "name": "Ink", "version": "1.0.0", "base": "classic", "htmlClass": "ink"}
    files = {"dist/entry.js": {"type": "text/javascript", "size": 1, "sha256": "ab" * 32},
             "dist/theme.css": {"type": "text/css", "size": 1, "sha256": "cd" * 32}}
    package = ThemePackage(id=uuid.uuid4(), theme_id="acme/ink", version="1.0.0", owner_type="workspace",
                           workspace_id=WS, digest="ab" * 32, manifest=manifest, file_manifest=files,
                           storage_prefix="themes/acme/ink/1.0.0/")
    install = ThemeInstallation(workspace_id=WS, theme_id="acme/ink", package_id=package.id)
    api.db.add_all([package, install])
    api.db.commit()
    assert api.client.put("/api/themes/choice", json={"choice": str(install.id)}).status_code == 200
    assert api.client.get("/api/themes/active").json()["theme"]["installId"] == str(install.id)
    api.answer = _answer(_feed([_row(digest=DROP, themeId="acme/ink", reason="Automatos revoked it")]))
    assert _pull(api).status_code == 200
    body = api.client.get("/api/themes/active").json()
    assert (body["choice"], body["theme"]) == ("classic", None)
    assert body["revoked"] == {"installId": str(install.id), "level": "disable", "reason": "Automatos revoked it"}
