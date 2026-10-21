"""PRD-261 C-05e — the admin's revocations: list, create (warn or disable; digest or theme id + range), lift.

The real router (``api/admin_theme_revocations.py``) runs on a mini FastAPI app over in-memory
SQLite beside the REAL install router (``api/themes_install.py``), so a revocation written here
is read back the way a member's next load reads it (``GET /api/themes/active``). The REAL
``caller_is_admin``, step-up and ``resolve_user_pk`` run; only the caller's workspace role is
stubbed for the install router's gate. C3-wire registers the router in ``main.py``. Pins (plan
§2.6 C-05e):

* list, create and lift; ``caller_is_admin`` first and a fresh step-up after it on create and
  lift; a non-admin gets the plain 403 on every route (never a reverification prompt); the local
  edition's operator revokes and lifts with no step-up;
* the target is checked before anything is written: a digest is 64 lower-case hex, a theme id
  ``manifest.ID_PATTERN``, a range ``manifest.RANGE_PATTERN`` (an unreadable one is refused,
  never stored); the level is warn or disable; the reason 1..2000 characters of plain text;
* a digest disable makes ``active`` answer the base; a warn by ``^1.0.0`` keeps the theme with
  the notice; lift restores it;
* lifting an ``origin='feed'`` row is 409 and changes nothing; a duplicate live revocation
  answers the existing row with 200, and so does a create that loses the race to an identical
  one (``uq_theme_revocations_live_local`` refuses its write); a create writes only
  ``origin='local'``.
"""
from __future__ import annotations

import inspect
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

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_theme_revocations as revocations_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    THEME_REVOCATION_ORIGIN_FEED, THEME_REVOCATION_ORIGIN_LOCAL, ThemeChoice, ThemeInstallation, ThemeLook,
    ThemeOptIn, ThemePackage, ThemeRevocation, ThemeRevocationSync, ThemeWorkspaceSettings,
)
from services.themes import revocation_admin  # noqa: E402

WS = uuid.uuid4()
ADMIN_PK, OPERATOR_PK = 7, 1
THEME = "acme/ink"
DIGEST = "ab" * 32
REASON = "reported: exfiltrates chat"
BASE = "/api/admin/themes/revocations"
LIFT = f"{BASE}/{{revocation_id}}/lift"
TABLES = [m.__table__ for m in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
    ThemeRevocationSync,
)]


def _ctx(role="admin", clerk="clerk-7", fresh=True):
    """A Clerk caller; ``fresh`` signed in a minute ago, so the step-up on create and lift passes."""
    claims = {"sub": clerk, "iat": int(time.time())}
    if fresh:
        claims["fva"] = [1, -1]
    user = UserContext(id=clerk, clerk_user_id=clerk, system_role=role, raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _operator():
    """The local edition's anonymous owner, as ``hybrid.py`` binds it: a super admin."""
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK})
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7'), (8, 'clerk-8')"))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx())
    app = FastAPI()
    app.include_router(revocations_api.router)
    app.include_router(install_api.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _package(db, version="1.0.0"):
    manifest = {"id": THEME, "name": "Ink", "version": version, "base": "classic", "htmlClass": "ink"}
    row = ThemePackage(
        theme_id=THEME, version=version, owner_type="marketplace", digest=uuid.uuid4().hex * 2, manifest=manifest,
        storage_prefix=f"themes/{THEME}/{version}/", reviewed=True, status="approved",
        file_manifest={"dist/entry.js": {"type": "text/javascript", "size": 1, "sha256": "ab" * 32},
                       "dist/theme.css": {"type": "text/css", "size": 1, "sha256": "cd" * 32}},
    )
    db.add(row)
    db.commit()
    return row


def _chosen(api, package):
    """Install ``package`` in the workspace and make it the caller's choice."""
    install = ThemeInstallation(workspace_id=WS, theme_id=package.theme_id, package_id=package.id)
    api.db.add(install)
    api.db.commit()
    assert api.client.put("/api/themes/choice", json={"choice": str(install.id)}).status_code == 200
    return install


def _active(api):
    resp = api.client.get("/api/themes/active")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _create(api, **body):
    return api.client.post(BASE, json={"level": "disable", "reason": REASON, **body})


def _lift(api, revocation_id):
    return api.client.post(f"{BASE}/{revocation_id}/lift")


def _rows(api):
    api.db.expire_all()
    return api.db.query(ThemeRevocation).all()


def _row(api, minutes_ago=0, lifted=False, origin=THEME_REVOCATION_ORIGIN_LOCAL, level="warn", **target):
    """A row written straight to the table (a pulled ``feed`` row, or an older ``local`` one)."""
    now = datetime.now(timezone.utc)
    row = ThemeRevocation(
        level=level, reason=f"r{minutes_ago}", origin=origin, created_at=now - timedelta(minutes=minutes_ago),
        lifted_at=now if lifted else None, **(target or {"theme_id": THEME}),
    )
    api.db.add(row)
    api.db.commit()
    return row


# --- a member's next load, through the install router -------------------------------------


TARGETS = {
    "digest": lambda package: {"digest": package.digest},
    "theme id": lambda package: {"themeId": THEME},
    "exact version": lambda package: {"themeId": THEME, "versionRange": package.version},
}


@pytest.mark.parametrize("target", list(TARGETS))
def test_a_disable_makes_active_answer_the_base_and_lift_restores_it(api, target):
    package = _package(api.db)
    install = _chosen(api, package)
    assert _active(api)["theme"]["installId"] == str(install.id)
    resp = _create(api, **TARGETS[target](package))
    assert resp.status_code == 201, resp.text
    body = _active(api)
    assert (body["choice"], body["theme"]) == ("classic", None)  # the package's base
    assert body["revoked"] == {"installId": str(install.id), "level": "disable", "reason": REASON}
    assert _lift(api, resp.json()["revocation"]["id"]).status_code == 200
    body = _active(api)
    assert body["choice"] == str(install.id) and body.get("revoked") is None
    assert (body["theme"]["installId"], body["theme"]["revoked"]) == (str(install.id), None)


def test_a_warn_by_a_caret_range_keeps_the_theme_with_the_notice_and_lift_clears_it(api):
    install = _chosen(api, _package(api.db, version="1.2.0"))
    assert _create(api, themeId=THEME, versionRange="^2.0.0", level="warn", reason="later only").status_code == 201
    assert _active(api)["theme"]["revoked"] is None  # ^2.0.0 does not cover 1.2.0
    resp = _create(api, themeId=THEME, versionRange="^1.0.0", level="warn", reason="slow on phones")
    assert resp.status_code == 201, resp.text
    body = _active(api)
    assert body["choice"] == str(install.id)
    assert body["theme"]["revoked"] == {"level": "warn", "reason": "slow on phones"}
    assert _lift(api, resp.json()["revocation"]["id"]).status_code == 200
    assert _active(api)["theme"]["revoked"] is None


# --- create ---------------------------------------------------------------------------------


def test_create_writes_a_local_row_by_the_admin_and_the_list_shows_it(api):
    resp = _create(api, digest=DIGEST)
    assert resp.status_code == 201, resp.text
    assert resp.json()["created"] is True
    view = resp.json()["revocation"]
    expected = {"digest": DIGEST, "themeId": None, "versionRange": None, "level": "disable", "reason": REASON,
                "origin": "local", "liftedAt": None}
    assert {key: view[key] for key in expected} == expected
    [row] = _rows(api)
    assert (str(row.id), row.origin, row.created_by, row.lifted_at) == (view["id"], "local", ADMIN_PK, None)
    assert view["createdAt"]
    assert api.client.get(BASE).json() == {"revocations": [view]}
    # the body has no origin: a caller can't write a feed row
    assert _create(api, digest="cd" * 32, origin=THEME_REVOCATION_ORIGIN_FEED).status_code == 422
    assert len(_rows(api)) == 1


GOOD = {"level": "disable", "reason": REASON}
ON_THEME, ON_DIGEST = {**GOOD, "themeId": THEME}, {**GOOD, "digest": DIGEST}
BAD_DIGESTS = (DIGEST.upper(), DIGEST[:63], "g" * 64, "")
BAD_THEME_IDS = ("Acme/Ink", "acme", "acme/ink/extra", "")
BAD_RANGES = ("*", ">=1.0.0", "~1.0.0", "1.0", "^1.0.0-beta.1", "", " ^1.0.0", "1.0.0 || 2.0.0")
BAD_LEVELS = ("block", "WARN", "")
BAD_REASONS = ("", "   ", "x" * (revocation_admin.REASON_MAX + 1), "two\nlines", "nul\x00here", "safe ‮ txt.exe")
BAD_BODIES = {
    "no target": GOOD,
    "range alone": {**GOOD, "versionRange": "1.0.0"},
    "digest and theme id": {**ON_DIGEST, "themeId": THEME},
    "digest with a range": {**ON_DIGEST, "versionRange": "1.0.0"},
    **{f"digest-{n}": {**GOOD, "digest": value} for n, value in enumerate(BAD_DIGESTS)},
    **{f"theme-id-{n}": {**GOOD, "themeId": value} for n, value in enumerate(BAD_THEME_IDS)},
    **{f"range-{n}": {**ON_THEME, "versionRange": value} for n, value in enumerate(BAD_RANGES)},
    **{f"level-{n}": {**ON_DIGEST, "level": value} for n, value in enumerate(BAD_LEVELS)},
    **{f"reason-{n}": {**ON_DIGEST, "reason": value} for n, value in enumerate(BAD_REASONS)},
}


@pytest.mark.parametrize("body", list(BAD_BODIES.values()), ids=list(BAD_BODIES))
def test_a_bad_target_level_or_reason_is_400_and_nothing_is_stored(api, body):
    resp = api.client.post(BASE, json=body)
    assert resp.status_code == 400, resp.text
    assert isinstance(resp.json()["detail"], str) and resp.json()["detail"]
    assert _rows(api) == []  # an unreadable range included: refused at write, never stored


def test_the_edges_are_accepted_and_the_reason_is_stored_trimmed(api):
    longest = "r" * revocation_admin.REASON_MAX
    resp = _create(api, themeId=THEME, versionRange="^0.9.12", level="warn", reason=f"  {longest} ")
    assert resp.status_code == 201, resp.text
    [row] = _rows(api)
    stored = (row.digest, row.theme_id, row.version_range, row.level, row.reason)
    assert stored == (None, THEME, "^0.9.12", "warn", longest)


def test_a_duplicate_live_revocation_answers_the_existing_row_with_200(api):
    first = _create(api, themeId=THEME, versionRange="^1.0.0")
    assert first.status_code == 201, first.text
    again = _create(api, themeId=THEME, versionRange="^1.0.0", reason="said twice")
    assert again.status_code == 200, again.text
    assert again.json() == {"revocation": first.json()["revocation"], "created": False}
    assert len(_rows(api)) == 1
    # another level, another range or no range is another revocation
    for other in ({"versionRange": "^1.0.0", "level": "warn"}, {"versionRange": "1.0.0"}, {}):
        assert _create(api, themeId=THEME, **other).status_code == 201, other
    assert len(_rows(api)) == 4
    # a lifted row no longer stands, so the same request writes a new one
    assert _lift(api, first.json()["revocation"]["id"]).status_code == 200
    renewed = _create(api, themeId=THEME, versionRange="^1.0.0")
    assert renewed.status_code == 201 and renewed.json()["revocation"]["id"] != first.json()["revocation"]["id"]


def test_a_create_that_loses_the_race_answers_the_row_that_won_with_200(api, monkeypatch):
    # The other request's row lands after this one checked for a repeat and before it writes.
    won = _row(api, level="disable", digest=DIGEST)
    checks = []
    real_check = revocation_admin._live_local_repeat

    def raced(db, target, level):
        checks.append(target)
        return None if len(checks) == 1 else real_check(db, target, level)

    monkeypatch.setattr(revocation_admin, "_live_local_repeat", raced)
    resp = _create(api, digest=DIGEST)
    assert resp.status_code == 200, resp.text
    assert resp.json()["created"] is False and resp.json()["revocation"]["id"] == str(won.id)
    assert [row.id for row in _rows(api)] == [won.id] and len(checks) == 2


def test_a_live_feed_row_is_no_duplicate_and_stays_as_it_was(api):
    feed = _row(api, origin=THEME_REVOCATION_ORIGIN_FEED, level="disable", digest=DIGEST)
    resp = _create(api, digest=DIGEST)
    assert resp.status_code == 201, resp.text
    assert resp.json()["revocation"]["origin"] == "local" and resp.json()["revocation"]["id"] != str(feed.id)
    api.db.refresh(feed)
    assert (feed.origin, feed.lifted_at, feed.reason) == ("feed", None, "r0")


# --- lift and list --------------------------------------------------------------------------


def test_lifting_a_feed_row_is_409_and_changes_nothing(api):
    feed = _row(api, origin=THEME_REVOCATION_ORIGIN_FEED)
    resp = _lift(api, feed.id)
    assert resp.status_code == 409 and resp.json()["detail"] == revocation_admin.FEED_ROW
    api.db.refresh(feed)
    assert (feed.origin, feed.lifted_at) == ("feed", None)


def test_lift_sets_lifted_at_once_and_an_unknown_id_is_404(api):
    created = _create(api, digest=DIGEST).json()["revocation"]
    first = _lift(api, created["id"])
    assert first.status_code == 200, first.text
    lifted = first.json()["revocation"]
    assert lifted["liftedAt"] and {**lifted, "liftedAt": None} == created
    again = _lift(api, created["id"])
    assert again.status_code == 200 and again.json()["revocation"] == lifted  # the first lift's time stands
    assert _lift(api, uuid.uuid4()).status_code == 404
    assert api.client.post(f"{BASE}/not-a-uuid/lift").status_code == 422


def test_the_list_shows_live_rows_first_then_the_newest_with_their_origin(api):
    _row(api, minutes_ago=30)
    _row(api, minutes_ago=1, lifted=True)
    _row(api, minutes_ago=5, level="disable")  # one live local row per target and level
    _row(api, minutes_ago=10, origin=THEME_REVOCATION_ORIGIN_FEED)
    listed = api.client.get(BASE).json()["revocations"]
    expected = [("r5", "local"), ("r10", "feed"), ("r30", "local"), ("r1", "local")]  # live first, then newest
    assert [(item["reason"], item["origin"]) for item in listed] == expected
    assert listed[3]["liftedAt"] and listed[0]["liftedAt"] is None
    assert api.client.get(BASE, params={"limit": 2}).json()["revocations"] == listed[:2]
    assert api.client.get(BASE, params={"limit": 0}).status_code == 422
    assert api.client.get(BASE, params={"limit": revocation_admin.LIST_LIMIT_MAX + 1}).status_code == 422


# --- gates and editions ---------------------------------------------------------------------


def _routes():
    return [route for route in revocations_api.router.routes if isinstance(route, APIRoute)]


def test_routes_are_plain_def_and_assert_admin_in_the_handler():
    seen = {(method, route.path) for route in _routes() for method in route.methods}
    assert seen == {("GET", BASE), ("POST", BASE), ("POST", LIFT)}
    for route in _routes():
        assert not inspect.iscoroutinefunction(route.endpoint), route.path
        # tests/authz_sweep_probe.py classifies ADMIN_GATED_IN_HANDLER by this text.
        assert "_require_admin(ctx" in inspect.getsource(route.endpoint), route.path


def test_create_and_lift_carry_the_admin_check_then_step_up_and_the_list_none():
    for route in _routes():
        calls = [dep.call for dep in route.dependant.dependencies]
        marked = [n for n, call in enumerate(calls) if getattr(call, step_up.STEP_UP_MARKER_ATTR, None)]
        if "POST" in route.methods:
            assert marked and calls.index(revocations_api.require_revocation_admin) < marked[0], route.path
            assert getattr(calls[marked[0]], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
        else:
            assert marked == [], route.path


CALLERS = {
    "member": lambda: _ctx(role="user", clerk="clerk-8"),
    "stale member": lambda: _ctx(role="user", clerk="clerk-8", fresh=False),
    "anonymous": lambda: RequestContext(workspace_id=WS, user=None, auth_type="anonymous"),
}


@pytest.mark.parametrize("caller", list(CALLERS))
def test_every_route_refuses_a_non_admin_with_the_plain_403_and_nothing_changes(api, caller):
    standing = _row(api)
    api.ctx = CALLERS[caller]()
    for resp in (api.client.get(BASE), _create(api, digest=DIGEST), _lift(api, standing.id)):
        assert resp.status_code == 403 and resp.json() == {"detail": revocations_api.ADMIN_REQUIRED}, resp.text
    [row] = _rows(api)
    assert (row.id, row.lifted_at) == (standing.id, None)


def test_a_stale_admin_is_asked_to_step_up_and_nothing_changes_and_a_fresh_super_admin_may_revoke(api):
    standing = _row(api)
    api.ctx = _ctx(fresh=False)
    for resp in (_create(api, digest=DIGEST), _lift(api, standing.id)):
        assert resp.status_code == 403, resp.text
        assert resp.json()["clerk_error"]["reason"] == step_up.REVERIFICATION_REASON
    [row] = _rows(api)
    assert (row.id, row.lifted_at) == (standing.id, None)
    assert api.client.get(BASE).status_code == 200  # the list carries no step-up
    api.ctx = _ctx(role="super_admin")
    assert _create(api, digest=DIGEST).status_code == 201


def test_the_local_editions_operator_revokes_and_lifts_without_step_up(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    api.ctx = _operator()
    resp = _create(api, digest=DIGEST)
    assert resp.status_code == 201, resp.text
    [row] = _rows(api)
    assert (row.created_by, row.origin) == (OPERATOR_PK, "local")
    assert _lift(api, row.id).status_code == 200
    assert api.client.get(BASE).json()["revocations"][0]["liftedAt"]
