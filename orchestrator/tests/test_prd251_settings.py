"""PRD-251 S0.1 — the two Socials switches, the workspace route and the approve permission.

Pins D1 (Socials is gated two ways, on every plan) and the D6 permission:

* the master switch — the ``socials.enabled`` system setting, read strictly
  through ``read_system_setting``: a readable row decides, no row (or an empty
  value) takes ``SOCIALS_ENABLED_DEFAULT``; the real read path is proven
  against an in-memory ``system_settings`` table;
* a read that cannot complete — SessionLocal raises, the query raises, the
  table is missing — is OFF even with ``SOCIALS_ENABLED_DEFAULT=true``, logged
  at ERROR: the gate answers 404 and ``/current`` reports Socials unavailable
  (P251-RVW-8);
* the migration seed — ``prd251_socials`` seeds the row from the config
  default, insert-if-absent, never overwriting a super-admin's choice;
* the workspace switch — ``parse_workspace_socials`` / ``validate_socials_update``
  are fail-closed;
* ``require_socials_enabled`` — 404 unless BOTH switches are on;
* ``PUT /api/workspaces/current/socials`` — ``workspace:manage`` (a viewer or
  editor gets 403), rejects what the voice-live route rejects, persists with
  ``flag_modified``; ``GET /api/workspaces/current`` reports both switches;
* ``socials:approve`` — owner, admin and editor; never viewer;
* no plan exposure key — every plan gets Socials.

DB-free: fakes and in-memory SQLite stand in for Postgres.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain; nothing
# here touches a real database (CI's real POSTGRES_* makes these no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.socials as socials_api  # noqa: E402
import api.workspaces as workspaces_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.system_settings import SettingCategory, SystemSetting  # noqa: E402
from modules.policy.roles import workspace_has_permission  # noqa: E402
from tests.helpers_unreadable_settings import UNREADABLE_MODES, settings_unreadable  # noqa: E402

WS_ID = uuid.uuid4()
SWITCH_ROUTE = "/api/workspaces/current/socials"
MIGRATION = _ORCH / "alembic" / "versions" / "prd251_socials.py"
MANIFEST = _ORCH / "reports" / "route-manifest.json"


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


def _workspace(settings=None):
    return SimpleNamespace(
        id=WS_ID,
        name="Socials Test",
        slug="socials-test",
        plan="basic",
        plan_limits={},
        settings={} if settings is None else settings,
        onboarding={},
        webhook_key="k" * 32,
        owner_id=1,
    )


class _Query:
    def __init__(self, db):
        self._db = db

    def get(self, ident):
        return self._db.lookup(ident)

    # PRD-251 S1.2 (US-106): turning Socials on seeds the social starters, which
    # look each one up by name first; this workspace holds no templates yet.
    def filter(self, *_criteria):
        return self

    def first(self):
        return None


class _FakeDB:
    """Answers ``db.query(Workspace).get(id)`` and ``db.get(Workspace, id)``, and keeps what is added."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.commits = 0
        self.added = []

    def add(self, row):
        self.added.append(row)

    def lookup(self, ident):
        ws = self.workspace
        return ws if ws is not None and ident == ws.id else None

    def query(self, _model):
        return _Query(self)

    def get(self, _model, ident):
        return self.lookup(ident)

    def commit(self):
        self.commits += 1


def _member_ctx():
    return RequestContext(
        workspace_id=WS_ID,
        user=UserContext(id="member-1", clerk_user_id="clerk_member", system_role="user"),
        auth_type="clerk",
    )


def _anonymous_ctx():
    return RequestContext(
        workspace_id=WS_ID,
        user=UserContext(id="local", system_role="user"),
        auth_type="anonymous",
    )


def _client(db, ctx):
    app = FastAPI()
    app.include_router(workspaces_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _as_role(monkeypatch, role):
    """The caller's workspace role, as the member lookup would resolve it."""
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role)


@pytest.fixture
def master(monkeypatch):
    """The ``socials.enabled`` row: ``None`` = no row (the default applies)."""
    state = {"value": None}

    def fake_read_system_setting(category, key):
        assert (category, key) == ("socials", "enabled")
        return state["value"]

    monkeypatch.setattr(socials_settings, "read_system_setting", fake_read_system_setting)
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", False)
    return state


@pytest.fixture
def flag_spy(monkeypatch):
    """Records ``flag_modified`` — the fake workspace carries no ORM state."""
    calls = []
    monkeypatch.setattr(
        "sqlalchemy.orm.attributes.flag_modified",
        lambda instance, key: calls.append((instance, key)),
    )
    return calls


def _settings_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SystemSetting.__table__.create(bind=engine)
    return engine


def _load_migration():
    spec = importlib.util.spec_from_file_location("prd251_socials_migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The master switch
# ---------------------------------------------------------------------------


def test_the_master_switch_lives_in_the_socials_category():
    assert socials_settings.SOCIALS_SETTINGS_CATEGORY == SettingCategory.SOCIALS.value == "socials"
    assert socials_settings.KEY_ENABLED == "enabled"


def test_master_switch_is_off_by_default(master):
    assert socials_settings.socials_master_enabled() is False


def test_master_switch_is_on_when_the_system_setting_is_true(master):
    master["value"] = "true"
    assert socials_settings.socials_master_enabled() is True


def test_master_switch_follows_the_config_default_when_no_row_exists(master, monkeypatch):
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", True)
    assert socials_settings.socials_master_enabled() is True


def test_the_row_wins_over_the_config_default(master, monkeypatch):
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", True)
    master["value"] = "false"
    assert socials_settings.socials_master_enabled() is False


def test_master_switch_reads_the_real_system_settings_row(monkeypatch):
    """The real read path: read_system_setting → SessionLocal → system_settings."""
    import core.database.database as database_mod

    engine = _settings_engine()
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", False)
    table = SystemSetting.__table__
    try:
        assert socials_settings.socials_master_enabled() is False  # no row → default off

        with engine.begin() as conn:
            conn.execute(
                table.insert().values(
                    category="socials", key="enabled", value="true", value_type="boolean"
                )
            )
        assert socials_settings.socials_master_enabled() is True  # no restart needed

        with engine.begin() as conn:
            conn.execute(
                table.update()
                .where(table.c.category == "socials", table.c.key == "enabled")
                .values(value="false")
            )
        assert socials_settings.socials_master_enabled() is False
    finally:
        engine.dispose()


_NO_ROW = object()


@pytest.mark.parametrize(
    "default_on, row, expected",
    [
        pytest.param(False, _NO_ROW, False, id="no row, default off"),
        pytest.param(True, _NO_ROW, True, id="no row, default on"),
        pytest.param(False, None, False, id="NULL value, default off"),
        pytest.param(True, None, True, id="NULL value, default on"),
        pytest.param(False, "", False, id="empty value, default off"),
        pytest.param(True, "", True, id="empty value, default on"),
        pytest.param(False, "true", True, id="true, default off"),
        pytest.param(True, "true", True, id="true, default on"),
        pytest.param(True, "false", False, id="false, default on"),
        pytest.param(True, "yes", False, id="any other value, default on"),
    ],
)
def test_a_readable_row_decides_through_the_real_read_path(monkeypatch, default_on, row, expected):
    """read_system_setting → SessionLocal → system_settings: a readable row
    decides, and only no row or an empty value takes the config default."""
    import core.database.database as database_mod

    engine = _settings_engine()
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", default_on)
    try:
        if row is not _NO_ROW:
            with engine.begin() as conn:
                conn.execute(
                    SystemSetting.__table__.insert().values(
                        category="socials", key="enabled", value=row, value_type="boolean"
                    )
                )
        assert socials_settings.socials_master_enabled() is expected
    finally:
        engine.dispose()


def test_with_the_default_on_a_super_admins_off_applies_on_the_next_call(monkeypatch):
    """The owner's socials stack: SOCIALS_ENABLED_DEFAULT=true, then a
    super-admin switches Socials off and on again, with no restart."""
    import core.database.database as database_mod

    engine = _settings_engine()
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", True)
    table = SystemSetting.__table__
    where = (table.c.category == "socials", table.c.key == "enabled")
    try:
        assert socials_settings.socials_master_enabled() is True  # no row → default on

        with engine.begin() as conn:
            conn.execute(
                table.insert().values(category="socials", key="enabled", value="false", value_type="boolean")
            )
        assert socials_settings.socials_master_enabled() is False

        with engine.begin() as conn:
            conn.execute(table.update().where(*where).values(value="true"))
        assert socials_settings.socials_master_enabled() is True

        with engine.begin() as conn:
            conn.execute(table.update().where(*where).values(value="false"))
        assert socials_settings.socials_master_enabled() is False
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A read that cannot complete is OFF, whatever the default (P251-RVW-8)
# ---------------------------------------------------------------------------


@pytest.fixture(params=UNREADABLE_MODES)
def master_unreadable(request, monkeypatch):
    """The ``socials.enabled`` row cannot be read, on a stack whose default is
    ON (the owner's socials stack sets SOCIALS_ENABLED_DEFAULT=true): SessionLocal
    raises (an exhausted pool), the query raises (a dropped connection) or the
    real query runs on a database without system_settings. Yields the module's
    logger, mocked."""
    monkeypatch.setattr(socials_settings.config, "SOCIALS_ENABLED_DEFAULT", True)
    logger = MagicMock(name="logger")
    monkeypatch.setattr(socials_settings, "logger", logger)
    with settings_unreadable(request.param, monkeypatch):
        yield logger


def test_a_failed_read_is_off_and_logged_once_at_error(master_unreadable):
    assert socials_settings.socials_master_enabled() is False

    master_unreadable.error.assert_called_once()
    assert "could not be read" in master_unreadable.error.call_args.args[0]
    assert master_unreadable.error.call_args.kwargs["exc_info"] is True


def test_a_failed_read_reports_socials_unavailable(master_unreadable):
    assert socials_settings.socials_state({"socials": {"enabled": True}}) == {
        "available": False,
        "enabled": True,
    }


def test_a_failed_read_closes_the_gate_for_a_workspace_whose_switch_is_on(master_unreadable):
    with pytest.raises(HTTPException) as exc:
        _gate(_FakeDB(_workspace({"socials": {"enabled": True}})))
    assert exc.value.status_code == 404


def test_a_failed_read_is_404_over_http_and_current_reports_socials_unavailable(master_unreadable):
    # The workspace comes from the request's own session (get_db), a separate
    # checkout that can still succeed while the settings read fails.
    db = _FakeDB(_workspace({"socials": {"enabled": True}}))
    app = FastAPI()
    app.include_router(socials_api.router)
    app.include_router(workspaces_api.router)
    app.dependency_overrides[get_request_context_hybrid] = _anonymous_ctx
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)
    assert "/api/socials/posts" in {route.path for route in app.routes}  # the 404 is the gate's

    assert client.get("/api/socials/posts").status_code == 404

    resp = client.get("/api/workspaces/current")
    assert resp.status_code == 200
    assert resp.json()["socials"] == {"available": False, "enabled": True}


# ---------------------------------------------------------------------------
# The migration seed (the voice_live mechanism)
# ---------------------------------------------------------------------------


def test_migration_is_the_one_socials_revision_on_kb_multimodal_tables():
    mod = _load_migration()
    assert mod.revision == "prd251_socials"
    assert mod.down_revision == "kb_multimodal_tables"


@pytest.mark.parametrize("default_on, expected", [(False, "false"), (True, "true")])
def test_migration_seeds_the_master_switch_from_the_config_default(monkeypatch, default_on, expected):
    import config as config_module

    monkeypatch.setattr(config_module.config, "SOCIALS_ENABLED_DEFAULT", default_on)
    mod = _load_migration()
    engine = _settings_engine()
    try:
        with engine.begin() as conn:
            mod._seed_settings(conn, mod._socials_settings_seed())
            mod._seed_settings(conn, mod._socials_settings_seed())  # a re-run adds nothing
            rows = conn.execute(
                text(
                    "SELECT category, key, value, value_type, default_value, created_by "
                    "FROM system_settings"
                )
            ).fetchall()
    finally:
        engine.dispose()
    assert [tuple(r) for r in rows] == [
        ("socials", "enabled", expected, "boolean", expected, "prd251")
    ]


def test_migration_seed_never_overwrites_the_super_admins_choice(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.config, "SOCIALS_ENABLED_DEFAULT", False)
    mod = _load_migration()
    engine = _settings_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                SystemSetting.__table__.insert().values(
                    category="socials", key="enabled", value="true",
                    value_type="boolean", created_by="admin",
                )
            )
            mod._seed_settings(conn, mod._socials_settings_seed())
            rows = conn.execute(
                text("SELECT value, created_by FROM system_settings WHERE category = 'socials'")
            ).fetchall()
    finally:
        engine.dispose()
    assert [tuple(r) for r in rows] == [("true", "admin")]


# ---------------------------------------------------------------------------
# The workspace switch — pure parse / validate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings, expected",
    [
        (None, False),
        ({}, False),
        ({"socials": None}, False),
        ({"socials": "on"}, False),
        ({"socials": {}}, False),
        ({"socials": {"enabled": "true"}}, False),
        ({"socials": {"enabled": 1}}, False),
        ({"socials": {"enabled": False}}, False),
        ({"socials": {"enabled": True}}, True),
    ],
)
def test_parse_workspace_socials_is_fail_closed(settings, expected):
    assert socials_settings.parse_workspace_socials(settings).enabled is expected


@pytest.mark.parametrize("enabled", [True, False])
def test_validate_socials_update_accepts_a_boolean_enabled(enabled):
    assert socials_settings.validate_socials_update({"enabled": enabled}) == {"enabled": enabled}


@pytest.mark.parametrize(
    "value, reason",
    [
        (None, "object"),
        ("on", "object"),
        ([], "object"),
        ({}, "required"),
        ({"enabled": "true"}, "boolean"),
        ({"enabled": 1}, "boolean"),
        ({"enabled": True, "plan": "pro"}, "subset"),
    ],
)
def test_validate_socials_update_is_fail_closed(value, reason):
    with pytest.raises(ValueError) as exc:
        socials_settings.validate_socials_update(value)
    assert reason in str(exc.value)


# ---------------------------------------------------------------------------
# require_socials_enabled — 404 unless both switches are on
# ---------------------------------------------------------------------------


def _gate(db, ctx=None):
    return asyncio.run(socials_settings.require_socials_enabled(ctx=ctx or _member_ctx(), db=db))


def test_gate_is_404_when_the_master_switch_is_off(master):
    master["value"] = "false"
    with pytest.raises(HTTPException) as exc:
        _gate(_FakeDB(_workspace({"socials": {"enabled": True}})))
    assert exc.value.status_code == 404


def test_gate_is_404_when_the_master_is_on_but_the_workspace_switch_is_off(master):
    master["value"] = "true"
    for settings in ({}, {"socials": {"enabled": False}}):
        with pytest.raises(HTTPException) as exc:
            _gate(_FakeDB(_workspace(settings)))
        assert exc.value.status_code == 404


def test_gate_is_404_when_the_workspace_is_missing(master):
    master["value"] = "true"
    with pytest.raises(HTTPException) as exc:
        _gate(_FakeDB(None))
    assert exc.value.status_code == 404


def test_gate_passes_when_both_switches_are_on(master):
    master["value"] = "true"
    ctx = _member_ctx()
    assert _gate(_FakeDB(_workspace({"socials": {"enabled": True}})), ctx) is ctx


def test_gate_answers_404_over_http_and_follows_both_switches_live(master):
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(socials_settings.require_socials_enabled)])
    def probe():
        return {"ok": True}

    db = _FakeDB(_workspace({"socials": {"enabled": True}}))
    app.dependency_overrides[get_request_context_hybrid] = _member_ctx
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)

    master["value"] = "false"
    assert client.get("/probe").status_code == 404
    master["value"] = "true"
    assert client.get("/probe").status_code == 200
    db.workspace.settings = {"socials": {"enabled": False}}
    assert client.get("/probe").status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/workspaces/current/socials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["viewer", "editor"])
def test_put_needs_workspace_manage(monkeypatch, flag_spy, master, role):
    _as_role(monkeypatch, role)
    ws = _workspace({})
    db = _FakeDB(ws)

    resp = _client(db, _member_ctx()).put(SWITCH_ROUTE, json={"socials": {"enabled": True}})

    assert resp.status_code == 403
    assert "workspace:manage" in resp.json()["detail"]
    assert ws.settings == {} and db.commits == 0 and flag_spy == []


@pytest.mark.parametrize("role", ["owner", "admin"])
def test_put_persists_the_switch_with_flag_modified(monkeypatch, flag_spy, master, role):
    _as_role(monkeypatch, role)
    master["value"] = "true"
    ws = _workspace({"voice_live": {"enabled": True}})
    db = _FakeDB(ws)
    client = _client(db, _member_ctx())

    resp = client.put(SWITCH_ROUTE, json={"socials": {"enabled": True}})

    assert resp.status_code == 200
    assert resp.json() == {"status": "saved", "socials": {"available": True, "enabled": True}}
    assert ws.settings == {"voice_live": {"enabled": True}, "socials": {"enabled": True}}
    assert flag_spy == [(ws, "settings")]
    assert db.commits == 1

    resp = client.put(SWITCH_ROUTE, json={"socials": {"enabled": False}})
    assert resp.status_code == 200
    assert ws.settings["socials"] == {"enabled": False}


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"socials": {"enabled": True, "plan": "pro"}}, "subset"),
        ({"socials": {"enabled": "yes"}}, "boolean"),
        ({"socials": "on"}, "object"),
        ({}, "object"),
    ],
)
def test_put_rejects_malformed_bodies_and_writes_nothing(monkeypatch, flag_spy, master, body, reason):
    _as_role(monkeypatch, "owner")
    ws = _workspace({})
    db = _FakeDB(ws)

    resp = _client(db, _member_ctx()).put(SWITCH_ROUTE, json=body)

    assert resp.status_code == 400
    assert reason in resp.json()["detail"]
    assert ws.settings == {} and db.commits == 0 and flag_spy == []


@pytest.mark.parametrize(
    "route, key",
    [
        ("/api/workspaces/current/voice-live", "voice_live"),
        (SWITCH_ROUTE, "socials"),
    ],
)
@pytest.mark.parametrize("bad", [{"enabled": "yes"}, {"enabled": True, "unknown_key": 1}])
def test_socials_route_rejects_what_the_voice_live_route_rejects(monkeypatch, flag_spy, master, route, key, bad):
    _as_role(monkeypatch, "owner")
    db = _FakeDB(_workspace({}))

    resp = _client(db, _member_ctx()).put(route, json={key: bad})

    assert resp.status_code == 400
    assert db.commits == 0


def test_put_is_404_for_a_missing_workspace(monkeypatch, flag_spy, master):
    _as_role(monkeypatch, "owner")
    resp = _client(_FakeDB(None), _member_ctx()).put(SWITCH_ROUTE, json={"socials": {"enabled": True}})
    assert resp.status_code == 404


def test_put_route_is_gated_and_in_the_committed_manifest():
    route = next(
        r for r in workspaces_api.router.routes if getattr(r, "path", None) == SWITCH_ROUTE
    )
    assert route.methods == {"PUT"}
    markers = [
        getattr(dep.call, permission_mod.PERMISSION_MARKER_ATTR, None)
        for dep in route.dependant.dependencies
    ]
    assert "workspace:manage" in markers

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "PUT", "path": SWITCH_ROUTE} in manifest["routes"]


# ---------------------------------------------------------------------------
# GET /api/workspaces/current — both switches, as the frontend gate reads them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "master_value, ws_settings, expected",
    [
        ("false", {}, {"available": False, "enabled": False}),
        ("false", {"socials": {"enabled": True}}, {"available": False, "enabled": True}),
        ("true", {}, {"available": True, "enabled": False}),
        ("true", {"socials": {"enabled": True}}, {"available": True, "enabled": True}),
    ],
)
def test_get_current_reports_both_switches(master, master_value, ws_settings, expected):
    master["value"] = master_value

    resp = _client(_FakeDB(_workspace(ws_settings)), _anonymous_ctx()).get("/api/workspaces/current")

    assert resp.status_code == 200
    assert resp.json()["socials"] == expected


# ---------------------------------------------------------------------------
# socials:approve (D6) and no plan exposure key (D1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role, allowed",
    [("owner", True), ("admin", True), ("editor", True), ("viewer", False)],
)
def test_socials_approve_is_owner_admin_and_editor(role, allowed):
    assert workspace_has_permission(role, "socials:approve") is allowed


def test_socials_env_reads_live_only_in_config_py():
    """The SOCIALS_* group is read from the environment in config.py alone."""
    # Built dynamically so this guard file does not contain the literals it hunts.
    prefix = "SOCIALS" + "_"
    needles = (
        f'getenv("{prefix}', f"getenv('{prefix}",
        f'environ["{prefix}', f"environ['{prefix}",
        f'environ.get("{prefix}', f"environ.get('{prefix}",
    )
    offenders = []
    for py in _ORCH.rglob("*.py"):
        if "__pycache__" in py.parts or "tests" in py.parts or py.name == "config.py":
            continue
        try:
            source = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(n in source for n in needles):
            offenders.append(str(py.relative_to(_ORCH)))
    assert not offenders, f"SOCIALS_* read from the environment outside config.py: {offenders}"


def test_every_plan_gets_socials_there_is_no_plan_exposure_key():
    from config import _PLAN_TIERS_DEFAULTS

    assert "socials" not in (_ORCH / "services" / "plan_tiers.py").read_text(encoding="utf-8").lower()
    for tier in _PLAN_TIERS_DEFAULTS.values():
        assert "socials" not in (tier.get("families") or {})
