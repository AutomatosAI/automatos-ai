"""PRD-142 Wave 4 (W4-S5): the workspace power-mode knob.

Power mode is workspace-scoped, not an agent attribute. This wave adds a
persistent per-workspace default (``workspace.settings['power_mode']``) with
three units under test:

  1. the ``platform_set_power_mode`` handler — validates the tier (fail-closed)
     and writes the workspace setting via a fresh dict (JSONB mutation detection);
  2. ``coordinator_service._workspace_power_mode_default`` — the per-task read
     a Mission run uses to inherit the default, best-effort (never raises);
  3. the HARNESS apply branch — ``power_mode_*`` prescriptions map onto the
     action, and the action is registered as a platform tool.

Dummy POSTGRES_* + the apscheduler stub let the harness_service / coordinator
import chains load without a real DB or the prod-only scheduler.
"""
import asyncio
import os
import sys
import types
from uuid import UUID

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

from modules.tools.discovery.handlers_power import set_power_mode  # noqa: E402
from services.coordinator_service import _workspace_power_mode_default  # noqa: E402
from services.harness_service import get_harness_service  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


def _run(coro):
    return asyncio.run(coro)


class _FakeWorkspace:
    def __init__(self, settings=None):
        self.id = _WS
        self.settings = settings


class _WSQuery:
    def __init__(self, ws):
        self._ws = ws

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._ws


class _WSDB:
    """Routes every ``query()`` to one workspace row; tracks commit/rollback."""

    def __init__(self, ws):
        self._ws = ws
        self.committed = False
        self.rolledback = False

    def query(self, _model):
        return _WSQuery(self._ws)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolledback = True


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, name, params):
        self.calls.append((name, params))
        return {"success": True}


# --- the handler -----------------------------------------------------------

def test_set_power_mode_persists_and_preserves_other_settings():
    ws = _FakeWorkspace(settings={"existing_key": 1})
    db = _WSDB(ws)
    out = _run(set_power_mode(db, _WS, {"power_mode": "max"}))

    assert out["success"] is True
    assert ws.settings["power_mode"] == "max"
    assert ws.settings["existing_key"] == 1          # untouched
    assert out["data"]["previous_power_mode"] is None
    assert out["data"]["workspace_id"] == str(_WS)
    assert db.committed is True


def test_set_power_mode_reports_previous():
    ws = _FakeWorkspace(settings={"power_mode": "standard"})
    out = _run(set_power_mode(_WSDB(ws), _WS, {"power_mode": "light"}))
    assert out["success"] is True
    assert out["data"]["previous_power_mode"] == "standard"
    assert ws.settings["power_mode"] == "light"


def test_set_power_mode_rejects_unknown_tier_without_writing():
    ws = _FakeWorkspace(settings={})
    db = _WSDB(ws)
    out = _run(set_power_mode(db, _WS, {"power_mode": "turbo"}))

    assert out["success"] is False
    assert "power_mode" in out["error"]
    assert "power_mode" not in ws.settings           # fail-closed, no write
    assert db.committed is False


def test_set_power_mode_missing_param_rejected():
    ws = _FakeWorkspace(settings={})
    out = _run(set_power_mode(_WSDB(ws), _WS, {}))
    assert out["success"] is False
    assert "power_mode" not in ws.settings


def test_set_power_mode_workspace_not_found():
    out = _run(set_power_mode(_WSDB(None), _WS, {"power_mode": "light"}))
    assert out["success"] is False
    assert "not found" in out["error"].lower()


# --- the coordinator inheritance read --------------------------------------

def test_workspace_default_returns_stored_mode():
    db = _WSDB(_FakeWorkspace(settings={"power_mode": "max"}))
    assert _workspace_power_mode_default(_WS, db) == "max"


def test_workspace_default_none_when_unset():
    db = _WSDB(_FakeWorkspace(settings={}))
    assert _workspace_power_mode_default(_WS, db) is None


def test_workspace_default_none_for_unknown_stored_value():
    # A garbage stored value never leaks to the dispatch path — the caller
    # falls back to 'standard' rather than feeding an unknown tier downstream.
    db = _WSDB(_FakeWorkspace(settings={"power_mode": "turbo"}))
    assert _workspace_power_mode_default(_WS, db) is None


def test_workspace_default_best_effort_on_db_error():
    class _BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db down")

    # Best-effort: a lookup failure degrades to the default, never fails the task.
    assert _workspace_power_mode_default(_WS, _BoomDB()) is None


# --- the HARNESS apply branch ---------------------------------------------

def test_power_mode_downgrade_maps_to_set_power_mode():
    svc = get_harness_service()
    ex = _FakeExecutor()
    rx = {"change_type": "power_mode_downgrade", "target_id": None,
          "proposed_value": {"power_mode": "light"}}
    out = _run(svc._auto_apply_prescription(ex, rx))
    assert out.get("success") is True
    assert ex.calls == [("platform_set_power_mode", {"power_mode": "light"})]


# --- registration ----------------------------------------------------------

def test_power_action_registered_with_enum():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_power import register_power_actions

    reg = ActionRegistry()
    register_power_actions(reg)

    # Read _actions directly: get() would trigger a full platform-registry init.
    action = reg._actions.get("platform_set_power_mode")
    assert action is not None
    assert action.permission_level == "write"
    props = action.parameters["properties"]["power_mode"]
    assert props["enum"] == ["light", "standard", "max"]
    assert action.parameters["required"] == ["power_mode"]
