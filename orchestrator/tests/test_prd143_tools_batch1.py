"""PRD-143 S10 — operator tools batch 1: the setup-surface gap-fill.

Inventory result (AC #1): agents, playbooks and missions were already fully
covered (create/update/delete/configure; execute/status; create-launches via
CoordinatorService). The honest gaps were:

  - model/power config READ  -> platform_get_power_mode
  - channel connect/configure -> platform_list_channels / platform_connect_channel
                                 / platform_configure_channel / platform_start_channel
                                 / platform_stop_channel
  - widget config             -> platform_get_widget_config / platform_update_widget_config
  - knowledge upload          -> platform_upload_document

Every handler is workspace-scoped and calls the same service/DB layer its REST
router uses (channel connect delegates to api.channels.connect_channel_for_workspace,
extracted from the POST /api/channels flow; upload reuses api.documents'
get_document_manager + UPLOAD_DIR/MAX_UPLOAD_BYTES; widget config reuses the
PUBLIC_WIDGET_CONFIG_KEYS whitelist). Import idiom mirrors
test_platform_actions_registration.py (dummy POSTGRES_* on a closed port +
apscheduler stub) — nothing here touches a DB.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
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

from modules.tools.discovery.handlers_channels import (  # noqa: E402
    configure_channel,
    connect_channel,
    list_channels,
    start_channel,
    stop_channel,
)
from modules.tools.discovery.handlers_documents import upload_document  # noqa: E402
from modules.tools.discovery.handlers_power import get_power_mode  # noqa: E402
from modules.tools.discovery.handlers_widgets import (  # noqa: E402
    get_widget_config,
    update_widget_config,
)

_WS = UUID("00000000-0000-0000-0000-0000000000aa")

BATCH1_TOOLS = {
    "platform_get_power_mode": "read",
    "platform_list_channels": "read",
    "platform_connect_channel": "write",
    "platform_configure_channel": "write",
    "platform_start_channel": "write",
    "platform_stop_channel": "write",
    "platform_get_widget_config": "read",
    "platform_update_widget_config": "write",
    "platform_upload_document": "write",
}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _CaptureQuery:
    """Chainable query that records filter expressions; first() is scripted."""

    def __init__(self, first=None):
        self._first = first
        self.filters = []

    def filter(self, *args, **kwargs):
        self.filters.extend(args)
        return self

    def first(self):
        return self._first


class _ORMDB:
    def __init__(self, first=None):
        self.q = _CaptureQuery(first)
        self.committed = False
        self.rolledback = False
        self.added = []

    def query(self, *a, **k):
        return self.q

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolledback = True

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 4242


def _bound_values(q):
    """Extract literal bind values from captured SQLAlchemy filter expressions."""
    vals = []
    for f in q.filters:
        right = getattr(f, "right", None)
        v = getattr(right, "value", None)
        if v is not None:
            vals.append(v)
    return vals


class _TextDB:
    """Records db.execute(text(...), params); fetchall/fetchone are scripted."""

    def __init__(self, rows=None, row=None):
        self.calls = []
        self._rows = rows or []
        self._row = row
        self.committed = False

    def execute(self, clause, params=None):
        self.calls.append((str(clause), dict(params or {})))
        res = MagicMock()
        res.fetchall.return_value = self._rows
        res.fetchone.return_value = self._row
        return res

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


class _FakeWorkspace:
    def __init__(self, settings=None):
        self.id = _WS
        self.settings = settings


# ---------------------------------------------------------------------------
# platform_get_power_mode
# ---------------------------------------------------------------------------

def test_get_power_mode_reads_workspace_setting():
    db = _ORMDB(first=_FakeWorkspace(settings={"power_mode": "max"}))
    out = _run(get_power_mode(db, _WS, {}))
    assert out["success"] is True
    assert out["power_mode"] == "max"
    assert out["source"] == "workspace_setting"
    assert _WS in _bound_values(db.q), "query must filter by workspace_id"


def test_get_power_mode_falls_back_to_platform_default():
    db = _ORMDB(first=_FakeWorkspace(settings={"power_mode": "warp-11"}))
    out = _run(get_power_mode(db, _WS, {}))
    assert out["success"] is True
    assert out["power_mode"] == "standard"
    assert out["source"] == "platform_default"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def _channel_row(**over):
    base = dict(
        id="11111111-2222-3333-4444-555555555555",
        platform="telegram",
        status="inactive",
        mode="polling",
        webhook_url=None,
        last_verified=None,
        last_error=None,
        metadata={},
        default_agent_id=None,
        message_count=3,
        last_activity_at=None,
        created_at=None,
        config={"bot_token": "tok"},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _stub_channel_manager(monkeypatch, running=False):
    manager = MagicMock()
    manager.is_running.return_value = running
    manager.start_calls = []
    manager.stop_calls = []

    async def _start(*a, **k):
        manager.start_calls.append((a, k))

    async def _stop(*a, **k):
        manager.stop_calls.append((a, k))

    manager.start_adapter = _start
    manager.stop_adapter = _stop
    mod = types.ModuleType("channels.manager")
    mod.get_channel_manager = lambda: manager
    monkeypatch.setitem(sys.modules, "channels.manager", mod)
    return manager


def test_list_channels_workspace_scoped(monkeypatch):
    _stub_channel_manager(monkeypatch, running=True)
    db = _TextDB(rows=[_channel_row()])
    out = _run(list_channels(db, _WS, {}))
    assert out["success"] is True
    assert out["count"] == 1
    assert out["channels"][0]["platform"] == "telegram"
    # live-state reconcile: adapter running -> reported active
    assert out["channels"][0]["status"] == "active"
    sql, params = db.calls[0]
    assert "workspace_id = :ws_id" in sql
    assert params["ws_id"] == str(_WS)


def test_connect_channel_delegates_to_canonical_flow(monkeypatch):
    import api.channels as channels_api

    captured = {}

    async def _fake_connect(db, workspace_id, platform, config, default_agent_id=None, mode=None):
        captured.update(
            workspace_id=workspace_id, platform=platform, config=config,
            default_agent_id=default_agent_id, mode=mode,
        )
        return {"id": "ch-1", "platform": platform, "status": "active", "mode": "polling"}

    monkeypatch.setattr(channels_api, "connect_channel_for_workspace", _fake_connect)
    out = _run(connect_channel(MagicMock(), _WS, {
        "platform": "telegram",
        "config": {"bot_token": "tok"},
    }))
    assert out["success"] is True
    assert out["status"] == "active"
    assert captured["workspace_id"] == str(_WS), "must scope connect to the caller workspace"
    assert captured["platform"] == "telegram"
    assert captured["config"] == {"bot_token": "tok"}


def test_connect_channel_invalid_input_fails_closed(monkeypatch):
    import api.channels as channels_api

    async def _fake_connect(*a, **k):
        raise ValueError("Platform must be one of ...")

    monkeypatch.setattr(channels_api, "connect_channel_for_workspace", _fake_connect)
    out = _run(connect_channel(MagicMock(), _WS, {"platform": "myspace", "config": {}}))
    assert out["success"] is False
    assert "Platform must be one of" in out["error"]


def test_connect_channel_canonical_helper_exists():
    """The router-extracted helper is the single connect flow (no handler reimplementation)."""
    import inspect

    import api.channels as channels_api

    helper = channels_api.connect_channel_for_workspace
    assert inspect.iscoroutinefunction(helper)
    params = list(inspect.signature(helper).parameters)
    for expected in ("db", "workspace_id", "platform", "config"):
        assert expected in params


def test_configure_channel_workspace_scoped():
    db = _TextDB(row=SimpleNamespace(id="ch-1"))
    out = _run(configure_channel(db, _WS, {
        "channel_id": "ch-1",
        "config": {"bot_token": "new"},
        "default_agent_id": "7",
    }))
    assert out["success"] is True
    select_sql, select_params = db.calls[0]
    assert "workspace_id = :ws_id" in select_sql
    assert select_params["ws_id"] == str(_WS)
    update_sql, update_params = db.calls[1]
    assert "UPDATE channel_connections" in update_sql
    assert update_params["id"] == "ch-1"
    assert db.committed is True


def test_configure_channel_not_found_fails_closed():
    db = _TextDB(row=None)
    out = _run(configure_channel(db, _WS, {"channel_id": "ghost", "config": {}}))
    assert out["success"] is False
    assert "not found" in out["error"].lower()


def test_start_channel_workspace_scoped(monkeypatch):
    manager = _stub_channel_manager(monkeypatch)
    db = _TextDB(row=_channel_row())
    out = _run(start_channel(db, _WS, {"channel_id": "ch-1"}))
    assert out["success"] is True
    assert out["status"] == "started"
    sql, params = db.calls[0]
    assert "workspace_id = :ws_id" in sql and params["ws_id"] == str(_WS)
    (args, _kwargs) = manager.start_calls[0]
    assert args[1] == str(_WS), "adapter must start under the caller workspace"
    update_sql, _ = db.calls[1]
    assert "'active'" in update_sql


def test_stop_channel_workspace_scoped(monkeypatch):
    manager = _stub_channel_manager(monkeypatch)
    db = _TextDB(row=_channel_row())
    out = _run(stop_channel(db, _WS, {"channel_id": "ch-1"}))
    assert out["success"] is True
    assert out["status"] == "stopped"
    sql, params = db.calls[0]
    assert "workspace_id = :ws_id" in sql and params["ws_id"] == str(_WS)
    assert manager.stop_calls, "must stop the adapter via ChannelManager"
    update_sql, _ = db.calls[1]
    assert "'inactive'" in update_sql


# ---------------------------------------------------------------------------
# Widget config
# ---------------------------------------------------------------------------

def test_get_widget_config_returns_public_slice_only():
    from api.widgets.config import PUBLIC_WIDGET_CONFIG_KEYS

    db = _ORMDB(first=_FakeWorkspace(settings={
        "widget_proactive": {"enabled": True},
        "integrations": {"secret": "nope"},
    }))
    out = _run(get_widget_config(db, _WS, {}))
    assert out["success"] is True
    assert out["widget_config"] == {"widget_proactive": {"enabled": True}}
    assert "integrations" not in out["widget_config"]
    assert out["configurable_keys"] == list(PUBLIC_WIDGET_CONFIG_KEYS)
    assert _WS in _bound_values(db.q), "query must filter by workspace_id"


def test_update_widget_config_happy_path():
    ws = _FakeWorkspace(settings={"cart_idle": {"enabled": False}, "other": 1})
    original_settings = ws.settings
    db = _ORMDB(first=ws)
    out = _run(update_widget_config(db, _WS, {
        "key": "cart_idle",
        "config": {"enabled": True, "idle_seconds": 30},
    }))
    assert out["success"] is True
    assert out["previous"] == {"enabled": False}
    assert ws.settings["cart_idle"] == {"enabled": True, "idle_seconds": 30}
    assert ws.settings["other"] == 1
    assert ws.settings is not original_settings, "must reassign a fresh dict (JSONB change detection)"
    assert db.committed is True
    assert _WS in _bound_values(db.q)


def test_update_widget_config_rejects_non_whitelisted_key():
    db = _ORMDB(first=_FakeWorkspace(settings={}))
    out = _run(update_widget_config(db, _WS, {
        "key": "integrations",
        "config": {"hack": True},
    }))
    assert out["success"] is False
    assert db.committed is False


# ---------------------------------------------------------------------------
# platform_upload_document
# ---------------------------------------------------------------------------

def _stub_modules_rag(monkeypatch):
    mod = types.ModuleType("modules.rag")

    class DocumentType:
        PDF = "pdf"
        TEXT = "text"
        MARKDOWN = "markdown"
        JSON = "json"

    mod.DocumentType = DocumentType
    monkeypatch.setitem(sys.modules, "modules.rag", mod)
    return DocumentType


def test_upload_document_happy_path(monkeypatch, tmp_path):
    import api.documents as documents_api

    doc_type = _stub_modules_rag(monkeypatch)
    processed = {}

    class _FakeManager:
        async def _process_document(self, doc_id, path, file_type):
            processed.update(doc_id=doc_id, path=path, file_type=file_type)

    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents_api, "get_document_manager", lambda ws: _FakeManager())

    db = _ORMDB(first=None)  # no duplicate
    out = _run(upload_document(db, _WS, {
        "filename": "notes.md",
        "content": "# Hello\nknowledge",
        "description": "test doc",
    }))
    assert out["success"] is True
    assert out["document_id"] == 4242
    assert processed["doc_id"] == 4242
    assert processed["file_type"] == doc_type.MARKDOWN
    doc = db.added[0]
    assert str(doc.workspace_id) == str(_WS), "document row must be workspace-scoped"
    assert doc.original_filename == "notes.md"
    # dedupe query bound the workspace too
    assert _WS in _bound_values(db.q)
    # content actually landed on disk for the processor
    assert processed["path"].startswith(str(tmp_path))


def test_upload_document_rejects_unsupported_extension(monkeypatch, tmp_path):
    import api.documents as documents_api

    _stub_modules_rag(monkeypatch)
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)
    db = _ORMDB(first=None)
    out = _run(upload_document(db, _WS, {"filename": "evil.exe", "content": "x"}))
    assert out["success"] is False
    assert db.added == []


def test_upload_document_duplicate_short_circuits(monkeypatch, tmp_path):
    import api.documents as documents_api

    _stub_modules_rag(monkeypatch)
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)
    existing = SimpleNamespace(id=99, filename="notes.md", status="processed")
    db = _ORMDB(first=existing)
    out = _run(upload_document(db, _WS, {"filename": "notes.md", "content": "same"}))
    assert out["success"] is True
    assert out["status"] == "duplicate"
    assert out["document_id"] == 99
    assert db.added == []


# ---------------------------------------------------------------------------
# Registration + tier + executor wiring
# ---------------------------------------------------------------------------

def test_batch1_tools_registered_and_operator_tier():
    from modules.tools.discovery.action_registry import ActionRegistry

    registry = ActionRegistry()
    actions = {a.name: a for a in registry.get_all()}
    for name, level in BATCH1_TOOLS.items():
        assert name in actions, f"{name} missing from registry"
        action = actions[name]
        assert action.super_admin_only is False, f"{name} must be operator tier (Rev 2 inversion)"
        assert action.admin_only is False, f"{name} must not be admin-gated"
        assert action.workspace_scoped is True, f"{name} must be workspace-scoped"
        assert action.permission_level == level, (
            f"{name}: expected permission_level={level!r}, got {action.permission_level!r}"
        )


def test_batch1_handlers_wired_in_executor():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(MagicMock(), uuid4())
    for name in BATCH1_TOOLS:
        assert name in executor._handlers, f"{name} has no executor handler"
