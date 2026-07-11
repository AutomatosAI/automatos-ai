"""PRD-194 S6 (P2-13, security §1.4.a, composio C.2/§J P0-2) — fix the
destructive-gate feeder and make a silent no-op sync fail LOUDLY.

``ComposioActionSyncService`` is the sole writer of
``composio_action_metadata`` — the table backing the destructive-action gate
(F018) — and it could never write a row:

- ``_get_all_enabled_apps`` returned a hardcoded 8-app placeholder, so the
  daily sync classified apps nobody had connected (and none they had);
- ``_fetch_app_actions`` ``await``-ed ``get_app_actions``, a synchronous
  ``def`` on the client — ``TypeError`` for every app, swallowed as a
  failed ``SyncResult``. The 04:00 cron ran for months writing nothing,
  and the gate fell back to the 8-keyword intent heuristic forever.

These tests pin the fixes: the enabled set is the REAL active-connection
set; the sync client call is a thread offload of the (still-synchronous)
client method — the in-tree idiom, no fake-async fork; a full ``sync_app``
run writes metadata rows; and connected-apps-with-empty-table now fails
LOUDLY (sync result status=failed + startup ERROR log) instead of silently.
The gate reader's fail-closed keyword floor is untouched.

Pure: stub Composio client (plain ``def``), stub/recording DB sessions,
no network, no live Composio, no live DB.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from modules.tools.capabilities.models import ComposioActionMetadata  # noqa: E402
from modules.tools.sync.composio_action_sync import (  # noqa: E402
    ComposioActionSyncService,
    check_action_metadata_populated,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- stubs

class _SyncClient:
    """Composio client stand-in whose get_app_actions is a plain def —
    exactly like the real core/composio/client.py method."""

    def __init__(self):
        self.calls: list[str] = []

    def get_app_actions(self, app_name):
        self.calls.append(app_name)
        return [
            {
                "name": "GITHUB_CREATE_ISSUE",
                "display_name": "Create Issue",
                "description": "Create a new issue in a repository",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "GITHUB_DELETE_REPOSITORY",
                "display_name": "Delete Repository",
                "description": "Permanently delete a repository",
                "parameters": {"type": "object", "properties": {}},
            },
        ]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _QueueDb:
    """db.execute() answers from a FIFO of _Rows — one per expected query."""

    def __init__(self, results):
        self._results = list(results)

    def execute(self, stmt):
        return self._results.pop(0)


class _RecordingDb:
    """Every lookup misses (new action); add/commit are recorded."""

    def __init__(self):
        self.added = []
        self.commits = 0

    def execute(self, stmt):
        return _Rows([])

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1


# ---------------------------------------------------------------- the sync bug

def test_fetch_app_actions_no_await_on_sync():
    """_fetch_app_actions returns actions from a client whose get_app_actions
    is a plain def (was: TypeError on `await`, swallowed per app)."""
    client = _SyncClient()
    svc = ComposioActionSyncService(db=object(), composio_client=client)
    actions = _run(svc._fetch_app_actions("GITHUB"))
    assert client.calls == ["GITHUB"]
    assert [a.id for a in actions] == ["GITHUB_CREATE_ISSUE", "GITHUB_DELETE_REPOSITORY"]
    assert all(a.app_id == "GITHUB" for a in actions)


def test_client_method_stays_sync_and_service_offloads():
    """The locked pick-ONE: the client method stays a synchronous def (every
    other call site calls it synchronously) and the async service offloads
    via asyncio.to_thread — no fake-async fork of the client."""
    import modules.tools.sync.composio_action_sync as sync_mod

    orch_root = Path(sync_mod.__file__).resolve().parents[3]
    client_src = (orch_root / "core" / "composio" / "client.py").read_text()
    assert "def get_app_actions(" in client_src
    assert "async def get_app_actions(" not in client_src

    sync_src = Path(sync_mod.__file__).read_text()
    assert "asyncio.to_thread" in sync_src
    assert "await self.composio_client.get_app_actions" not in sync_src


def test_enabled_apps_are_connected_not_placeholder():
    """_get_all_enabled_apps reflects the ACTIVE connection set, not the
    hardcoded 8-app placeholder."""
    db = _QueueDb([_Rows([("GITHUB",), ("SLACK",)])])
    svc = ComposioActionSyncService(db=db, composio_client=None)
    apps = _run(svc._get_all_enabled_apps())
    assert apps == ["GITHUB", "SLACK"]

    # And the placeholder is really gone from the source.
    import modules.tools.sync.composio_action_sync as sync_mod

    src = Path(sync_mod.__file__).read_text()
    assert '"google_calendar"' not in src  # the old constant list's tell


def test_enabled_apps_empty_when_nothing_connected():
    db = _QueueDb([_Rows([])])
    svc = ComposioActionSyncService(db=db, composio_client=None)
    assert _run(svc._get_all_enabled_apps()) == []


def test_sync_writes_metadata_rows():
    """A sync_app run against a stubbed client writes ComposioActionMetadata
    rows (was: 0 rows, TypeError per app)."""
    db = _RecordingDb()
    svc = ComposioActionSyncService(db=db, composio_client=_SyncClient())
    result = _run(svc.sync_app("GITHUB"))

    assert result.total_actions == 2
    assert result.classified == 2
    assert result.errors == 0
    metadata_rows = [o for o in db.added if isinstance(o, ComposioActionMetadata)]
    assert len(metadata_rows) == 2
    assert db.commits >= 1
    by_id = {m.action_id: m for m in metadata_rows}
    assert by_id["GITHUB_DELETE_REPOSITORY"].app_id == "GITHUB"


# ---------------------------------------------------------------- fail loud

def test_empty_metadata_for_connected_apps_flags_loud():
    """Connected apps + EMPTY metadata table ⇒ NOT ok (was: silent forever)."""
    ok, detail = check_action_metadata_populated(
        _QueueDb([_Rows([(1,)]), _Rows([])])  # a connection; no metadata
    )
    assert ok is False
    assert "EMPTY" in detail


def test_metadata_check_ok_on_cold_start_and_when_populated():
    # No active connections — nothing to classify, not a failure.
    ok, detail = check_action_metadata_populated(_QueueDb([_Rows([])]))
    assert ok is True

    # Connections + at least one metadata row — healthy.
    ok, detail = check_action_metadata_populated(
        _QueueDb([_Rows([(1,)]), _Rows([("GITHUB_CREATE_ISSUE",)])])
    )
    assert ok is True


def test_sync_all_flags_failed_when_metadata_empty(monkeypatch):
    """The daily job surfaces the assertion: status=failed + ERROR log when
    connected apps exist but the table stayed empty (was: quiet 'completed')."""
    import core.database.database as db_mod
    import jobs.sync_composio_actions as jobs_mod
    import modules.tools.sync as sync_pkg
    import modules.tools.sync.composio_action_sync as sync_mod

    class _Db:
        def close(self):
            pass

    class _StubService:
        def __init__(self, **kwargs):
            pass

        async def sync_all_enabled_apps(self):
            return []

    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _Db())
    monkeypatch.setattr(sync_pkg, "ComposioActionSyncService", _StubService)
    monkeypatch.setattr(jobs_mod, "_get_composio_client", lambda: None)
    monkeypatch.setattr(jobs_mod, "_get_llm_client", lambda: None)

    monkeypatch.setattr(
        sync_mod, "check_action_metadata_populated", lambda db: (False, "EMPTY sentinel")
    )
    result = _run(jobs_mod.sync_all_composio_actions())
    assert result["status"] == "failed"
    assert result["metadata_assertion"] == "EMPTY sentinel"

    monkeypatch.setattr(
        sync_mod, "check_action_metadata_populated", lambda db: (True, "populated")
    )
    result = _run(jobs_mod.sync_all_composio_actions())
    assert result["status"] == "completed"
    assert result["metadata_assertion"] == "populated"


def test_scheduler_startup_check_logs_error(monkeypatch, caplog):
    """The boot-time assertion says so at ERROR when the gate is blind —
    logged, never raised (a stale table must not crash boot)."""
    import core.database.database as db_mod
    import modules.tools.sync.composio_action_sync as sync_mod
    import services.composio_sync_scheduler as sched_mod

    class _Db:
        def close(self):
            pass

    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _Db())
    monkeypatch.setattr(
        sync_mod, "check_action_metadata_populated", lambda db: (False, "gate is blind")
    )
    with caplog.at_level(logging.ERROR):
        sched_mod.ComposioSyncScheduler._startup_metadata_check()
    assert any("STARTUP ASSERTION FAILED" in r.message for r in caplog.records)

    # And a healthy table logs info, no error — never raises either way.
    caplog.clear()
    monkeypatch.setattr(
        sync_mod, "check_action_metadata_populated", lambda db: (True, "populated")
    )
    with caplog.at_level(logging.ERROR):
        sched_mod.ComposioSyncScheduler._startup_metadata_check()
    assert not any("STARTUP ASSERTION FAILED" in r.message for r in caplog.records)
