"""PRD-263 US-023 — Auto's HARNESS tools read the run table and describe the report-only loop.

``platform_harness_history`` listed baseline JSON files from the workspace volume,
which no run writes since PRD-263 (US-010), so Auto answered "what did HARNESS do"
from files that stopped changing. It now lists the workspace's harness_runs rows,
newest first, and the baseline-file reader is gone. The three HARNESS action
descriptions describe the weekly report-only run on the weekday cron, coverage,
INSUFFICIENT_DATA and the one dormancy status left, and say nothing of queued cards
or model swaps. A trigger's reply says the run is report-only, and a failing status
or trigger call tells Auto it failed without the exception's text.
"""
from __future__ import annotations

import inspect
import json
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from tests import helpers_harness_runs as hr

OLDER_WEEK, NEWER_WEEK = "2026-W40", "2026-W41"
COUNTS = {"active_agents": 4, "judged_agents": 2, "insufficient_data": 2, "watch": 1, "regression": 0,
          "prescriptions": 3}
HARNESS_ACTIONS = ("platform_harness_status", "platform_harness_trigger", "platform_harness_history")
INTERNALS = 'relation "harness_runs" does not exist at /app/orchestrator/services/harness_runs.py'


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def workspaces(new_session):
    made = []

    def make():
        ws = hr.seed_workspace(new_session)
        made.append(ws)
        return ws

    yield make
    for ws in made:
        hr.sweep_workspace(new_session, ws)


def _record(new_session, ws, window, status, **fields) -> int:
    from services import harness_runs

    s = new_session()
    try:
        run_id = harness_runs.open_run(s, ws, window)
        harness_runs.close_run(s, run_id, status, **fields)
        return run_id
    finally:
        s.close()


async def _history(new_session, ws, params=None):
    from modules.tools.discovery.handlers_harness import harness_history

    s = new_session()
    try:
        return await harness_history(s, ws, params or {})
    finally:
        s.close()


@pytest.mark.asyncio
async def test_history_reads_harness_runs_newest_first(workspaces, new_session, tmp_path, monkeypatch):
    from config import config
    from services import harness_runs

    ws = workspaces()
    _record(new_session, ws, OLDER_WEEK, "completed", iteration=1, coverage=0.5, counts=COUNTS,
            verdicts={"convergence": "watching", "agents": {}}, report_status="ok")
    _record(new_session, ws, NEWER_WEEK, "failed", report_status=harness_runs.failure("boom"))
    # A baseline file as the loop before PRD-263 left it: the history no longer reads it.
    baselines = tmp_path / str(ws) / "harness" / "baselines"
    baselines.mkdir(parents=True)
    (baselines / "2026-09-06.json").write_text(json.dumps({"created_at": "2026-09-06T02:00:00+00:00", "iteration": 9}))
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))

    result = await _history(new_session, ws)

    assert result["success"] is True and result["data"]["total"] == 2
    newer, older = result["data"]["runs"]
    assert (newer["cur_window"], newer["status"], newer["error"]) == (NEWER_WEEK, "failed", "boom")
    assert newer["coverage"] is None and newer["watch"] is None  # not recorded, never 0
    assert older == {
        "cur_window": OLDER_WEEK, "status": "completed", "started_at": older["started_at"],
        "finished_at": older["finished_at"], "iteration": 1, "coverage": 0.5, "convergence": "watching",
        "active_agents": 4, "judged_agents": 2, "insufficient_data": 2, "watch": 1, "regression": 0,
        "prescription_count": 3, "applied_count": 0, "audit_report": "ok",
    }
    assert older["started_at"] and older["finished_at"]


@pytest.mark.asyncio
async def test_another_workspaces_runs_never_appear(workspaces, new_session):
    ws, other = workspaces(), workspaces()
    _record(new_session, ws, OLDER_WEEK, "completed", iteration=1, counts=COUNTS)
    _record(new_session, other, NEWER_WEEK, "completed", iteration=1, counts=COUNTS)

    result = await _history(new_session, ws)

    assert [run["cur_window"] for run in result["data"]["runs"]] == [OLDER_WEEK]
    assert result["data"]["total"] == 1


@pytest.mark.asyncio
async def test_the_limit_is_honoured(workspaces, new_session):
    ws = workspaces()
    for window in (OLDER_WEEK, NEWER_WEEK):
        _record(new_session, ws, window, "completed", iteration=1, counts=COUNTS)

    result = await _history(new_session, ws, {"limit": 1})

    assert [run["cur_window"] for run in result["data"]["runs"]] == [NEWER_WEEK]


def test_the_limit_is_read_at_the_boundary():
    from modules.tools.discovery.handlers_harness import (
        DEFAULT_HISTORY_LIMIT,
        MAX_HISTORY_LIMIT,
        _history_limit,
    )

    assert [_history_limit(raw) for raw in (None, "abc", True, 0, -5, "3", 10_000)] == [
        DEFAULT_HISTORY_LIMIT, DEFAULT_HISTORY_LIMIT, DEFAULT_HISTORY_LIMIT, 1, 1, 3, MAX_HISTORY_LIMIT]


def test_no_baseline_file_path_is_left_in_the_handlers():
    import modules.tools.discovery.handlers_harness as handlers

    source = inspect.getsource(handlers)
    for gone in ("baseline", "WORKSPACE_VOLUME_PATH", "os.path", "json.load"):
        assert gone not in source


def test_the_action_descriptions_describe_the_report_only_loop():
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    descriptions = {name: registry.get(name).description for name in HARNESS_ACTIONS}
    for text in descriptions.values():
        assert "report-only" in text
        for gone in ("queued", "model swap", "Sunday", "optimization run"):
            assert gone not in text
    status = descriptions["platform_harness_status"]
    for said in ("weekday cron", "coverage", "INSUFFICIENT_DATA", "dormant_insufficient_agents"):
        assert said in status
    for gone in ("dormant_insufficient_data", "heartbeat_days_available"):
        assert gone not in status
    assert "coverage" in descriptions["platform_harness_history"]


class _EnabledService:
    """A HARNESS service for an enabled workspace that records each trigger."""

    def __init__(self):
        self.triggered = []

    def workspace_enabled(self, db, workspace_id):
        return True

    async def trigger_now(self, workspace_id):
        self.triggered.append(workspace_id)


class _FailingService:
    """A HARNESS service whose every call fails with internals in the message."""

    def get_status(self, workspace_id, db=None):
        raise RuntimeError(INTERNALS)

    def workspace_enabled(self, db, workspace_id):
        raise RuntimeError(INTERNALS)


@pytest.mark.asyncio
async def test_the_trigger_reply_describes_the_report_only_run(monkeypatch):
    from modules.tools.discovery.handlers_harness import harness_trigger
    from services import harness_service

    service, ws = _EnabledService(), uuid4()
    monkeypatch.setattr(harness_service, "get_harness_service", lambda: service)

    reply = await harness_trigger(None, ws, {})

    assert reply["success"] is True and service.triggered == [ws]
    message = reply["data"]["message"]
    assert "report-only" in message and "changes nothing" in message
    assert "optimization" not in message


@pytest.mark.asyncio
async def test_a_failing_status_or_trigger_leaks_no_internals(monkeypatch):
    from modules.tools.discovery import handlers_harness
    from services import harness_service

    monkeypatch.setattr(harness_service, "get_harness_service", _FailingService)
    ws = uuid4()

    status = await handlers_harness.harness_status(None, ws, {})
    trigger = await handlers_harness.harness_trigger(None, ws, {})

    assert status == {"success": False, "error": handlers_harness.STATUS_UNAVAILABLE}
    assert trigger == {"success": False, "error": handlers_harness.TRIGGER_UNAVAILABLE}
