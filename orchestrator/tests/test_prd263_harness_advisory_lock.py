"""PRD-263 US-011 — one run per workspace per judged week, whatever the worker or replica count.

The guard was an in-memory dict, ``self._running``: each of the 4 uvicorn workers had
its own, ``trigger_now`` ran in whichever worker served the request, and every Railway
replica starts its own scheduler. A run now takes the workspace's Postgres advisory
lock (two-key form: the HARNESS namespace and hashtext(<workspace id>)) on a dedicated
connection; a held lock makes the call a logged no-op. Holding it, the sweep skips a
workspace whose judged week already has a completed or dormant run (a failed run is
retried), and closes a ``running`` row that no run holds as failed ("interrupted").
``trigger_now`` is exempt from the skip. get_status reads ``running`` from pg_locks,
without taking the lock.
"""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine, text

from tests import helpers_harness_runs as hr


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = hr.seed_workspace(new_session)
    yield ws
    hr.sweep_workspace(new_session, ws)


@pytest.fixture
def other_workspace(new_session):
    ws = hr.seed_workspace(new_session)
    yield ws
    hr.sweep_workspace(new_session, ws)


@pytest.fixture
def calls(monkeypatch, tmp_path):
    return hr.fake_platform(monkeypatch, tmp_path)


@pytest.fixture
def only(monkeypatch):
    """Let a sweep reach only the given workspaces (the test database holds others)."""
    from services.harness_service import HarnessService

    def _only(*workspaces):
        ids = {str(ws) for ws in workspaces}
        monkeypatch.setattr(HarnessService, "_workspace_opted_in", staticmethod(lambda ws: str(ws.id) in ids))

    return _only


def _status(new_session, ws):
    from services.harness_service import HarnessService

    s = new_session()
    try:
        return HarnessService().get_status(ws, db=s)
    finally:
        s.close()


@pytest.mark.asyncio
async def test_two_concurrent_ticks_for_one_workspace_run_the_phases_once(workspace, calls, new_session):
    """The first tick holds the lock while its faked platform calls yield; the second,
    on its own connection and session, finds the lock held and does nothing."""
    from services.harness_service import HarnessService

    await asyncio.gather(HarnessService()._harness_tick(workspace), HarnessService()._harness_tick(workspace))

    assert len(hr.runs(new_session, workspace)) == 1
    assert hr.reports(calls, workspace) == 1  # the audit report is the last phase


@pytest.mark.asyncio
async def test_ticks_for_two_workspaces_both_run(workspace, other_workspace, calls, new_session):
    from services.harness_service import HarnessService

    svc = HarnessService()
    await asyncio.gather(svc._harness_tick(workspace), svc._harness_tick(other_workspace))

    for ws in (workspace, other_workspace):
        assert [run["status"] for run in hr.runs(new_session, ws)] == ["completed"]
        assert hr.reports(calls, ws) == 1


@pytest.mark.asyncio
async def test_a_held_lock_makes_a_tick_a_no_op_and_status_reads_running(workspace, calls, new_session, engine):
    from services import harness_runs
    from services.harness_service import HarnessService

    with harness_runs.workspace_lock(engine, workspace) as held:
        assert held
        assert _status(new_session, workspace)["status"] == "running"
        await HarnessService()._harness_tick(workspace)
        assert hr.runs(new_session, workspace) == [] and hr.reports(calls, workspace) == 0

    assert _status(new_session, workspace)["status"] == "scheduled_not_run_yet"
    await HarnessService()._harness_tick(workspace)
    assert [run["status"] for run in hr.runs(new_session, workspace)] == ["completed"]


@pytest.mark.asyncio
async def test_two_sequential_sweeps_write_one_run_and_one_audit_report(workspace, calls, only, new_session):
    from services.harness_service import HarnessService

    only(workspace)
    await HarnessService()._harness_sweep()
    await HarnessService()._harness_sweep()

    (run,) = hr.runs(new_session, workspace)
    assert (run["status"], run["cur_window"]) == ("completed", hr.WINDOW)
    assert hr.reports(calls, workspace) == 1


@pytest.mark.asyncio
async def test_two_sweeps_of_a_dormant_workspace_write_one_run(workspace, calls, only, new_session, monkeypatch):
    """A second scheduler (another replica) reaching a dormant workspace after the
    first released its lock finds the week settled and writes nothing."""
    from services.harness_service import HarnessService

    short = {**hr.READY, "active_agents": 1, "agents_ok": False}
    monkeypatch.setattr(HarnessService, "_sufficiency_breakdown", lambda self, db, ws: dict(short))
    only(workspace)
    await HarnessService()._harness_sweep()
    await HarnessService()._harness_sweep()

    assert [(run["status"], run["cur_window"]) for run in hr.runs(new_session, workspace)] == [
        ("dormant_insufficient_agents", hr.WINDOW)]


@pytest.mark.asyncio
async def test_a_failed_run_does_not_settle_the_week(workspace, calls, only, new_session, monkeypatch):
    from services.harness_service import HarnessService

    def _boom(*_args, **_kwargs):
        raise RuntimeError("report builder down")

    only(workspace)
    with monkeypatch.context() as broken:
        broken.setattr(HarnessService, "_build_changelog_markdown", _boom)
        await HarnessService()._harness_sweep()
    await HarnessService()._harness_sweep()
    await HarnessService()._harness_sweep()

    assert [(run["status"], run["cur_window"]) for run in hr.runs(new_session, workspace)] == [
        ("failed", hr.WINDOW), ("completed", hr.WINDOW)]
    assert hr.reports(calls, workspace) == 1


@pytest.mark.asyncio
async def test_a_stale_running_row_is_closed_as_interrupted_and_the_run_proceeds(workspace, calls, only, new_session):
    from services.harness_service import HarnessService

    s = new_session()
    s.execute(text("INSERT INTO harness_runs (workspace_id, cur_window, status) "
                   "VALUES (CAST(:ws AS uuid), :window, 'running')"), {"ws": str(workspace), "window": hr.WINDOW})
    s.commit()
    s.close()
    status = _status(new_session, workspace)
    assert (status["status"], status["error"]) == ("failed", "interrupted")

    only(workspace)
    await HarnessService()._harness_sweep()

    stale, fresh = hr.runs(new_session, workspace)
    assert (stale["status"], stale["report_status"]) == ("failed", "failed: interrupted")
    assert stale["finished_at"] is not None
    assert (fresh["status"], fresh["cur_window"], fresh["iteration"]) == ("completed", hr.WINDOW, 1)


@pytest.mark.asyncio
async def test_trigger_now_is_exempt_from_the_weekly_skip(workspace, calls, only, new_session):
    from services.harness_service import HarnessService

    svc = HarnessService()
    only(workspace)
    await svc._harness_sweep()

    before = asyncio.all_tasks()
    await svc.trigger_now(workspace)
    await asyncio.gather(*(asyncio.all_tasks() - before))
    await svc._harness_sweep()

    assert [(run["status"], run["cur_window"]) for run in hr.runs(new_session, workspace)] == [
        ("completed", hr.WINDOW), ("completed", hr.WINDOW)]
    assert hr.reports(calls, workspace) == 2


def test_the_in_memory_guard_is_gone_and_the_lock_has_its_own_key():
    from pathlib import Path

    from core.database.boot_lock import BOOT_LOCK_ID
    from services.harness_runs import HARNESS_LOCK_NAMESPACE

    src = (Path(__file__).resolve().parents[1] / "services" / "harness_service.py").read_text()
    assert "_running" not in src
    assert HARNESS_LOCK_NAMESPACE != BOOT_LOCK_ID
    assert 0 < HARNESS_LOCK_NAMESPACE < 2**31  # the first key of pg_try_advisory_lock(int, int)
