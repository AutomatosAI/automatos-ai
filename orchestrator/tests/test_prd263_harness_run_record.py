"""PRD-263 US-010 — every HARNESS run is recorded in harness_runs, not in JSON files.

A tick wrote its outcome to <WORKSPACE_VOLUME_PATH>/<ws>/harness/runs/last_run.json and
dumped COLLECT and DIAGNOSE into traces/<day>_trace.json; get_status read that file,
so a worker whose volume held something else answered from it. Each tick now opens
one harness_runs row (status running, cur_window = the last complete ISO week) and
closes it with its status, finish time, counts, audit report and report outcome.
``iteration`` is the workspace's completed runs, this one included, and get_status
reads the latest row.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from tests import helpers_harness_runs as hr

DEFAULT_CRON = "0 10 * * 1"


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
def calls(monkeypatch, tmp_path):
    from config import config

    monkeypatch.setattr(config, "HARNESS_SWEEP_CRON", DEFAULT_CRON)
    return hr.fake_platform(monkeypatch, tmp_path)


def _status(new_session, ws):
    from services.harness_service import HarnessService

    s = new_session()
    try:
        return HarnessService().get_status(ws, db=s)
    finally:
        s.close()


@pytest.mark.asyncio
async def test_a_tick_writes_one_run_row_and_no_run_file(workspace, calls, new_session, tmp_path):
    from services.harness_service import HarnessService

    await HarnessService()._harness_tick(workspace)

    (run,) = hr.runs(new_session, workspace)
    assert (run["status"], run["cur_window"], run["iteration"], run["report_status"]) == (
        "completed", hr.WINDOW, 1, "ok")
    assert run["finished_at"] is not None and run["finished_at"] >= run["started_at"]
    assert run["report_md"].startswith("# HARNESS Weekly Org Review — Run #1")
    assert run["counts"]["prescriptions"] == 0 and run["prescriptions"] == []
    assert run["verdicts"] is None and run["coverage"] is None  # DIAGNOSE fills them (US-014a/b)
    assert hr.reports(calls, workspace) == 1

    from config import config

    assert config.WORKSPACE_VOLUME_PATH == str(tmp_path)  # the volume the tick wrote to
    assert not list(tmp_path.rglob("last_run.json"))
    assert not list(tmp_path.rglob("*_trace.json"))
    assert not (tmp_path / str(workspace) / "harness" / "traces").exists()


@pytest.mark.asyncio
async def test_iteration_counts_completed_runs_and_status_reads_the_latest_row(workspace, calls, new_session):
    from services.harness_service import HarnessService

    await HarnessService()._harness_tick(workspace)
    await HarnessService()._harness_tick(workspace)

    first, second = hr.runs(new_session, workspace)
    assert (first["iteration"], second["iteration"]) == (1, 2)
    assert second["report_md"].startswith("# HARNESS Weekly Org Review — Run #2")

    status = _status(new_session, workspace)
    assert (status["status"], status["iteration_count"]) == ("completed", 2)
    assert status["artifacts"] == {"audit_report": "ok"}
    assert status["last_run_at"] == second["finished_at"].isoformat()
    nxt = datetime.fromisoformat(status["next_scheduled_run"])
    assert (nxt.utcoffset().total_seconds(), nxt.weekday(), nxt.hour, nxt.minute) == (0, 0, 10, 0)


@pytest.mark.asyncio
async def test_a_failed_run_is_closed_failed_with_its_reason(workspace, calls, new_session, monkeypatch):
    from services.harness_service import HarnessService

    def _boom(*_args, **_kwargs):
        raise RuntimeError("report builder down")

    await HarnessService()._harness_tick(workspace)
    monkeypatch.setattr(HarnessService, "_build_changelog_markdown", _boom)
    await HarnessService()._harness_tick(workspace)

    completed, failed = hr.runs(new_session, workspace)
    assert (failed["status"], failed["report_status"], failed["iteration"]) == (
        "failed", "failed: report builder down", 1)
    assert failed["finished_at"] is not None and failed["report_md"] is None
    status = _status(new_session, workspace)
    assert (status["status"], status["error"], status["iteration_count"]) == ("failed", "report builder down", 1)


@pytest.mark.asyncio
async def test_a_dormant_run_is_one_row_with_its_reason_and_no_report(workspace, calls, new_session, monkeypatch):
    from services.harness_service import HarnessService

    short = {**hr.READY, "active_agents": 1, "agents_ok": False}
    monkeypatch.setattr(HarnessService, "_sufficiency_breakdown", lambda self, db, ws: dict(short))

    await HarnessService()._harness_tick(workspace)

    (run,) = hr.runs(new_session, workspace)
    assert (run["status"], run["iteration"], run["counts"]) == ("dormant_insufficient_agents", 0, short)
    assert run["finished_at"] is not None and run["report_md"] is None
    assert hr.reports(calls, workspace) == 0
    assert _status(new_session, workspace)["status"] == "dormant_insufficient_agents"


def test_the_judged_week_is_the_last_complete_iso_week():
    from services.harness_runs import last_complete_iso_week

    assert last_complete_iso_week(datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)) == "2026-W41"  # a Sunday
    assert last_complete_iso_week(datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)) == "2026-W42"  # a Monday
    assert last_complete_iso_week(datetime(2027, 1, 4, 10, 0, tzinfo=timezone.utc)) == "2026-W53"  # across a year end


def test_the_last_run_file_and_the_trace_writer_are_gone():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "services" / "harness_service.py").read_text()
    for name in ("_last_run_path", "_read_last_run", "_write_last_run", "last_run.json", "_trace.json",
                 "_make_serializable"):
        assert name not in src, name
