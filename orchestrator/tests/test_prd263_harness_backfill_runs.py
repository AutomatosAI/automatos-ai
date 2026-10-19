"""PRD-263 US-023 — the HARNESS history on tenant volumes is imported into harness_runs once.

Runs before PRD-263 left their outcome in files on the workspace volume: one baseline
per completed run under ``harness/baselines/<day>.json`` (the newest copied to
``baseline_latest.json``) and the latest outcome in ``harness/runs/last_run.json``.
``scripts/harness_backfill_runs.py`` copies them into harness_runs, one row per
workspace and ISO week, insert-if-absent; it defaults to a dry run, never writes or
deletes a file, and ``--summary-only`` prints totals that name no workspace.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from tests import helpers_harness_runs as hr

WORKSPACE_NAME = "prd263-harness-runs"  # helpers_harness_runs.seed_workspace's name
FIRST = {"created_at": "2026-09-06T02:00:05+00:00", "iteration": 1,
         "convergence": {"iteration_count": 1, "status": "exploring"},
         "applied_changes": [{"prescription_id": "rx-1", "change_type": "heartbeat_tune", "result": "applied"}],
         "queued_changes": [{"prescription_id": "rx-2", "change_type": "model_change", "board_task_id": 5}]}
SECOND = {"created_at": "2026-09-13T02:00:03+00:00", "iteration": 2,
          "convergence": {"iteration_count": 2, "status": "converged"},
          "applied_changes": [], "queued_changes": []}
LAST_RUN = {"status": "failed", "timestamp": "2026-09-20T02:00:09+00:00", "error": "boom"}
# The ISO week each run judged: the week before the one it ran in.
FIRST_WEEK, SECOND_WEEK, LAST_RUN_WEEK = "2026-W35", "2026-W36", "2026-W37"
LINKED_RUN_AT = "2026-09-27T02:00:00+00:00"  # a run that judged 2026-W38, in files outside the workspace


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
def volume(workspace, tmp_path, monkeypatch):
    """The workspace's HARNESS files as a run before PRD-263 left them, plus one unreadable
    baseline and one harness directory that is not a workspace's."""
    from config import config

    harness = tmp_path / str(workspace) / "harness"
    (harness / "baselines").mkdir(parents=True)
    (harness / "runs").mkdir()
    (harness / "baselines" / "2026-09-06.json").write_text(json.dumps(FIRST))
    (harness / "baselines" / "2026-09-13.json").write_text(json.dumps(SECOND))
    (harness / "baselines" / "broken.json").write_text("{not json")
    (harness / "baseline_latest.json").write_text(json.dumps(SECOND))
    (harness / "runs" / "last_run.json").write_text(json.dumps(LAST_RUN))
    (tmp_path / "not-a-workspace" / "harness").mkdir(parents=True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    return tmp_path


def _backfill(*args: str) -> int:
    from scripts.harness_backfill_runs import main

    return main(list(args))


def _snapshot(root: Path):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in sorted(root.rglob("*")) if path.is_file()}


def test_dry_run_is_the_default_and_inserts_nothing(workspace, volume, new_session, capsys):
    assert _backfill() == 0

    assert hr.runs(new_session, workspace) == []
    out = capsys.readouterr().out
    assert f"workspace {workspace}: 5 files (1 unusable), 3 weeks, would insert 3, already in harness_runs 0" in out
    assert "skipped 1 harness directories" in out and "Dry run: nothing was written" in out


def test_apply_imports_each_week_once_and_a_second_run_inserts_nothing(workspace, volume, new_session, capsys):
    assert _backfill("--apply") == 0
    rows = {row["cur_window"]: row for row in hr.runs(new_session, workspace)}

    assert sorted(rows) == [FIRST_WEEK, SECOND_WEEK, LAST_RUN_WEEK]
    first, last = rows[FIRST_WEEK], rows[LAST_RUN_WEEK]
    assert (first["status"], first["iteration"]) == ("completed", 1)
    assert first["counts"] == {"prescriptions": 2, "applied": 1, "queued": 1}
    assert [rx["outcome"] for rx in first["prescriptions"]] == ["applied", "queued"]
    # The old loop's flat-rule judgement is never stored as verdicts DIAGNOSE would read.
    assert all(row["verdicts"] is None and row["coverage"] is None for row in rows.values())
    assert (last["status"], last["report_status"], last["iteration"]) == ("failed", "failed: boom", 2)

    capsys.readouterr()
    assert _backfill("--apply") == 0
    assert len(hr.runs(new_session, workspace)) == 3
    assert "inserted 0, already in harness_runs 3" in capsys.readouterr().out


def test_summary_only_output_holds_no_workspace_id(workspace, volume, capsys):
    assert _backfill("--summary-only") == 0

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert "total over 1 workspaces: 5 files (1 unusable), 3 weeks, would insert 3" in printed
    for identifying in (str(workspace), workspace.hex, WORKSPACE_NAME):
        assert identifying not in printed


def test_files_are_never_written_or_deleted(workspace, volume):
    before = _snapshot(volume)

    assert _backfill("--apply") == 0

    assert _snapshot(volume) == before


def test_a_recorded_week_is_kept_and_the_newest_run_stays_latest(workspace, volume, new_session):
    from services import harness_runs

    s = new_session()
    run_id = harness_runs.open_run(s, workspace, LAST_RUN_WEEK)
    harness_runs.close_run(s, run_id, "completed", iteration=1)
    s.close()

    assert _backfill("--apply") == 0

    windows = [row["cur_window"] for row in hr.runs(new_session, workspace)]
    assert sorted(windows) == [FIRST_WEEK, SECOND_WEEK, LAST_RUN_WEEK]  # the recorded week was not imported again
    s = new_session()
    try:
        # The imported rows have higher ids but older start times: the run since PRD-263 is the latest.
        assert harness_runs.latest_run(s, workspace).id == run_id
        assert [run.cur_window for run in harness_runs.recent_runs(s, workspace, 10)] == [
            LAST_RUN_WEEK, SECOND_WEEK, FIRST_WEEK]
    finally:
        s.close()


def test_a_symbolic_link_is_never_followed(workspace, volume, new_session, tmp_path_factory, capsys):
    # A workspace's tools write to its volume: a link there could point at another workspace's files.
    elsewhere = tmp_path_factory.mktemp("another-workspace") / "2026-09-27.json"
    elsewhere.write_text(json.dumps({**SECOND, "created_at": "2026-09-27T02:00:00+00:00", "iteration": 3}))
    (volume / str(workspace) / "harness" / "baselines" / "2026-09-27.json").symlink_to(elsewhere)

    assert _backfill("--apply") == 0

    assert sorted(row["cur_window"] for row in hr.runs(new_session, workspace)) == [
        FIRST_WEEK, SECOND_WEEK, LAST_RUN_WEEK]
    assert "6 files (2 unusable), 3 weeks, inserted 3" in capsys.readouterr().out


@pytest.mark.parametrize("linked, kept", [
    ("baselines", [SECOND_WEEK, LAST_RUN_WEEK]),  # baseline_latest.json and runs/ stay readable
    ("runs", [FIRST_WEEK, SECOND_WEEK]),
])
def test_a_linked_directory_is_never_followed(workspace, volume, new_session, tmp_path_factory, linked, kept):
    # Glob and is_file follow a linked directory, and a file reached through one is no link itself.
    elsewhere = tmp_path_factory.mktemp("another-workspace")
    (elsewhere / "2026-09-27.json").write_text(json.dumps({**SECOND, "created_at": LINKED_RUN_AT, "iteration": 3}))
    (elsewhere / "last_run.json").write_text(json.dumps({**LAST_RUN, "timestamp": LINKED_RUN_AT}))
    harness = volume / str(workspace) / "harness"
    shutil.rmtree(harness / linked)
    (harness / linked).symlink_to(elsewhere, target_is_directory=True)

    assert _backfill("--apply") == 0

    assert sorted(row["cur_window"] for row in hr.runs(new_session, workspace)) == kept


def test_a_linked_workspace_directory_is_never_followed(workspace, volume, new_session, tmp_path_factory, capsys):
    # The workspace directory on the volume is itself a link: nothing under it is read.
    elsewhere = tmp_path_factory.mktemp("another-place") / str(workspace)
    shutil.move(str(volume / str(workspace)), str(elsewhere))
    (volume / str(workspace)).symlink_to(elsewhere, target_is_directory=True)

    assert _backfill("--apply") == 0

    assert hr.runs(new_session, workspace) == []
    assert f"workspace {workspace}" not in capsys.readouterr().out


def test_a_run_dated_this_week_or_later_is_never_imported(workspace, volume, new_session, capsys):
    # A file on the volume could be planted: a time from the current ISO week on would
    # settle the week the sweep is about to judge, so it is unusable.
    from datetime import datetime, timedelta, timezone

    harness = volume / str(workspace) / "harness"
    now = datetime.now(timezone.utc)
    for name, at in (("now", now), ("future", now + timedelta(days=30))):
        (harness / "baselines" / f"{name}.json").write_text(json.dumps({**SECOND, "created_at": at.isoformat()}))

    assert _backfill("--apply") == 0

    assert sorted(row["cur_window"] for row in hr.runs(new_session, workspace)) == [
        FIRST_WEEK, SECOND_WEEK, LAST_RUN_WEEK]
    assert "7 files (3 unusable), 3 weeks, inserted 3" in capsys.readouterr().out


def test_only_known_integer_sufficiency_keys_reach_counts(workspace, volume, new_session):
    harness = volume / str(workspace) / "harness"
    (harness / "runs" / "last_run.json").write_text(json.dumps({
        "status": "dormant_insufficient_agents", "timestamp": LAST_RUN["timestamp"],
        "sufficiency": {"active_agents": 2, "min_required_agents": 3, "agents_ok": False,
                        "note": "x" * 50, "nested": {"a": 1}},
    }))

    assert _backfill("--apply") == 0

    last = {row["cur_window"]: row for row in hr.runs(new_session, workspace)}[LAST_RUN_WEEK]
    assert last["counts"] == {"active_agents": 2, "min_required_agents": 3}
