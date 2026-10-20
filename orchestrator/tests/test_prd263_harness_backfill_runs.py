"""PRD-263 US-023, US-059 — the HARNESS history on tenant volumes is imported into
harness_runs once, and the files leave the volume only after a verified import.

Runs before PRD-263 left their outcome in files on the workspace volume: one baseline
per completed run under ``harness/baselines/<day>.json`` (the newest copied to
``baseline_latest.json``) and the latest outcome in ``harness/runs/last_run.json``.
``scripts/harness_backfill_runs.py`` copies them into harness_runs, one row per
workspace and ISO week, insert-if-absent; it defaults to a dry run, never writes a
file, and ``--summary-only`` prints totals that name no workspace. Without
``--remove-files`` it deletes no file; with it (and ``--apply``) a workspace's JSON
files go only once every week they hold is in harness_runs.
"""
from __future__ import annotations

import errno
import json
import os
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


def test_remove_files_only_after_verified_import(workspace, volume, new_session, monkeypatch, capsys):
    """A dry run, and an import whose rows never land, remove nothing; once every week
    the files hold is in harness_runs, --apply --remove-files deletes every JSON file
    under harness/ (a trace the import never reads too) and nothing else."""
    from services import harness_runs

    harness = volume / str(workspace) / "harness"
    (harness / "traces").mkdir()
    (harness / "traces" / "2026-09-13_trace.json").write_text("{}")
    (harness / "notes.txt").write_text("not HARNESS JSON")
    before = _snapshot(volume)

    assert _backfill("--remove-files") == 0  # the default dry run, before any import
    assert _snapshot(volume) == before
    assert "would remove 0 JSON files, import count not verified (files kept)" in capsys.readouterr().out

    with monkeypatch.context() as patch:
        patch.setattr(harness_runs, "record_imported_runs", lambda db, ws, runs: 0)  # the rows never land
        assert _backfill("--apply", "--remove-files") == 0
    assert _snapshot(volume) == before and hr.runs(new_session, workspace) == []
    assert "removed 0 JSON files, import count not verified in 1 workspaces (files kept)" in capsys.readouterr().out

    assert _backfill("--apply") == 0
    assert _backfill("--remove-files") == 0  # verified now, but still a dry run
    assert _snapshot(volume) == before
    assert "would remove 6 JSON files, import count not verified in 0 workspaces" in capsys.readouterr().out

    assert _backfill("--apply", "--remove-files") == 0
    assert (f"workspace {workspace}: 5 files (1 unusable), 3 weeks, inserted 0, already in harness_runs 3, "
            "removed 6 JSON files") in capsys.readouterr().out
    assert list(harness.rglob("*.json")) == []
    assert (harness / "notes.txt").read_text() == "not HARNESS JSON"
    assert len(hr.runs(new_session, workspace)) == 3


def test_remove_files_never_follows_or_removes_a_link(workspace, volume, tmp_path_factory):
    # A workspace's tools write to its volume: a link there could point at another workspace's files.
    elsewhere = tmp_path_factory.mktemp("another-workspace")
    (elsewhere / "victim.json").write_text("{}")
    harness = volume / str(workspace) / "harness"
    (harness / "linked.json").symlink_to(elsewhere / "victim.json")
    (harness / "linked-dir").symlink_to(elsewhere, target_is_directory=True)

    assert _backfill("--apply", "--remove-files") == 0

    assert (elsewhere / "victim.json").read_text() == "{}"
    assert (harness / "linked.json").is_symlink() and (harness / "linked-dir").is_symlink()
    assert not (harness / "runs" / "last_run.json").exists()


def test_remove_files_never_opens_a_linked_directory(workspace, volume, tmp_path_factory, monkeypatch):
    # Opening a linked directory opens its target: one the script cannot read would stop every run.
    harness = volume / str(workspace) / "harness"
    (harness / "linked-dir").symlink_to(tmp_path_factory.mktemp("unreadable"), target_is_directory=True)
    real_open = os.open

    def unreadable_target(path, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(path) == "linked-dir":
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", unreadable_target)
        assert _backfill("--apply", "--remove-files") == 0

    assert (harness / "linked-dir").is_symlink()
    assert not (harness / "runs" / "last_run.json").exists()


def test_remove_files_summary_holds_no_workspace_id(workspace, volume, capsys):
    assert _backfill("--apply") == 0
    capsys.readouterr()

    assert _backfill("--apply", "--remove-files", "--summary-only") == 0

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert ("total over 1 workspaces: 5 files (1 unusable), 3 weeks, inserted 0, already in harness_runs 3, "
            "removed 5 JSON files, import count not verified in 0 workspaces (files kept)") in printed
    for identifying in (str(workspace), workspace.hex, WORKSPACE_NAME):
        assert identifying not in printed
