"""Where a night's evidence goes: ``~/.automatos-sim/runs/<run_id>/`` and the campaign store.

    run.json          the whole record (rewritten after every scenario, so a
                      crash still leaves the scenarios that finished)
    scorecard.md      the four rows and the findings, readable in a minute
    trace.jsonl       every HTTP exchange, full payloads
    run.log           the runner's own log
    llm-settings-snapshot.json   what the global tiers were before the run

``campaign.sqlite`` keeps one row per run and per scenario so nights can be
compared without opening the JSON.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import CAMPAIGN_DB, RUNS_DIR, SIM_HOME

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, pack TEXT, started_at TEXT, ended_at TEXT, "
    "status TEXT, workspace_id TEXT, model_id TEXT, cost_usd REAL, usability REAL, quality REAL, "
    "usefulness REAL, outcome_rate REAL, scenarios INTEGER, ok INTEGER, run_dir TEXT)",
    "CREATE TABLE IF NOT EXISTS scenarios (run_id TEXT, scenario_id TEXT, kind TEXT, outcome TEXT, ok INTEGER, "
    "duration_s REAL, cost_usd REAL, calls INTEGER, usability REAL, quality REAL, usefulness REAL, "
    "PRIMARY KEY (run_id, scenario_id))",
)


class RunLocked(RuntimeError):
    """Another run holds the lock — two runs would snapshot and restore the same global rows."""


def acquire_run_lock(sim_home: Path = SIM_HOME):
    """One run at a time per machine. Returns the open handle; closing it releases the lock."""
    sim_home.mkdir(parents=True, exist_ok=True)
    handle = (sim_home / "run.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RunLocked(f"{sim_home / 'run.lock'} is held by another run (pid {_lock_holder(sim_home)})") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _lock_holder(sim_home: Path) -> str:
    try:
        return (sim_home / "run.lock").read_text(encoding="utf-8").strip() or "?"
    except OSError:
        return "?"


def new_run_dir(pack: str, runs_dir: Path = RUNS_DIR) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{pack}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def write_json(path: Path, obj: Any) -> None:
    """Atomic: written beside, then renamed over."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_run(run_dir: Path, run: Mapping[str, Any]) -> None:
    write_json(run_dir / "run.json", run)


def update_latest(run_dir: Path, runs_dir: Path = RUNS_DIR) -> None:
    link = runs_dir / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run_dir.name)


def latest_run_dir(runs_dir: Path = RUNS_DIR) -> Path | None:
    link = runs_dir / "latest"
    return link.resolve() if link.exists() else None


def record_campaign(run: Mapping[str, Any], run_dir: Path, db_path: Path = CAMPAIGN_DB) -> None:
    card = run.get("scorecard") or {}
    with sqlite3.connect(db_path) as conn:
        for statement in SCHEMA:
            conn.execute(statement)
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.get("run_id"), run.get("pack"), run.get("started_at"), run.get("ended_at"), run.get("status"),
             (run.get("workspace") or {}).get("id"), (run.get("model") or {}).get("model_id"),
             card.get("cost_usd"), card.get("usability"), card.get("quality"), card.get("usefulness"),
             card.get("outcome_rate"), card.get("scenarios"), card.get("ok"), str(run_dir)),
        )
        for sc in run.get("scenarios") or []:
            scores = sc.get("scores") or {}
            conn.execute(
                "INSERT OR REPLACE INTO scenarios VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run.get("run_id"), sc.get("id"), sc.get("kind"), sc.get("outcome"), 1 if sc.get("ok") else 0,
                 scores.get("duration_s"), scores.get("cost_usd"), scores.get("calls"), scores.get("usability"),
                 scores.get("quality"), scores.get("usefulness")),
            )


def previous_runs(pack: str, limit: int = 5, db_path: Path = CAMPAIGN_DB) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT run_id, started_at, status, cost_usd, usability, quality, usefulness, outcome_rate, ok, scenarios "
            "FROM runs WHERE pack = ? ORDER BY started_at DESC LIMIT ?", (pack, limit)).fetchall()
    return [dict(r) for r in rows]
