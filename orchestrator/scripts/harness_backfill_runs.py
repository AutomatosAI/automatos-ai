"""
PRD-263 US-023: import the HARNESS history on tenant volumes into harness_runs
==============================================================================

Before PRD-263 a HARNESS run kept its outcome in files on the workspace volume:
``<WORKSPACE_VOLUME_PATH>/<ws>/harness/baselines/<day>.json`` (one per completed run,
newest copied to ``baseline_latest.json``) and ``harness/runs/last_run.json`` (the
latest run's outcome, also when it failed or went dormant). Runs are now rows of
``harness_runs`` (US-010), and Auto's ``platform_harness_history`` reads only those,
so this one-off copies the old runs in, one row per workspace and ISO week:

* a baseline is a completed run, judging the ISO week before the one it ran in
  (``last_complete_iso_week``, as a run does today). Its counts keep how many changes
  that run applied and queued, its prescriptions the changes themselves. Coverage and
  verdicts stay empty: the old loop judged with a flat 10% rule, and DIAGNOSE must
  never read those as flags (US-014b);
* last_run.json is imported only when its week has no baseline: a failed or dormant
  run, or a completed one whose baseline is gone.

Insert-if-absent by workspace and ``cur_window``: a week that already has a row
(a run since PRD-263, or an earlier import) is left alone, so a second run inserts
nothing. Of several files for one week, the newest baseline is kept. A directory
that is not a workspace id, or names no workspace, is skipped and counted; a
symbolic link, to a file or to a directory under ``harness/``, is never followed (a
workspace's tools write to its volume).

It reads files and never writes or deletes one (removing them is US-059, after a
verified import). It writes nothing unless given ``--apply``; the default is a dry
run. Output is per-workspace counts; ``--summary-only`` prints totals with no
workspace id, name or slug, for pasting into a public issue. Run it where the API
runs (it reads the same volume and database)::

    cd orchestrator
    python scripts/harness_backfill_runs.py                       # dry run, per workspace
    python scripts/harness_backfill_runs.py --summary-only        # dry run, totals only
    python scripts/harness_backfill_runs.py --apply               # insert the missing rows
    python scripts/harness_backfill_runs.py --apply --workspace-id <uuid>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.harness import (  # noqa: E402
    RUN_COMPLETED,
    RUN_DORMANT_AGENTS,
    RUN_DORMANT_DATA,
    RUN_FAILED,
)
from services import harness_runs  # noqa: E402

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger("harness_backfill_runs")

HARNESS_DIR = "harness"
BASELINES_DIR = "baselines"
BASELINE_LATEST = "baseline_latest.json"
LAST_RUN = os.path.join("runs", "last_run.json")
# The statuses last_run.json held (HarnessService._write_last_run before PRD-263).
LAST_RUN_STATUSES = frozenset({RUN_COMPLETED, RUN_FAILED, RUN_DORMANT_AGENTS, RUN_DORMANT_DATA})
# The changelog lists a baseline kept, and the counts key each one's length goes under.
APPLIED_CHANGES = "applied_changes"
QUEUED_CHANGES = "queued_changes"
QUEUED_COUNT = "queued"
# What a failed run that left no reason says it failed of.
UNKNOWN_ERROR = "unknown"


@dataclasses.dataclass(frozen=True)
class Tally:
    """One workspace's outcome. ``weeks`` is the ISO weeks its files hold a run for:
    each is either ``inserted`` (on a dry run, would be) or ``present`` (already in
    harness_runs), so weeks == inserted + present once a run has gone through."""

    files: int = 0
    unusable: int = 0
    weeks: int = 0
    inserted: int = 0
    present: int = 0

    def plus(self, other: "Tally") -> "Tally":
        return Tally(*(a + b for a, b in zip(dataclasses.astuple(self), dataclasses.astuple(other))))


# ---------------------------------------------------------------------- reading files


def _linked(path: Path, harness_dir: Path) -> bool:
    """Whether the path, or a directory between ``harness_dir`` and it, is a symbolic link.

    Glob and is_file follow a linked directory, and a file reached through one is no
    link itself, so each step below the harness directory is checked.
    """
    parts = path.relative_to(harness_dir).parts
    return any(harness_dir.joinpath(*parts[:depth]).is_symlink() for depth in range(1, len(parts) + 1))


def _load_json(path: Path, harness_dir: Path) -> Optional[Dict[str, Any]]:
    """The file's JSON object, or None (logged) when it cannot be read or is not an object.

    A symbolic link is never followed: a workspace's tools write to its volume, and a
    link could point at another workspace's files.
    """
    if _linked(path, harness_dir):
        logger.warning("%s is or lies under a symbolic link; skipped", path)
        return None
    try:
        if not path.resolve(strict=True).is_relative_to(harness_dir.resolve(strict=True)):
            logger.warning("%s resolves outside its workspace's harness directory; skipped", path)
            return None
        # O_NOFOLLOW: a link swapped in after the checks above is refused, not followed.
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        logger.warning("Cannot read %s as JSON; skipped", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        logger.warning("%s holds no JSON object; skipped", path)
        return None
    return data


def _parse_time(value: Any, fallback: Optional[datetime] = None) -> Optional[datetime]:
    """An ISO 8601 time as an aware UTC datetime (naive reads as UTC), else ``fallback``."""
    if not isinstance(value, str):
        return fallback
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return fallback
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _day_of(path: Path) -> Optional[datetime]:
    """Midnight UTC of a ``<YYYY-MM-DD>.json`` archive's day, or None for another name."""
    return _parse_time(path.stem)


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _changes(data: Dict[str, Any], key: str, outcome: str) -> List[Dict[str, Any]]:
    """The baseline's ``key`` change list, each change tagged with what the old loop did."""
    entries = data.get(key)
    if not isinstance(entries, list):
        return []
    return [{**entry, "outcome": outcome} for entry in entries if isinstance(entry, dict)]


def _baseline_row(data: Dict[str, Any], path: Path) -> Optional[Dict[str, Any]]:
    """A completed run's harness_runs row from its baseline, or None without a time."""
    at = _parse_time(data.get("created_at"), _day_of(path))
    if at is None:
        return None
    applied = _changes(data, APPLIED_CHANGES, harness_runs.IMPORTED_APPLIED)
    queued = _changes(data, QUEUED_CHANGES, QUEUED_COUNT)
    convergence = data.get("convergence") if isinstance(data.get("convergence"), dict) else {}
    return {
        "cur_window": harness_runs.last_complete_iso_week(at),
        "started_at": at,
        "finished_at": at,
        "status": RUN_COMPLETED,
        "iteration": _int(data.get("iteration"), _int(convergence.get("iteration_count"))),
        "counts": {
            harness_runs.PRESCRIPTION_COUNT: len(applied) + len(queued),
            harness_runs.IMPORTED_APPLIED: len(applied),
            QUEUED_COUNT: len(queued),
        },
        "prescriptions": applied + queued,
    }


def _last_run_row(data: Dict[str, Any], baselines: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The latest run's row from last_run.json, or None when its status or time is unreadable.

    Its iteration is the one it recorded, else that of the newest baseline before it.
    """
    status, at = data.get("status"), _parse_time(data.get("timestamp"))
    if status not in LAST_RUN_STATUSES or at is None:
        return None
    earlier = [row["iteration"] for row in baselines if row["started_at"] <= at]
    row: Dict[str, Any] = {
        "cur_window": harness_runs.last_complete_iso_week(at),
        "started_at": at,
        "finished_at": at,
        "status": status,
        "iteration": _int(data.get("iteration"), max(earlier, default=0)),
    }
    if status == RUN_FAILED:
        row["report_status"] = harness_runs.failure(str(data.get("error") or UNKNOWN_ERROR))
    if isinstance(data.get("sufficiency"), dict):
        row["counts"] = data["sufficiency"]
    return row


def _baselines(harness_dir: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    """The completed runs the workspace's baselines hold, newest first; plus how many
    baseline files were read and how many could not be used."""
    paths = sorted((harness_dir / BASELINES_DIR).glob("*.json"))
    latest = harness_dir / BASELINE_LATEST
    paths += [latest] if latest.is_file() else []
    rows = []
    for path in paths:
        data = _load_json(path, harness_dir)
        row = _baseline_row(data, path) if data is not None else None
        rows += [row] if row else []
    rows.sort(key=lambda row: row["started_at"], reverse=True)
    return rows, len(paths), len(paths) - len(rows)


def read_workspace_runs(harness_dir: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    """The rows a workspace's HARNESS files hold, baselines newest first and then
    last_run; plus how many files were read and how many could not be used."""
    rows, files, unusable = _baselines(harness_dir)
    path = harness_dir / LAST_RUN
    if not path.is_file():
        return rows, files, unusable
    data = _load_json(path, harness_dir)
    last = _last_run_row(data, rows) if data is not None else None
    return rows + ([last] if last else []), files + 1, unusable + (last is None)


def plan_inserts(rows: Sequence[Dict[str, Any]], recorded: AbstractSet[str]) -> List[Dict[str, Any]]:
    """One row for each ISO week the rows hold and harness_runs does not: the first
    row given for a week wins (the newest baseline, before last_run)."""
    chosen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["cur_window"] not in recorded:
            chosen.setdefault(row["cur_window"], row)
    return list(chosen.values())


# ---------------------------------------------------------------------- the database


def workspace_dirs(volume: Path, only: Optional[UUID]) -> Tuple[List[Tuple[UUID, Path]], int]:
    """(workspace id, harness directory) for each directory on the volume named by a
    UUID that holds a harness directory (only ``only`` when given); plus how many
    harness directories were skipped because their directory is not named by a UUID."""
    found: List[Tuple[UUID, Path]] = []
    skipped = 0
    real_volume = volume.resolve() if volume.is_dir() else volume
    for entry in sorted(volume.iterdir()) if volume.is_dir() else []:
        harness_dir = entry / HARNESS_DIR
        # The workspace directory itself may not be a link either, and the harness
        # directory must really sit at <volume>/<workspace>/harness.
        if entry.is_symlink() or harness_dir.is_symlink() or not harness_dir.is_dir():
            continue
        if harness_dir.resolve() != real_volume / entry.name / HARNESS_DIR:
            continue
        try:
            workspace_id = UUID(entry.name)
        except ValueError:
            skipped += 1
            continue
        if only is None or workspace_id == only:
            found.append((workspace_id, harness_dir))
    return found, skipped


def _workspace_exists(db: "Session", workspace_id: UUID) -> bool:
    from core.models.workspaces import Workspace

    return db.query(Workspace.id).filter(Workspace.id == workspace_id).first() is not None


def backfill_workspace(db: "Session", workspace_id: UUID, harness_dir: Path, apply: bool) -> Tally:
    """Read one workspace's files and, with ``apply``, insert the weeks harness_runs lacks."""
    rows, files, unusable = read_workspace_runs(harness_dir)
    recorded = harness_runs.recorded_windows(db, workspace_id)
    to_insert = plan_inserts(rows, recorded)
    if apply and to_insert:
        harness_runs.record_imported_runs(db, workspace_id, to_insert)
    weeks = {row["cur_window"] for row in rows}
    return Tally(files=files, unusable=unusable, weeks=len(weeks), inserted=len(to_insert),
                 present=len(weeks & recorded))


def backfill(
    session_factory: Callable[[], "Session"], volume: Path, only: Optional[UUID], apply: bool
) -> Tuple[List[Tuple[UUID, Tally]], int]:
    """Each workspace's tally, and how many harness directories name no workspace."""
    dirs, skipped = workspace_dirs(volume, only)
    results: List[Tuple[UUID, Tally]] = []
    with session_factory() as db:
        for workspace_id, harness_dir in dirs:
            if not _workspace_exists(db, workspace_id):
                skipped += 1
                continue
            results.append((workspace_id, backfill_workspace(db, workspace_id, harness_dir, apply)))
    return results, skipped


# ---------------------------------------------------------------------- output


def _line(label: str, tally: Tally, verb: str) -> str:
    return (f"{label}: {tally.files} files ({tally.unusable} unusable), {tally.weeks} weeks, "
            f"{verb} {tally.inserted}, already in harness_runs {tally.present}")


def report_lines(results: Sequence[Tuple[UUID, Tally]], skipped: int, apply: bool, summary_only: bool) -> List[str]:
    """Per-workspace lines (none with ``summary_only``), then the totals, which name no workspace."""
    verb = "inserted" if apply else "would insert"
    lines = [] if summary_only else [_line(f"workspace {ws}", tally, verb) for ws, tally in results]
    total = reduce(Tally.plus, (tally for _ws, tally in results), Tally())
    lines.append(_line(f"total over {len(results)} workspaces", total, verb))
    lines.append(f"skipped {skipped} harness directories that name no workspace")
    if not apply:
        lines.append("Dry run: nothing was written. Pass --apply to insert the missing weeks.")
    return lines


def _args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import HARNESS run history from tenant volumes (PRD-263 US-023)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false", help="Count what would be inserted (default)")
    mode.add_argument("--apply", dest="apply", action="store_true", help="Insert the missing weeks")
    parser.set_defaults(apply=False)
    parser.add_argument("--summary-only", action="store_true",
                        help="Print totals only, with no workspace id (safe for a public issue)")
    parser.add_argument("--workspace-id", type=UUID, default=None, help="Only this workspace (UUID)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # A skipped file's warning names its path, which holds the workspace id.
    logger.setLevel(logging.ERROR if args.summary_only else logging.INFO)
    from config import config
    from core.database.database import SessionLocal

    results, skipped = backfill(SessionLocal, Path(config.WORKSPACE_VOLUME_PATH), args.workspace_id, args.apply)
    for line in report_lines(results, skipped, args.apply, args.summary_only):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
