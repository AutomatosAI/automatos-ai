"""HARNESS handler functions for PlatformActionExecutor (PRD-121)."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.harness import RUN_FAILED

if TYPE_CHECKING:
    from core.models.harness import HarnessRun

logger = logging.getLogger(__name__)

# platform_harness_history: runs listed when Auto names no limit, and the most it lists.
DEFAULT_HISTORY_LIMIT = 10
MAX_HISTORY_LIMIT = 52
HISTORY_UNAVAILABLE = "The HARNESS run history could not be read. Try again shortly."
# What Auto hears when a call fails: the detail goes to the server log, never into chat.
STATUS_UNAVAILABLE = "The HARNESS status could not be read. Try again shortly."
TRIGGER_UNAVAILABLE = "The HARNESS review could not be started. Try again shortly."
# A trigger only schedules the run: a run already under way in another worker makes it a no-op.
TRIGGERED_MESSAGE = (
    "HARNESS review triggered. Like every run it is report-only: it judges the last "
    "complete ISO week, files the audit report in Reports and changes nothing."
)


async def harness_status(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return current HARNESS state."""
    from services.harness_service import get_harness_service

    try:
        status = get_harness_service().get_status(workspace_id, db=db)
        return {"success": True, "data": status}
    except Exception:
        logger.exception("[HARNESS] harness_status failed for %s", workspace_id)
        return {"success": False, "error": STATUS_UNAVAILABLE}


async def harness_trigger(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Manually trigger a HARNESS run. A workspace whose mode is disabled runs
    nothing, so the reply says that instead of reporting a run (PRD-263 US-016)."""
    from services.harness_mode import DISABLED
    from services.harness_service import DISABLED_MESSAGE, get_harness_service

    try:
        service = get_harness_service()
        if not service.workspace_enabled(db, workspace_id):
            return {"success": True, "data": {"status": DISABLED, "message": DISABLED_MESSAGE}}
        await service.trigger_now(workspace_id)
        return {"success": True, "data": {"message": TRIGGERED_MESSAGE}}
    except Exception:
        logger.exception("[HARNESS] harness_trigger failed for %s", workspace_id)
        return {"success": False, "error": TRIGGER_UNAVAILABLE}


async def harness_history(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """List the workspace's HARNESS runs, newest first, from harness_runs (PRD-263 US-023)."""
    from services import harness_runs

    try:
        limit = _history_limit(params.get("limit"))
        runs = [_run_summary(run) for run in harness_runs.recent_runs(db, workspace_id, limit)]
    except Exception:
        logger.exception("[HARNESS] harness_history failed for %s", workspace_id)
        return {"success": False, "error": HISTORY_UNAVAILABLE}
    return {"success": True, "data": {"runs": runs, "total": len(runs)}}


def _history_limit(raw: Any) -> int:
    """The ``limit`` Auto asked for, as an int in 1..MAX_HISTORY_LIMIT (default when unreadable)."""
    if isinstance(raw, bool):
        return DEFAULT_HISTORY_LIMIT
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_HISTORY_LIMIT
    return min(max(limit, 1), MAX_HISTORY_LIMIT)


def _run_summary(run: "HarnessRun") -> Dict[str, Any]:
    """One run as Auto reads it: the week it judged, how it ended, coverage and the
    verdict counts (None where the run recorded none, never 0), and its suggestions."""
    from services import harness_diagnose, harness_runs

    counts = run.counts if isinstance(run.counts, dict) else {}
    outcome = (
        {"error": harness_runs.failure_reason(run.report_status)}
        if run.status == RUN_FAILED
        else {"audit_report": run.report_status}
    )
    return {
        "cur_window": run.cur_window,
        "status": run.status,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "iteration": run.iteration,
        **harness_diagnose.status_fields(run.coverage, run.counts, run.verdicts),
        "prescription_count": counts.get(harness_runs.PRESCRIPTION_COUNT),
        # Only a run imported from before PRD-263 applied anything; a report-only run reads 0.
        "applied_count": counts.get(harness_runs.IMPORTED_APPLIED, 0),
        **outcome,
    }


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.isoformat() if moment else None
