"""HARNESS HTTP API: the Command Center Self-Learning tile's endpoint.

``GET /self-learning`` (PRD-263 US-021, US-058) feeds the tile: counts, coverage and
mode only, with how many of HARNESS's proposals are pending, applied, declined and
expired, readable by any member of the workspace (the US-002 decision, maintainer
issue #1148).

The approve and reject routes for queued HARNESS prescriptions are gone (PRD-263
US-047, maintainer issue #1149). A change HARNESS proposes applies only on the owner's
click on its Approvals card (US-052, US-054), never through a route here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.composio_cache import ToolExecutionLog
from core.models.harness import LEDGER_APPLIED, LEDGER_DENIED, LEDGER_EXPIRED, LEDGER_PENDING
from core.models.workspaces import Workspace
from services.harness_metrics import PRODUCTION_TELEMETRY
from services.harness_mode import harness_mode

if TYPE_CHECKING:
    from core.models.harness import HarnessRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/harness", tags=["HARNESS"])

# PRD-263 US-021, the US-002 decision (#1148, built at its proposed default until the
# decision is recorded): every member of the workspace reads the tile's counts,
# coverage and mode, and nobody reads grant text through it. agents:read is the
# permission every workspace role holds (modules/policy/roles.py); the local
# edition's anonymous caller reads as the owner.
SELF_LEARNING_PERMISSION = "agents:read"
# The Signals figure counts this workspace's production tool calls over this window.
SIGNALS_WINDOW = timedelta(days=7)
# The response's sections; one that fails to load is named in ``unavailable``.
SECTION_HARNESS = "harness"
SECTION_SIGNALS = "signals"
SECTION_LEDGER = "ledger"
# The latest run's figures the tile shows, read from its harness_runs row; the rest of
# the row (report, suggestions, verdict details) stays server-side.
RECORDED_FIGURES = ("coverage", "insufficient_data", "watch", "regression")
RUN_FIGURES = ("iteration_count",) + RECORDED_FIGURES
HARNESS_FIELDS = ("status",) + RUN_FIGURES + ("mode",)
SIGNALS_FIELDS = ("signals_7d",)
# PRD-263 US-058: what became of HARNESS's proposals, as harness_task_ledger counts by
# state (a pending row by its effective state, US-053; a declined card is ``denied``).
LEDGER_FIELDS = (LEDGER_PENDING, LEDGER_APPLIED, LEDGER_DENIED, LEDGER_EXPIRED)


@router.get("/self-learning", dependencies=[Depends(require_workspace_permission(SELF_LEARNING_PERMISSION))])
def self_learning_health(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Self-learning health for the Command Center tile (PRD-263 US-021, FR-18).

    HARNESS's ``status`` as get_status reads it, the latest harness_runs row's
    ``iteration_count``, ``coverage`` and ``insufficient_data``, ``watch`` and
    ``regression`` counts, the workspace's HARNESS ``mode``, ``signals_7d``: this
    workspace's production tool calls over the last 7 days, and the ``pending``,
    ``applied``, ``denied`` and ``expired`` ledger counts (US-058). Counts only, no
    grant or report text, and nothing is written. A plain ``def`` over the synchronous
    session, so FastAPI runs it in its threadpool and the event loop never waits on
    the database.

    A figure with no value is None, never 0. A section that fails to load is
    logged, its figures are None and its name is in ``unavailable``, so the tile
    shows "—" for it instead of failing whole.
    """
    ws = ctx.workspace_id
    harness, harness_ok = _load_section(db, SECTION_HARNESS, HARNESS_FIELDS, lambda: _harness_fields(db, ws))
    signals, signals_ok = _load_section(db, SECTION_SIGNALS, SIGNALS_FIELDS, lambda: {"signals_7d": _signals_7d(db, ws)})
    ledger, ledger_ok = _load_section(db, SECTION_LEDGER, LEDGER_FIELDS, lambda: _ledger_counts(db, ws))
    loaded = ((SECTION_HARNESS, harness_ok), (SECTION_SIGNALS, signals_ok), (SECTION_LEDGER, ledger_ok))
    unavailable = [name for name, ok in loaded if not ok]
    return {"workspace_id": str(ws), **harness, **signals, **ledger, "unavailable": unavailable}


def _load_section(
    db: Session, name: str, fields: Tuple[str, ...], read: Callable[[], Dict[str, Any]]
) -> Tuple[Dict[str, Any], bool]:
    """``read()`` and True; on a failure (logged, the transaction rolled back so the
    next section can read) the section's fields as None, and False."""
    try:
        return read(), True
    except Exception:
        logger.exception("[HARNESS] The self-learning %s section could not be read", name)
        db.rollback()
        return {field: None for field in fields}, False


def _harness_fields(db: Session, workspace_id: UUID) -> Dict[str, Any]:
    """HARNESS's status (get_status, which reads harness_runs and the lock), the latest
    run's figures and the workspace's mode."""
    from services import harness_runs
    from services.harness_service import get_harness_service

    status = get_harness_service().get_status(workspace_id, db=db).get("status")
    workspace = db.get(Workspace, workspace_id)
    mode = harness_mode(workspace.settings if workspace is not None else None)
    return {"status": status, **_run_figures(harness_runs.latest_run(db, workspace_id)), "mode": mode}


def _run_figures(run: Optional["HarnessRun"]) -> Dict[str, Any]:
    """The latest run's iteration, coverage and verdict counts; None for any it did not
    record, and for all of them before the first run (get_status says 0 there)."""
    from services import harness_diagnose

    if run is None:
        return {field: None for field in RUN_FIGURES}
    recorded = harness_diagnose.status_fields(run.coverage, run.counts, run.verdicts)
    return {"iteration_count": run.iteration, **{field: recorded.get(field) for field in RECORDED_FIGURES}}


def _signals_7d(db: Session, workspace_id: UUID) -> int:
    """This workspace's production tool calls over SIGNALS_WINDOW, from tool_execution_logs."""
    # executed_at is stored as naive UTC (datetime.utcnow), so the bound is naive UTC too.
    since = datetime.now(timezone.utc).replace(tzinfo=None) - SIGNALS_WINDOW
    count = (
        db.query(func.count(ToolExecutionLog.id))
        .filter(
            ToolExecutionLog.workspace_id == workspace_id,
            ToolExecutionLog.telemetry_source == PRODUCTION_TELEMETRY,
            ToolExecutionLog.executed_at >= since,
        )
        .scalar()
    )
    return int(count or 0)


def _ledger_counts(db: Session, workspace_id: UUID) -> Dict[str, int]:
    """How many of the workspace's HARNESS proposals read as pending, applied, denied and
    expired (services.harness_grants.ledger_counts, which writes nothing); 0 for a state
    with no row."""
    from services import harness_grants

    counts = harness_grants.ledger_counts(db, workspace_id)
    return {state: counts.get(state, 0) for state in LEDGER_FIELDS}
