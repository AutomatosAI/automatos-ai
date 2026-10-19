"""HARNESS self-management HTTP API — PRD-142 Wave 4 (W4-S1).

Authenticated approve/reject endpoints for queued high-risk HARNESS
prescriptions. The US-024 escalation points a workspace admin **here** (the
Command Center), never to a channel reply: the inbound webhook
(``api/webhooks.py``) performs **no sender authorization** — possession of the
URL-as-secret is its only credential — so a channel command can't be trusted to
mutate state. Approval happens on this authenticated, tenant-scoped surface,
where the calling principal is known and re-checked as a workspace admin.

The approve/reject logic itself lives in ``api.harness_commands`` (the handler
the design always intended, with its own fail-closed admin gate). This router
only (1) resolves the authenticated principal to the integer ``users.id`` that
gate expects, and (2) maps the handler's result to HTTP status codes. The flag
``HARNESS_SELF_MANAGEMENT_ENABLED`` still governs whether anything happens.

Approval is **human-admin-only by design**: a principal that doesn't resolve to
an active owner/admin ``users`` row — e.g. an API-key / service principal, which
carries no clerk id or email — fails closed (403). The platform never grants a
service account self-management authority over itself.

``GET /self-learning`` (PRD-263 US-021) feeds the Command Center Self-Learning
tile: counts, coverage and mode only, readable by any member of the workspace
(the US-002 decision, maintainer issue #1148).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.harness_commands import handle_harness_command
from core.auth.actor import resolve_internal_user_id
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.composio_cache import ToolExecutionLog
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
# The latest run's figures the tile shows, read from its harness_runs row; the rest of
# the row (report, suggestions, verdict details) stays server-side.
RECORDED_FIGURES = ("coverage", "insufficient_data", "watch", "regression")
RUN_FIGURES = ("iteration_count",) + RECORDED_FIGURES
HARNESS_FIELDS = ("status",) + RUN_FIGURES + ("mode",)
SIGNALS_FIELDS = ("signals_7d",)


# The canonical resolver lives in core.auth.actor (extracted from the copy that
# used to sit here). Fail-closed semantics are unchanged: an unresolved
# principal yields None, and the handler's admin gate then refuses.
_resolve_internal_user_id = resolve_internal_user_id


async def _run_command(
    db: Session, ctx: RequestContext, command: str, rx_id: str
) -> Dict[str, Any]:
    """Resolve the caller, run the HARNESS command, map the result to HTTP."""
    user_id = _resolve_internal_user_id(db, ctx)
    # caller_identity carries the integer users.id; a None id fails the handler's
    # admin gate closed (→ 403), so an unresolved principal changes nothing.
    result = await handle_harness_command(
        db, ctx.workspace_id, command, rx_id, {"user_id": user_id}
    )
    if result.get("unauthorized"):
        raise HTTPException(status_code=403, detail=result.get("message"))
    if not result.get("success"):
        # Flag disabled, unknown/already-rejected rx, unresolved target, etc.
        # The message carries the reason; 409 keeps it distinct from an authz 403.
        raise HTTPException(status_code=409, detail=result.get("message"))
    return result


@router.post("/prescriptions/{rx_id}/approve", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def approve_prescription(
    rx_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve a queued HARNESS prescription — applies it now, audited + reversible."""
    return await _run_command(db, ctx, "/approve", rx_id)


@router.post("/prescriptions/{rx_id}/reject", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def reject_prescription(
    rx_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Reject a queued HARNESS prescription — suppresses future re-proposal."""
    return await _run_command(db, ctx, "/reject", rx_id)


@router.get("/self-learning", dependencies=[Depends(require_workspace_permission(SELF_LEARNING_PERMISSION))])
def self_learning_health(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Self-learning health for the Command Center tile (PRD-263 US-021, FR-18).

    HARNESS's ``status`` as get_status reads it, the latest harness_runs row's
    ``iteration_count``, ``coverage`` and ``insufficient_data``, ``watch`` and
    ``regression`` counts, the workspace's HARNESS ``mode``, and ``signals_7d``:
    this workspace's production tool calls over the last 7 days. Counts only, no
    grant or report text. A plain ``def`` over the synchronous session, so FastAPI
    runs it in its threadpool and the event loop never waits on the database.

    A figure with no value is None, never 0. A section that fails to load is
    logged, its figures are None and its name is in ``unavailable``, so the tile
    shows "—" for it instead of failing whole.
    """
    ws = ctx.workspace_id
    harness, harness_ok = _load_section(db, SECTION_HARNESS, HARNESS_FIELDS, lambda: _harness_fields(db, ws))
    signals, signals_ok = _load_section(db, SECTION_SIGNALS, SIGNALS_FIELDS, lambda: {"signals_7d": _signals_7d(db, ws)})
    unavailable = [name for name, ok in ((SECTION_HARNESS, harness_ok), (SECTION_SIGNALS, signals_ok)) if not ok]
    return {"workspace_id": str(ws), **harness, **signals, "unavailable": unavailable}


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
