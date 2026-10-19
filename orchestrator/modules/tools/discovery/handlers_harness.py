"""HARNESS handler functions for PlatformActionExecutor (PRD-121)."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.approval_grants import GrantStatus
from core.models.harness import (
    LEDGER_APPLIED,
    LEDGER_APPLYING,
    LEDGER_DRIFTED,
    LEDGER_EXPIRED,
    LEDGER_FAILED,
    LEDGER_PENDING,
    RUN_FAILED,
)

if TYPE_CHECKING:
    from core.models.harness import HarnessRun, HarnessTaskLedger
    from services.harness_changes import InnerCall

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
# platform_harness_apply (PRD-263 US-052): what the owner's card shows when the click
# changes nothing, and what the ledger keeps of why.
APPLY_EXPIRED = "This approval expired; nothing changed"
APPLY_DRIFTED = "This setting changed since the proposal; nothing changed"
APPLY_NOT_GRANTED = "This change has no granted approval on its Approvals card; nothing changed"
APPLY_DECIDED = "This change is already {state}; nothing changed"
APPLY_UNKNOWN = "This workspace has no HARNESS change {rx_id}; nothing changed"
APPLY_FAILED = "The change could not be made ({error}); nothing changed"
APPLY_UNAVAILABLE = "The HARNESS change could not be applied. Try again shortly."
APPLIED_MESSAGE = "Applied the approved HARNESS change {rx_id} ({kind}) to agent #{agent_id}."
NO_AUTO = "this workspace has no Auto to make it"
INNER_RAISED = "the change raised an error"
# How much of a failed inner call's error the ledger and the card keep.
ERROR_CHARS = 200
# A grant granted, or closed as lapsed, goes on to the expiry check; one still pending,
# declined or withdrawn was never approved.
APPROVED_OR_LAPSED = (GrantStatus.GRANTED.value, GrantStatus.EXPIRED.value)


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


async def harness_apply(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Make one HARNESS change its owner approved (PRD-263 US-052), the only HARNESS path that
    changes an agent setting (FR-11). The ledger row of ``rx_id`` must be ``pending``, its
    card's grant granted and unexpired (else ``expired``) and the setting still its
    ``prior_value`` (else ``drifted``); ``pending → applying`` is a guarded UPDATE, so a
    second call changes nothing; the change runs as the workspace's Auto and the row ends
    ``applied`` or ``failed``. A grant never granted changes nothing (reconciliation records
    its end). Not committed: the approval API's resume commits with the grant's result.
    """
    rx_id = params.get("rx_id") if isinstance(params, dict) else None
    if not isinstance(rx_id, str) or not rx_id:
        return {"success": False, "error": "Missing required parameter: rx_id"}
    try:
        return await _apply(db, workspace_id, rx_id)
    except Exception:
        logger.exception("[HARNESS] platform_harness_apply failed for %s (%s)", workspace_id, rx_id)
        return {"success": False, "error": APPLY_UNAVAILABLE}


def _nothing_changed(message: str) -> Dict[str, Any]:
    return {"success": False, "error": message}


async def _apply(db: Session, workspace_id: UUID, rx_id: str) -> Dict[str, Any]:
    from core.services.approval_grants import is_authorising
    from services import harness_grants

    row = harness_grants.ledger_row(db, workspace_id, rx_id)
    if row is None:
        return _nothing_changed(APPLY_UNKNOWN.format(rx_id=rx_id))
    if row.state != LEDGER_PENDING:
        return _nothing_changed(APPLY_DECIDED.format(state=row.state))
    grant = harness_grants.grant_of(db, row)
    if grant is None or grant.status not in APPROVED_OR_LAPSED:
        return _nothing_changed(APPLY_NOT_GRANTED)
    if not is_authorising(grant):
        harness_grants.move(db, row, LEDGER_EXPIRED)
        return _nothing_changed(APPLY_EXPIRED)
    current = harness_grants.current_value(db, row)
    if not harness_grants.unchanged(current, row):
        harness_grants.move(db, row, LEDGER_DRIFTED)
        return _nothing_changed(APPLY_DRIFTED)
    if not harness_grants.move(db, row, LEDGER_APPLYING):
        return _nothing_changed(APPLY_DECIDED.format(state=LEDGER_APPLYING))
    return await _apply_as_auto(db, workspace_id, row, current)


async def _apply_as_auto(
    db: Session, workspace_id: UUID, row: "HarnessTaskLedger", current: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the change kind's inner action as the workspace's Auto, resolved here and
    never from the caller's ``_agent_id``, and record ``applied`` or ``failed``."""
    from services import harness_changes, harness_grants

    auto = harness_grants.resolve_auto_agent(db, workspace_id)
    if auto is None:
        return _failed(db, row, NO_AUTO)
    try:
        call = harness_changes.KINDS[row.kind].params_builder(row.target_agent_id, row.proposed_value)
    except ValueError as exc:  # a stored value the kind refuses: nothing to run
        return _failed(db, row, str(exc)[:ERROR_CHARS])
    result = await _run_inner(db, workspace_id, call, auto["id"])
    if result.get("success") is not True:
        return _failed(db, row, str(result.get("error") or harness_grants.NO_OUTCOME)[:ERROR_CHARS])
    harness_grants.move(db, row, LEDGER_APPLIED, from_state=LEDGER_APPLYING,
                        applied_at=datetime.now(timezone.utc), value_at_apply=current)
    return {
        "success": True,
        "rx_id": row.rx_id,
        "agent_id": row.target_agent_id,
        "message": APPLIED_MESSAGE.format(rx_id=row.rx_id, kind=row.kind, agent_id=row.target_agent_id),
    }


async def _run_inner(db: Session, workspace_id: UUID, call: "InnerCall", auto_id: int) -> Dict[str, Any]:
    """The inner platform action through PlatformActionExecutor with Auto as its actor
    (``_agent_id``), so the admin gate, the hierarchy check (Auto passes on
    system_actor_bypass), the rate limit and the destructive backstop run on it. It has
    no caller context: the owner's click was the consent, so the owner-only and
    confirmation gates do not ask again."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    try:
        result = await PlatformActionExecutor(db, workspace_id).execute(
            call.action, {**call.params, "_agent_id": auto_id})
    except Exception:
        logger.exception("[HARNESS] %s raised for a HARNESS change in %s", call.action, workspace_id)
        return {"success": False, "error": INNER_RAISED}
    return result if isinstance(result, dict) else {"success": False}


def _failed(db: Session, row: "HarnessTaskLedger", reason: str) -> Dict[str, Any]:
    """Record the row ``failed`` with its reason, from ``applying``; or from ``pending`` when
    the inner call's handler raised and the executor rolled the session back, ``applying``
    with it. The card says nothing changed."""
    from services import harness_grants

    if not harness_grants.move(db, row, LEDGER_FAILED, from_state=LEDGER_APPLYING, reason=reason):
        harness_grants.move(db, row, LEDGER_FAILED, reason=reason)
    return _nothing_changed(APPLY_FAILED.format(error=reason))
