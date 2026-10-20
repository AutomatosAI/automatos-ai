"""PRD-263 US-052/US-053: the governed apply rail's ledger, and lapsed cards closed within a day.

Each owner's Approvals card is a tool_call grant for ``platform_harness_apply`` on
``{"rx_id": ...}`` (US-054) with its ``harness_task_ledger`` row (US-051). :func:`move` is
the only way a row changes state, never out of a final one (FR-15); the apply handler
(US-052) reads rows, grants and settings here too. :func:`reconcile` ends each open row
by its grant (:func:`outcome`) at the start of a HARNESS tick (ORG-2) and in the daily
sweep, never on a read; a revoke after the change applied leaves it ``applied``, since a
revoke undoes nothing. :func:`ledger_counts` is the status read.
No platform action, no tool executor and no model call (FR-12).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union
from uuid import UUID

from sqlalchemy import func

from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant, GrantStatus
from core.models.harness import (
    LEDGER_APPLYING,
    LEDGER_DENIED,
    LEDGER_DRIFTED,
    LEDGER_EXPIRED,
    LEDGER_FAILED,
    LEDGER_PENDING,
    LEDGER_WITHDRAWN,
    HarnessTaskLedger,
)

if TYPE_CHECKING:
    from apscheduler.triggers.cron import CronTrigger
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The action every HARNESS Approvals card approves, and its one parameter (US-052, US-054).
APPLY_ACTION = "platform_harness_apply"
RX_PARAM = "rx_id"
# The rows reconciliation reads: a proposal waiting for its click, and one being applied.
OPEN_STATES = (LEDGER_PENDING, LEDGER_APPLYING)
# Why a row closed, kept in its ``entry`` under REASON_KEY.
REASON_KEY = "reason"
CARD_BLOCKED = "evidence card was blocked; nothing changed"
NO_OUTCOME = "no outcome"
INTERRUPTED = "interrupted"
NO_GRANT = "no approval grant"
# What the approval API records when an Approve re-queued the evidence card, dragged to
# Blocked, instead of running the call (api/approval_grants._resume_tool_call).
BOARD_REQUEUE = "board_task_requeue"
# Who closed a lapsed or drifted grant (its revoked_by), and the daily sweep's job id.
SWEEP_ACTOR = "system:harness"
SWEEP_JOB_ID = "harness_grant_sweep"


@dataclass(frozen=True)
class Outcome:
    """Where reconciliation moves an open row, why, and whether its PENDING grant closes first."""

    state: str
    reason: Optional[str] = None
    close_grant: bool = False


def _now(now: Optional[datetime] = None) -> datetime:
    return now or datetime.now(timezone.utc)


def _utc(moment: datetime) -> datetime:
    """A stored timestamp as an aware UTC one (a naive one is read as UTC)."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _workspace(workspace_id: Union[UUID, str]) -> UUID:
    return UUID(str(workspace_id))


def resolve_auto_agent(db: "Session", workspace_id: Union[UUID, str]) -> Optional[Dict[str, Any]]:
    """The workspace's Auto, {"id", "name"} or None, who signs the audit report and makes each
    approved change (US-052): the system agent ``auto-<workspace>``, else one named Auto, else any."""
    from core.models import Agent

    slug = f"auto-{workspace_id}"
    candidates = [
        (Agent.slug == slug, Agent.is_system_agent.is_(True)),
        (Agent.is_system_agent.is_(True), Agent.name == "Auto"),
        (Agent.is_system_agent.is_(True),),
    ]
    for clauses in candidates:
        agent = db.query(Agent).filter(Agent.workspace_id == workspace_id, *clauses).first()
        if agent is not None:
            return {"id": agent.id, "name": agent.name}
    return None


# ------------------------------------------------------------------ rows, grants and values


def ledger_row(db: "Session", workspace_id: Union[UUID, str], rx_id: str) -> Optional[HarnessTaskLedger]:
    """The workspace's ledger row for one prescription id, as the database holds it now
    (a move is an UPDATE the session's copy does not see), or None."""
    return (db.query(HarnessTaskLedger).populate_existing()
            .filter(HarnessTaskLedger.workspace_id == _workspace(workspace_id), HarnessTaskLedger.rx_id == rx_id)
            .first())


def _binds(grant: ApprovalGrant, row: HarnessTaskLedger) -> bool:
    """Whether ``grant`` is the tool_call grant for exactly platform_harness_apply on this
    row's rx_id: the subject the card's grant was raised with (tool_grants)."""
    from modules.tools.execution.tool_grants import tool_call_subject_id

    subject = tool_call_subject_id(row.workspace_id, APPLY_ACTION, {RX_PARAM: row.rx_id})
    return grant.subject_type == SUBJECT_TOOL_CALL and grant.subject_id == subject


def _bound_grants(db: "Session", workspace_id: UUID, rows: List[HarnessTaskLedger]) -> Dict[int, ApprovalGrant]:
    """Each row's approval grant by row id: only its own card's grant, in this workspace."""
    ids = {row.grant_id for row in rows if row.grant_id is not None}
    if not ids:
        return {}
    grants = {grant.id: grant for grant in db.query(ApprovalGrant).populate_existing()
              .filter(ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.id.in_(ids))}
    return {row.id: grants[row.grant_id] for row in rows
            if row.grant_id in grants and _binds(grants[row.grant_id], row)}


def grant_of(db: "Session", row: HarnessTaskLedger) -> Optional[ApprovalGrant]:
    """The row's approval grant (see _bound_grants), or None."""
    return _bound_grants(db, row.workspace_id, [row]).get(row.id)


def current_value(db: "Session", row: HarnessTaskLedger) -> Optional[Dict[str, Any]]:
    """The row's target setting now, read by its change kind's prior-value reader; None when
    the kind applies nothing, the agent is not in the workspace or the change cannot be read."""
    from services.harness_changes import KINDS

    kind = KINDS.get(row.kind)
    if kind is None or not kind.applies or row.target_agent_id is None:
        return None
    try:
        return kind.prior_value_reader(db, row.workspace_id, row.target_agent_id, row.proposed_value or {})
    except ValueError:
        logger.warning("[HARNESS] Ledger row %s holds a change its kind cannot read", row.id)
        return None


def unchanged(current: Optional[Mapping[str, Any]], row: HarnessTaskLedger) -> bool:
    """Whether the setting still reads the row's prior value (a setting that cannot be
    read is never unchanged)."""
    return current is not None and current == row.prior_value


# ------------------------------------------------------------------ moves


def move(db: "Session", row: HarnessTaskLedger, to_state: str, *, from_state: str = LEDGER_PENDING,
         reason: Optional[str] = None, now: Optional[datetime] = None, **columns: Any) -> bool:
    """Move ``row`` to ``to_state`` only while it is still ``from_state`` (a guarded UPDATE: a
    second caller matches no row), keeping ``reason`` in its ``entry`` and setting
    ``columns``. True when this call moved it. Not committed: the caller owns the transaction."""
    values: Dict[Any, Any] = {HarnessTaskLedger.state: to_state, HarnessTaskLedger.updated_at: _now(now)}
    values.update({getattr(HarnessTaskLedger, name): value for name, value in columns.items()})
    if reason:
        entry = row.entry if isinstance(row.entry, dict) else {}
        values[HarnessTaskLedger.entry] = {**entry, REASON_KEY: reason}
    moved = (db.query(HarnessTaskLedger)
             .filter(HarnessTaskLedger.id == row.id, HarnessTaskLedger.workspace_id == row.workspace_id,
                     HarnessTaskLedger.state == from_state)
             .update(values, synchronize_session=False))
    return bool(moved)


# ------------------------------------------------------------------ reconciliation


def _executed(grant: ApprovalGrant) -> Optional[Mapping[str, Any]]:
    """What the click's resume recorded on the grant (``details.executed_result``), or None."""
    details = grant.details if isinstance(grant.details, dict) else {}
    result = details.get("executed_result")
    return result if isinstance(result, dict) else None


def _older(moment: Optional[datetime], now: datetime, grace: timedelta) -> bool:
    """Whether ``moment`` is more than ``grace`` before ``now`` (no moment counts as older)."""
    return moment is None or now - _utc(moment) > grace


def _lapsed(grant: ApprovalGrant, now: datetime) -> bool:
    return grant.expires_at is not None and now >= _utc(grant.expires_at)


def _after_the_click(grant: ApprovalGrant, now: datetime, grace: timedelta) -> Optional[Outcome]:
    """A clicked grant whose row is still pending is ``failed`` when the click's resume ran
    and recorded no outcome ("evidence card was blocked; nothing changed" when the Approve
    re-queued the card, section 11-Q30), or when it was granted longer than the grace ago.
    Otherwise the click may still be running."""
    result = _executed(grant)
    if result is not None:
        return Outcome(LEDGER_FAILED, CARD_BLOCKED if result.get("resumed_via") == BOARD_REQUEUE else NO_OUTCOME)
    if _older(grant.granted_at, now, grace):
        return Outcome(LEDGER_FAILED, NO_OUTCOME)
    return None


def outcome(row: HarnessTaskLedger, grant: Optional[ApprovalGrant], now: datetime,
            grace: timedelta) -> Optional[Outcome]:
    """Where reconciliation moves an open row by its grant and age, or None when it stays:
    grant DENIED is ``denied``; EXPIRED, or PENDING past ``expires_at`` (closed first), is
    ``expired``; REVOKED with no ``executed_result`` is ``withdrawn``; a clicked grant is
    ``failed`` once the click recorded no outcome (_after_the_click). An ``applying`` row
    older than HARNESS_RECONCILE_GRACE_MINUTES is ``failed`` ("interrupted")."""
    if row.state == LEDGER_APPLYING:
        return Outcome(LEDGER_FAILED, INTERRUPTED) if _older(row.updated_at, now, grace) else None
    if grant is None:
        return Outcome(LEDGER_FAILED, NO_GRANT)
    status = grant.status
    if status == GrantStatus.DENIED.value:
        return Outcome(LEDGER_DENIED)
    if status == GrantStatus.EXPIRED.value:
        return Outcome(LEDGER_EXPIRED)
    if status == GrantStatus.PENDING.value:
        return Outcome(LEDGER_EXPIRED, close_grant=True) if _lapsed(grant, now) else None
    if status == GrantStatus.REVOKED.value and _executed(grant) is None:
        return Outcome(LEDGER_WITHDRAWN)
    return _after_the_click(grant, now, grace)


def _drift_outcome(db: "Session", row: HarnessTaskLedger, grant: Optional[ApprovalGrant]) -> Optional[Outcome]:
    """A waiting card (pending row, PENDING grant) over a setting that no longer reads its
    prior value is closed and ``drifted``."""
    waiting = row.state == LEDGER_PENDING and grant is not None and grant.status == GrantStatus.PENDING.value
    if waiting and not unchanged(current_value(db, row), row):
        return Outcome(LEDGER_DRIFTED, close_grant=True)
    return None


def _apply_outcome(db: "Session", workspace_id: UUID, row: HarnessTaskLedger, decided: Outcome,
                   now: datetime) -> bool:
    """Close the row's PENDING grant first when asked (one decided a moment ago keeps the row
    as it is), then move the row."""
    if decided.close_grant:
        from core.services.approval_grants import expire_pending_grants

        if not expire_pending_grants(db, workspace_id, [row.grant_id], revoked_by=SWEEP_ACTOR, now=now):
            return False
    return move(db, row, decided.state, from_state=row.state, reason=decided.reason, now=now)


def _reconcile(db: "Session", workspace_id: UUID, now: datetime) -> Dict[str, int]:
    from config import config

    grace = timedelta(minutes=config.HARNESS_RECONCILE_GRACE_MINUTES)
    rows = (db.query(HarnessTaskLedger).populate_existing()
            .filter(HarnessTaskLedger.workspace_id == workspace_id, HarnessTaskLedger.state.in_(OPEN_STATES))
            .order_by(HarnessTaskLedger.id).all())
    grants = _bound_grants(db, workspace_id, rows)
    moved: Counter = Counter()
    for row in rows:
        grant = grants.get(row.id)
        decided = outcome(row, grant, now, grace) or _drift_outcome(db, row, grant)
        if decided is not None and _apply_outcome(db, workspace_id, row, decided, now):
            moved[decided.state] += 1
    return dict(moved)


def reconcile(db: "Session", workspace_id: Union[UUID, str], now: Optional[datetime] = None) -> Optional[Dict[str, int]]:
    """Reconcile the workspace's open ledger rows with their grants and commit. Returns how
    many rows moved to each state, or None when it failed (logged, rolled back, retried by
    the next tick or sweep)."""
    try:
        moved = _reconcile(db, _workspace(workspace_id), _now(now))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("[HARNESS] Could not reconcile the HARNESS ledger of %s", workspace_id)
        return None
    if moved:
        logger.info("[HARNESS] Reconciled the HARNESS ledger of %s: %s", workspace_id, moved)
    return moved


# ------------------------------------------------------------------ the status read


def effective_state(row: HarnessTaskLedger, grant: Optional[ApprovalGrant], now: Optional[datetime] = None) -> str:
    """A row's state as the status read shows it, derived without writing: a pending row
    whose grant was declined or withdrawn, or no longer authorises (``is_authorising``,
    or PENDING past ``expires_at``), reads as reconciliation will record it."""
    from core.services.approval_grants import is_authorising

    if row.state != LEDGER_PENDING or grant is None:
        return row.state
    moment, status = _now(now), grant.status
    if status == GrantStatus.DENIED.value:
        return LEDGER_DENIED
    if status == GrantStatus.REVOKED.value and _executed(grant) is None:
        return LEDGER_WITHDRAWN
    if status == GrantStatus.EXPIRED.value or (status == GrantStatus.PENDING.value and _lapsed(grant, moment)):
        return LEDGER_EXPIRED
    if status == GrantStatus.GRANTED.value and not is_authorising(grant, now=moment):
        return LEDGER_EXPIRED
    return LEDGER_PENDING


def ledger_counts(db: "Session", workspace_id: Union[UUID, str], now: Optional[datetime] = None) -> Dict[str, int]:
    """How many of the workspace's ledger rows read as each state, pending rows by their
    effective state. Reads only: nothing is moved, closed or committed."""
    ws = _workspace(workspace_id)
    stored = (db.query(HarnessTaskLedger.state, func.count(HarnessTaskLedger.id))
              .filter(HarnessTaskLedger.workspace_id == ws, HarnessTaskLedger.state != LEDGER_PENDING)
              .group_by(HarnessTaskLedger.state).all())
    pending = (db.query(HarnessTaskLedger).populate_existing()
               .filter(HarnessTaskLedger.workspace_id == ws, HarnessTaskLedger.state == LEDGER_PENDING).all())
    grants = _bound_grants(db, ws, pending)
    counts = Counter({state: count for state, count in stored})
    counts.update(effective_state(row, grants.get(row.id), now) for row in pending)
    return dict(counts)


# ------------------------------------------------------------------ the daily sweep


def sweep_trigger() -> "CronTrigger":
    """The daily sweep's trigger: ``config.HARNESS_GRANT_SWEEP_CRON`` in UTC. ValueError for a bad cron."""
    from config import config
    from services.schedule_util import scheduler_cron_trigger

    return scheduler_cron_trigger(config.HARNESS_GRANT_SWEEP_CRON, timezone=timezone.utc)


def schedule_sweep(scheduler: Any) -> None:
    """Register the daily sweep on the shared scheduler the HARNESS service runs on."""
    from config import config

    scheduler.add_job(run_daily_sweep, trigger=sweep_trigger(), id=SWEEP_JOB_ID, replace_existing=True,
                      max_instances=1)
    logger.info("[HARNESS] Registered the daily grant sweep (cron: %s UTC)", config.HARNESS_GRANT_SWEEP_CRON)


def sweep(db: "Session", now: Optional[datetime] = None) -> Dict[str, int]:
    """Reconcile every workspace with an open ledger row (one that fails does not stop the
    rest); returns the moves summed."""
    workspaces: List[UUID] = [ws for (ws,) in db.query(HarnessTaskLedger.workspace_id)
                              .filter(HarnessTaskLedger.state.in_(OPEN_STATES)).distinct()]
    totals: Counter = Counter()
    for workspace_id in workspaces:
        totals.update(reconcile(db, workspace_id, now) or {})
    logger.info("[HARNESS] Daily grant sweep: %d workspace(s), %s", len(workspaces), dict(totals))
    return dict(totals)


def run_daily_sweep() -> Dict[str, int]:
    """The daily job, on a session of its own."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return sweep(db)
    finally:
        db.close()

