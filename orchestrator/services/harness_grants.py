"""PRD-263 US-052: the governed apply rail's ledger rows and grants, read and moved safely.

Each owner's Approvals card is a tool_call grant for ``platform_harness_apply`` on
``{"rx_id": ...}`` (US-054) with its ``harness_task_ledger`` row (US-051). The
platform_harness_apply handler reads a row, its card's grant and the setting it changes
through here, and :func:`move` is the only way a row changes state, never out of a
final one (FR-15). :func:`resolve_auto_agent` names the agent HARNESS acts as.
No platform action, no tool executor and no model call (FR-12).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union
from uuid import UUID

from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant
from core.models.harness import LEDGER_PENDING, HarnessTaskLedger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The action every HARNESS Approvals card approves, and its one parameter (US-052, US-054).
APPLY_ACTION = "platform_harness_apply"
RX_PARAM = "rx_id"
# Why a row closed, kept in its ``entry`` under REASON_KEY.
REASON_KEY = "reason"
NO_OUTCOME = "no outcome"


def _now(now: Optional[datetime] = None) -> datetime:
    return now or datetime.now(timezone.utc)


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
