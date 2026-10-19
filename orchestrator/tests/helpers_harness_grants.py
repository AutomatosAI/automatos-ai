"""Seeding the governed apply rail on the real database, for the PRD-263 P3 suites.

A workspace's Auto and one agent, Scout, with an enabled heartbeat every PRIOR minutes;
a harness_task_ledger row for a heartbeat_tune proposal (PRIOR → PROPOSED minutes); and
its tool_call approval grant for platform_harness_apply on ``{"rx_id": ...}``, the call
US-054 raises. Rows go into the transactional ``db_session`` and are rolled back.
Timestamps are relative to the real clock, because the apply handler reads it.
Shared by test_prd263_harness_apply_action.py and test_prd263_harness_reconcile_sweep.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import text

from tests import helpers_harness_diagnose as hdx

PRIOR, PROPOSED = 60, 120
APPLY = "platform_harness_apply"
HEARTBEAT_TUNE = "heartbeat_tune"
# A live grant has hours left; a lapsed one ran out an hour ago.
LIVE, LAPSED = timedelta(hours=23), timedelta(hours=-1)
# The statuses a click (or a revoke after it) leaves a granted_at on.
CLICKED = ("granted", "revoked")


def now() -> datetime:
    return datetime.now(timezone.utc)


def team(db, ws: str) -> SimpleNamespace:
    """The workspace's Auto and Scout, whose heartbeat runs every PRIOR minutes."""
    auto = hdx.auto(db, ws)
    scout = hdx.agent(db, ws, "Scout", configuration={"heartbeat": {"enabled": True, "interval_minutes": PRIOR}})
    return SimpleNamespace(ws=ws, auto=auto, scout=scout)


def grant(db, ws: str, rx_id: str, status: str = "granted", *, expires_in: timedelta = LIVE,
          granted_ago: Optional[timedelta] = timedelta(minutes=1), details: Optional[Dict[str, Any]] = None,
          subject_params: Optional[Dict[str, Any]] = None) -> int:
    """The card's tool_call grant for platform_harness_apply on ``{"rx_id": rx_id}``
    (``subject_params`` raises it on other params instead); returns its id."""
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant
    from modules.tools.execution.tool_grants import tool_call_subject_id

    moment = now()
    row = ApprovalGrant(
        workspace_id=UUID(ws), subject_type=SUBJECT_TOOL_CALL, tool_name=APPLY, status=status,
        subject_id=tool_call_subject_id(ws, APPLY, subject_params or {"rx_id": rx_id}),
        requested_at=moment - timedelta(hours=1), expires_at=moment + expires_in,
        granted_at=(moment - granted_ago) if granted_ago is not None and status in CLICKED else None,
        details=details if details is not None else {"action": APPLY, "params": {"rx_id": rx_id}},
    )
    db.add(row)
    db.flush()
    return row.id


def ledger(db, ws: str, rx_id: str, agent_id: int, grant_id: Optional[int], *, state: str = "pending",
           kind: str = HEARTBEAT_TUNE, prior: Optional[Dict[str, Any]] = None,
           proposed: Optional[Dict[str, Any]] = None, updated_at: Optional[datetime] = None) -> int:
    """The proposal's ledger row; returns its id."""
    from core.models.harness import HarnessTaskLedger

    row = HarnessTaskLedger(
        workspace_id=UUID(ws), state=state, rx_id=rx_id, run_id=1, grant_id=grant_id, kind=kind,
        target_agent_id=agent_id, prior_value=prior or {"interval_minutes": PRIOR},
        proposed_value=proposed or {"interval_minutes": PROPOSED},
    )
    if updated_at is not None:
        row.updated_at = updated_at
    db.add(row)
    db.flush()
    return row.id


def proposal(db, crew: SimpleNamespace, rx_id: str = "rx-1-001", status: str = "granted", **grant_kw: Any):
    """A pending heartbeat_tune proposal for Scout with its grant: (ledger id, grant id)."""
    grant_id = grant(db, crew.ws, rx_id, status, **grant_kw)
    return ledger(db, crew.ws, rx_id, crew.scout, grant_id), grant_id


def row(db, ledger_id: int) -> Dict[str, Any]:
    """The ledger row as the database holds it (moves are guarded UPDATEs the ORM does not see)."""
    return dict(db.execute(text("SELECT * FROM harness_task_ledger WHERE id = :id"), {"id": ledger_id})
                .mappings().one())


def grant_status(db, grant_id: int) -> str:
    return db.execute(text("SELECT status FROM approval_grants WHERE id = :id"), {"id": grant_id}).scalar()


def interval(db, agent_id: int) -> int:
    """The agent's heartbeat interval as stored."""
    configuration = db.execute(text("SELECT configuration FROM agents WHERE id = :id"), {"id": agent_id}).scalar()
    return configuration["heartbeat"]["interval_minutes"]


def set_interval(db, agent_id: int, minutes: int) -> None:
    """An owner's own edit of the agent's heartbeat interval (the session forgets what it loaded)."""
    db.execute(text("UPDATE agents SET configuration = CAST(jsonb_set(CAST(configuration AS jsonb), "
                    "'{heartbeat,interval_minutes}', to_jsonb(CAST(:m AS int))) AS json) WHERE id = :id"),
               {"m": minutes, "id": agent_id})
    db.expire_all()
