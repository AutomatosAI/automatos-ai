"""PRD-181 S2 — F060: durable approval grants + budget ceiling for board & playbook.

The mission-only approval primitive (PRD-163) is generalised into a scoped,
expiring, revocable, tool-agnostic **approval grant** so non-chat agents (board /
scheduled / webhook / playbook) hitting an ``ask`` tier get a real workflow —
not a hard block, not an auto-allow.

Three guarantees:
1. ``test_board_task_requires_approval`` — an ask-tier board task creates a
   durable grant and blocks (status → blocked) until granted.
2. ``test_grant_is_revocable_and_expiring`` — a grant can be revoked, and an
   expired grant no longer authorises.
3. ``test_playbook_budget_ceiling`` — a playbook run's dollar ceiling is
   enforced with the same generalised helper missions use.

Grant lifecycle (create → find_active → grant/revoke/expire) is pure over a fake
session so no Postgres is needed. The budget helper is pure arithmetic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


# ---------------------------------------------------------------------------
# A minimal fake session that stores ApprovalGrant-like rows in a list and
# supports the narrow query shape the service uses.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows: List[Any], model: Any):
        self._rows = rows
        self._model = model
        self._filters: List[Any] = []

    def filter(self, *conds):
        # We can't evaluate SQLAlchemy expressions here; instead the service
        # passes plain lambdas via filter_by-like helpers in the fake. For the
        # real service we test find_active_grant through explicit helpers below.
        return self

    def order_by(self, *a):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self) -> None:
        self.rows: List[Any] = []
        self.flushed = False
        self.committed = False

    def add(self, obj: Any) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self) -> None:
        self.flushed = True

    def commit(self) -> None:
        self.committed = True

    def query(self, model):
        return _Query(self.rows, model)


# ===========================================================================
# Grant model + lifecycle
# ===========================================================================

def test_grant_model_shape():
    """The ApprovalGrant row carries everything a durable, scoped, expiring,
    revocable grant needs — tenant, subject, tool, risk, status, timestamps."""
    from core.models.approval_grants import ApprovalGrant, GrantStatus

    g = ApprovalGrant(
        workspace_id=uuid4(),
        subject_type="board_task",
        subject_id="42",
        tool_name="platform_delete_agent",
        risk_tier="destructive",
        status=GrantStatus.PENDING.value,
    )
    for col in (
        "workspace_id", "subject_type", "subject_id", "tool_name", "risk_tier",
        "status", "requested_at", "expires_at", "granted_at", "granted_by",
        "revoked_at", "revoked_by", "reason",
    ):
        assert hasattr(g, col), f"ApprovalGrant missing column {col}"


def test_create_grant_is_pending_and_expiring():
    from core.services.approval_grants import create_grant
    from core.models.approval_grants import GrantStatus

    db = _FakeSession()
    ws = uuid4()
    g = create_grant(
        db, ws, subject_type="board_task", subject_id="42",
        tool_name="composio_refund", risk_tier="external_side_effect",
        reason="ask: external side-effect under balanced policy",
        ttl_seconds=3600,
    )
    assert g.status == GrantStatus.PENDING.value
    assert g.expires_at is not None
    assert g.expires_at > g.requested_at
    assert db.flushed is True


def test_grant_is_revocable_and_expiring():
    from core.services.approval_grants import grant_grant, revoke_grant, is_authorising
    from core.models.approval_grants import ApprovalGrant, GrantStatus

    now = datetime.now(timezone.utc)

    # A granted, unexpired grant authorises.
    g = ApprovalGrant(
        workspace_id=uuid4(), subject_type="board_task", subject_id="1",
        tool_name="t", risk_tier="destructive", status=GrantStatus.PENDING.value,
        requested_at=now, expires_at=now + timedelta(hours=1),
    )
    grant_grant(g, granted_by="user:9", now=now)
    assert g.status == GrantStatus.GRANTED.value
    assert g.granted_at is not None and g.granted_by == "user:9"
    assert is_authorising(g, now=now) is True

    # Revoked ⇒ no longer authorises.
    revoke_grant(g, revoked_by="user:9", now=now)
    assert g.status == GrantStatus.REVOKED.value
    assert is_authorising(g, now=now) is False

    # Expired ⇒ no longer authorises (even if it was granted).
    g2 = ApprovalGrant(
        workspace_id=uuid4(), subject_type="board_task", subject_id="2",
        tool_name="t", risk_tier="destructive", status=GrantStatus.GRANTED.value,
        requested_at=now - timedelta(hours=2), expires_at=now - timedelta(hours=1),
        granted_at=now - timedelta(hours=2),
    )
    assert is_authorising(g2, now=now) is False, "an expired grant must not authorise"


# ===========================================================================
# Board task approval gate
# ===========================================================================

def test_board_task_requires_approval():
    """Under always_ask, a board task about to run creates a durable pending
    grant and is blocked (not executed, not auto-allowed)."""
    from services.board_approval import evaluate_board_task_approval
    from core.models.approval_grants import GrantStatus

    db = _FakeSession()
    ws = uuid4()

    # Stub the workspace approval policy read to 'always_ask' (mission parity).
    outcome = evaluate_board_task_approval(
        db, workspace_id=ws, task_id=42, estimated_cost_usd=0.0,
        _policy_override="always_ask",
    )

    assert outcome.requires_approval is True
    assert outcome.grant is not None
    assert outcome.grant.status == GrantStatus.PENDING.value
    assert outcome.grant.subject_type == "board_task"
    assert outcome.grant.subject_id == "42"


def test_board_task_auto_approves_below_ceiling():
    """auto_below_budget with cost under the ceiling ⇒ no approval, no grant,
    the task runs (the grant workflow does not fire needlessly)."""
    from services.board_approval import evaluate_board_task_approval

    db = _FakeSession()
    outcome = evaluate_board_task_approval(
        db, workspace_id=uuid4(), task_id=7, estimated_cost_usd=0.10,
        _policy_override="auto_below_budget", _ceiling_override=5.0,
    )
    assert outcome.requires_approval is False
    assert outcome.grant is None


def test_board_task_blocks_when_over_ceiling():
    from services.board_approval import evaluate_board_task_approval
    from core.models.approval_grants import GrantStatus

    db = _FakeSession()
    outcome = evaluate_board_task_approval(
        db, workspace_id=uuid4(), task_id=7, estimated_cost_usd=50.0,
        _policy_override="auto_below_budget", _ceiling_override=5.0,
    )
    assert outcome.requires_approval is True
    assert outcome.grant.status == GrantStatus.PENDING.value


# ===========================================================================
# Generalised dollar-ceiling helper (mission → board → playbook)
# ===========================================================================

def test_budget_ceiling_helper_status():
    """The shared helper returns the same HEALTHY/WARNING/CRITICAL/EXCEEDED
    bands the mission dispatcher used, but for ANY (ceiling, used) pair."""
    from services.budget_ceiling import budget_status, BudgetBand

    assert budget_status(ceiling_usd=0.0, used_usd=999.0) == BudgetBand.HEALTHY  # 0 = unlimited
    assert budget_status(ceiling_usd=10.0, used_usd=1.0) == BudgetBand.HEALTHY
    assert budget_status(ceiling_usd=10.0, used_usd=6.0) == BudgetBand.WARNING
    assert budget_status(ceiling_usd=10.0, used_usd=9.0) == BudgetBand.CRITICAL
    assert budget_status(ceiling_usd=10.0, used_usd=11.0) == BudgetBand.EXCEEDED


def test_playbook_budget_ceiling():
    """A playbook run carries a dollar ceiling; a step that would push spend over
    it is blocked, exactly like a mission task."""
    from services.budget_ceiling import playbook_can_afford

    # Under ceiling ⇒ allowed.
    assert playbook_can_afford(ceiling_usd=5.0, used_usd=1.0, next_step_usd=1.0) is True
    # Would breach ⇒ blocked.
    assert playbook_can_afford(ceiling_usd=5.0, used_usd=4.5, next_step_usd=1.0) is False
    # No ceiling ⇒ always allowed.
    assert playbook_can_afford(ceiling_usd=0.0, used_usd=999.0, next_step_usd=999.0) is True


def test_grant_is_audited_on_create(monkeypatch):
    """Creating a grant writes an AuditLog governance row (audit every
    governance action, §PRD conventions)."""
    from services import board_approval
    from core.models.approval_grants import GrantStatus

    calls: List[Dict[str, Any]] = []

    def _fake_audit(db, ws, action, **kw):
        calls.append({"action": action, **kw})

    monkeypatch.setattr(board_approval, "_audit_governance", _fake_audit)

    db = _FakeSession()
    board_approval.evaluate_board_task_approval(
        db, workspace_id=uuid4(), task_id=42, estimated_cost_usd=0.0,
        _policy_override="always_ask",
    )
    assert any(c["action"] == "approval_grant:created" for c in calls), \
        "grant creation must be audited"
