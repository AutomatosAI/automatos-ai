"""PRD-163 S3 — approval policy engine.

Pure: the decision matrix (policy × under/over ceiling × gate on/off × override).
Integration: countdown auto-proceed fires and is cancelable.
"""

from __future__ import annotations

import os
import sys
import types
import uuid

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402

import core.services.approval_policy as ap  # noqa: E402


def _set_policy(monkeypatch, policy, ceiling=None, countdown=None):
    monkeypatch.setattr(
        ap, "load_approval_policy",
        lambda db, ws: {"policy": policy, "approval_dollar_ceiling": ceiling,
                        "auto_proceed_after_seconds": countdown},
    )


def _set_gate(monkeypatch, on):
    import core.services.auto_autonomy as aa
    monkeypatch.setattr(aa, "is_full_autonomy", lambda db, ws: on)


class TestDecisionMatrix:
    def test_always_ask_never_auto(self, monkeypatch):
        _set_policy(monkeypatch, ap.ALWAYS_ASK)
        d = ap.evaluate_approval(None, "ws", 0.01)
        assert d.auto_approve is False

    def test_auto_below_budget_under_ceiling_approves(self, monkeypatch):
        _set_policy(monkeypatch, ap.AUTO_BELOW_BUDGET, ceiling=5.0)
        d = ap.evaluate_approval(None, "ws", 3.0)
        assert d.auto_approve is True

    def test_auto_below_budget_over_ceiling_asks(self, monkeypatch):
        _set_policy(monkeypatch, ap.AUTO_BELOW_BUDGET, ceiling=5.0)
        d = ap.evaluate_approval(None, "ws", 7.0)
        assert d.auto_approve is False

    def test_full_auto_with_gate_on_approves(self, monkeypatch):
        _set_policy(monkeypatch, ap.FULL_AUTO)
        _set_gate(monkeypatch, True)
        d = ap.evaluate_approval(None, "ws", 999.0)
        assert d.auto_approve is True

    def test_full_auto_without_gate_asks(self, monkeypatch):
        """Fail-safe: full_auto must NOT run unsupervised without the §12.3 gate."""
        _set_policy(monkeypatch, ap.FULL_AUTO)
        _set_gate(monkeypatch, False)
        d = ap.evaluate_approval(None, "ws", 0.01)
        assert d.auto_approve is False

    def test_per_request_override_forces_auto(self, monkeypatch):
        _set_policy(monkeypatch, ap.ALWAYS_ASK)
        d = ap.evaluate_approval(None, "ws", 999.0, override_auto_approve=True)
        assert d.auto_approve is True

    def test_countdown_carried_on_ask(self, monkeypatch):
        _set_policy(monkeypatch, ap.ALWAYS_ASK, countdown=30)
        d = ap.evaluate_approval(None, "ws", 1.0)
        assert d.auto_approve is False
        assert d.countdown_seconds == 30
        # audit snapshot carries the policy + ceiling + cost
        snap = d.audit_snapshot()
        assert snap["policy"] == ap.ALWAYS_ASK and snap["auto_approved"] is False


class TestSetPolicyValidation:
    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError):
            ap.set_approval_policy(None, "ws", policy="banana")


# --------------------------------------------------------------------------- #
# Integration — countdown fires + cancelable
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_countdown_auto_proceeds_when_elapsed(db_session, seed_workspace):
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import text
    from services.coordinator_service import CoordinatorService
    from core.models.orchestration_enums import RunState

    ws = seed_workspace()
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()

    def _seed(deadline):
        return db_session.execute(
            text(
                "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type, config) "
                "VALUES (CAST(:ws AS uuid), 'g', 'user_x', :st, 'initial', CAST(:cfg AS jsonb)) RETURNING id"
            ),
            {"ws": ws, "st": RunState.AWAITING_APPROVAL.value,
             "cfg": f'{{"approval_deadline_at": "{deadline}"}}'},
        ).scalar()

    expired = _seed(past)
    pending = _seed(future)
    db_session.flush()

    n = CoordinatorService().check_approval_countdowns(db_session, uuid.UUID(ws))
    assert n == 1   # only the expired one proceeded

    from core.models.orchestration import OrchestrationRun
    assert db_session.query(OrchestrationRun).get(expired).state == RunState.RUNNING.value
    assert db_session.query(OrchestrationRun).get(pending).state == RunState.AWAITING_APPROVAL.value
