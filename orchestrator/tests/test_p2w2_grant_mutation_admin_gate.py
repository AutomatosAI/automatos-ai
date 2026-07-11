"""PRD-196 S2 (P2-15, governance C.8) — arm the door before opening it.

Until this change the grant-mutation endpoints depended only on
``get_request_context_hybrid``: ANY workspace member could authorise an
agent's destructive action by POSTing grant/deny/revoke. And nothing told a
human that a grant was pending — a blocked board task waited silently.

Pinned here:
- every approval-grant endpoint (list + the three mutations) carries the
  canonical ``require_workspace_admin`` dependency (PRD-185 S12) — plain
  members 403, workspace owner/admin and the super-admin pass;
- creating a fresh pending grant dispatches exactly one ``approval_pending``
  notification (grant id / subject / risk in the payload), the reuse branch
  does not re-spam, and denial semantics are untouched;
- ``approval_pending`` is part of the dispatcher's event vocabulary.

Pure: fake ctx + MagicMock sessions (the PRD-185 S12 test shape); the
notification dispatch is captured at the module seam — no DB, no network,
no live dispatcher.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import approval_grants as grants_api  # noqa: E402
from core.auth.workspace_admin import require_workspace_admin  # noqa: E402
from services import board_approval  # noqa: E402


def _ctx(*, system_role="user", role="user", clerk_user_id=None, workspace_id=None):
    """A minimal RequestContext-shaped fake (the PRD-185 S12 shape)."""
    user = SimpleNamespace(
        id="u", email=None, role=role, system_role=system_role, clerk_user_id=clerk_user_id
    )
    return SimpleNamespace(user=user, workspace_id=workspace_id or uuid.uuid4())


# ---------------------------------------------------------------------------
# The gate is wired: every approval-grant route depends on require_workspace_admin
# ---------------------------------------------------------------------------

def _dependant_calls(dependant) -> set:
    calls = {getattr(dependant, "call", None)}
    for sub in getattr(dependant, "dependencies", []) or []:
        calls |= _dependant_calls(sub)
    return calls


def test_every_grant_route_carries_workspace_admin_gate():
    expected = {
        ("/api/v1/approval-grants", "GET"),
        ("/api/v1/approval-grants/{grant_id}/grant", "POST"),
        ("/api/v1/approval-grants/{grant_id}/deny", "POST"),
        ("/api/v1/approval-grants/{grant_id}/revoke", "POST"),
    }
    seen = set()
    for route in grants_api.router.routes:
        for method in route.methods or ():
            key = (route.path, method)
            if key in expected:
                assert require_workspace_admin in _dependant_calls(route.dependant), (
                    f"{method} {route.path} is not gated by require_workspace_admin"
                )
                seen.add(key)
    assert seen == expected, f"missing routes: {expected - seen}"


def test_plain_member_403s_and_workspace_admin_passes():
    """Drive the dependency itself: no owner/admin membership row ⇒ 403;
    super-admin (and any principal may_see_own_workspace_health accepts) ⇒ ctx."""
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None  # plain member — no admin row
    with pytest.raises(HTTPException) as ei:
        asyncio.run(require_workspace_admin(ctx=_ctx(clerk_user_id="clerk_1"), db=db))
    assert ei.value.status_code == 403

    ctx = _ctx(system_role="super_admin")
    assert asyncio.run(require_workspace_admin(ctx=ctx, db=MagicMock())) is ctx

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (1,)  # owner/admin membership row
    ctx = _ctx(clerk_user_id="clerk_1")
    assert asyncio.run(require_workspace_admin(ctx=ctx, db=db)) is ctx


# ---------------------------------------------------------------------------
# Pending-grant creation dispatches approval_pending (and only fresh creation)
# ---------------------------------------------------------------------------

@pytest.fixture
def capture_dispatch(monkeypatch):
    calls = []

    async def _fake_dispatch(workspace_id, grant_id, subject_id, risk_tier, reason, estimated_cost_usd):
        calls.append(
            {
                "workspace_id": workspace_id,
                "grant_id": grant_id,
                "subject_id": subject_id,
                "risk_tier": risk_tier,
                "reason": reason,
                "estimated_cost_usd": estimated_cost_usd,
            }
        )

    monkeypatch.setattr(board_approval, "_dispatch_approval_pending", _fake_dispatch)
    monkeypatch.setattr(board_approval, "_audit_governance", lambda *a, **k: None)
    return calls


def test_pending_grant_dispatches_notification(monkeypatch, capture_dispatch):
    ws = uuid.uuid4()
    fake_grant = SimpleNamespace(id=7, subject_id="42")
    monkeypatch.setattr("core.services.approval_grants.find_pending_grant", lambda *a, **k: None)
    monkeypatch.setattr("core.services.approval_grants.create_grant", lambda *a, **k: fake_grant)

    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(),
        workspace_id=ws,
        task_id=42,
        estimated_cost_usd=1.25,
        risk_tier="high",
        _policy_override="always_ask",
    )

    assert outcome.requires_approval is True
    assert len(capture_dispatch) == 1, "exactly one approval_pending per fresh grant"
    call = capture_dispatch[0]
    assert call["workspace_id"] == str(ws)
    assert call["grant_id"] == 7
    assert call["subject_id"] == "42"
    assert call["risk_tier"] == "high"
    assert call["estimated_cost_usd"] == 1.25


def test_existing_pending_grant_does_not_renotify(monkeypatch, capture_dispatch):
    existing = SimpleNamespace(id=5, subject_id="42")
    monkeypatch.setattr("core.services.approval_grants.find_pending_grant", lambda *a, **k: existing)
    monkeypatch.setattr(
        "core.services.approval_grants.create_grant",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse, not create")),
    )

    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=42, _policy_override="always_ask",
    )

    assert outcome.requires_approval is True
    assert outcome.grant is existing
    assert capture_dispatch == [], "the reuse branch must not re-spam admins"


def test_notification_fault_never_wedges_the_gate(monkeypatch):
    """A dispatch/scheduling fault is swallowed — approval still asks."""
    monkeypatch.setattr("core.services.approval_grants.find_pending_grant", lambda *a, **k: None)
    monkeypatch.setattr(
        "core.services.approval_grants.create_grant",
        lambda *a, **k: SimpleNamespace(id=9, subject_id="7"),
    )
    monkeypatch.setattr(board_approval, "_audit_governance", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("dispatcher down")

    monkeypatch.setattr(board_approval, "_dispatch_approval_pending", _boom)

    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=7, _policy_override="always_ask",
    )
    assert outcome.requires_approval is True


def test_deny_does_not_requeue():
    """Guard the existing denial semantics while touching the file: a denied
    subject fails — it is never returned to the dispatch queue."""
    task = SimpleNamespace(status="blocked", error_message=None, completed_at=None)
    db = MagicMock()
    db.query.return_value.get.return_value = task
    grant = SimpleNamespace(
        subject_type="board_task", subject_id="11", workspace_id=uuid.uuid4()
    )

    grants_api._fail_subject(db, grant)

    assert task.status == "failed"
    assert task.error_message == "Approval denied by a human reviewer"
    assert task.completed_at is not None


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_approval_pending_is_a_valid_event_type():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert "approval_pending" in VALID_EVENT_TYPES
