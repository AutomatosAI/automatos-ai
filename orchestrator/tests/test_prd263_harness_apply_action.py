"""PRD-263 US-052 — one apply action, and it acts only on a live grant over an unchanged setting.

``platform_harness_apply`` (handlers_harness.harness_apply) is the only HARNESS path that
changes an agent setting (FR-11). The owner's click on a change's Approvals card resumes it
with the card's ``{"rx_id": ...}``. It loads the proposal's harness_task_ledger row in the
workspace; needs the row's grant, the tool_call grant for exactly this call, granted and
not past ``expires_at`` (``is_authorising``), else the row ends ``expired``; needs the
target setting, read through the change kind's prior-value reader, to still read the row's
``prior_value``, else ``drifted``; moves the row ``pending → applying`` with a guarded
UPDATE, so a second call changes nothing; and runs the kind's inner action as the
workspace's Auto, resolved server-side and never from ``params['_agent_id']``, through
PlatformActionExecutor. There the hierarchy check passes on system_actor_bypass: the
direct apply HARNESS made before PRD-263 named no actor and was refused as an
``anonymous_actor``. The row ends ``applied`` (with ``applied_at`` and the value read) or
``failed``. A grant still pending, declined or withdrawn changes nothing, so a chat or
heartbeat call applies nothing the owner has not approved.

The action is a write with no card (the click is the consent), and the hierarchy audit
lists it on ALLOW_LIST with its reason (US-004, #1150).

Everything runs on the real database (``db_session``, rolled back), with the rate limiter
stubbed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy import text

from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_grants as hgx

ORCH = Path(__file__).resolve().parents[1]
RX = "rx-1-001"
EXPIRED = "This approval expired; nothing changed"
DRIFTED = "This setting changed since the proposal; nothing changed"
NOT_GRANTED = "This change has no granted approval on its Approvals card; nothing changed"
NO_AUTO = "this workspace has no Auto to make it"
# What an inner handler's exception says that no owner should read.
INTERNAL_DETAIL = "server closed the connection unexpectedly (pool worker 7)"


@pytest.fixture
def crew(db_session, seed_workspace):
    return hgx.team(db_session, seed_workspace(name="harness-apply"))


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("core.security.rate_limiter.check_rate_limit", AsyncMock(return_value=None))


@pytest.fixture
def decisions(monkeypatch):
    """Every hierarchy decision the executor makes, as can_actor_modify returned it."""
    from modules.tools.discovery import platform_executor

    seen = []
    real = platform_executor.can_actor_modify

    def recording(db, **kwargs):
        decision = real(db, **kwargs)
        seen.append(decision)
        return decision

    monkeypatch.setattr(platform_executor, "can_actor_modify", recording)
    return seen


def _apply(db, crew, params=None):
    from modules.tools.discovery.handlers_harness import harness_apply

    return asyncio.run(harness_apply(db, UUID(crew.ws), params or {"rx_id": RX}))


def _unchanged(db, crew, ledger_id, state="pending"):
    assert hgx.row(db, ledger_id)["state"] == state
    assert hgx.interval(db, crew.scout) == hgx.PRIOR


# ── the click on a live grant applies, as Auto ───────────────────────────────


def test_a_granted_grant_over_the_prior_value_applies_as_auto(db_session, crew, decisions):
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = _apply(db_session, crew)

    assert reply["success"] is True and reply["rx_id"] == RX, reply
    assert hgx.interval(db_session, crew.scout) == hgx.PROPOSED
    row = hgx.row(db_session, ledger_id)
    assert row["state"] == "applied" and row["applied_at"] is not None
    assert row["value_at_apply"] == {"interval_minutes": hgx.PRIOR}
    # The inner call named Auto: no anonymous_actor refusal, the system-actor bypass.
    assert [(d.allowed, d.reason, d.actor_agent_id) for d in decisions] == [
        (True, "system_actor_bypass", crew.auto)]


def test_an_agent_id_in_the_params_is_ignored(db_session, crew, decisions):
    """An ordinary agent would be refused Scout's heartbeat; the handler acts as Auto whoever called."""
    intern = hdx.agent(db_session, crew.ws, "Intern")
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = _apply(db_session, crew, {"rx_id": RX, "_agent_id": intern})

    assert reply["success"] is True, reply
    assert hgx.row(db_session, ledger_id)["state"] == "applied"
    assert [d.actor_agent_id for d in decisions] == [crew.auto]


# ── a late, stale or second click changes nothing ────────────────────────────


def test_an_expired_grant_records_expired_and_changes_nothing(db_session, crew, decisions):
    ledger_id, _ = hgx.proposal(db_session, crew, expires_in=hgx.LAPSED)

    assert _apply(db_session, crew) == {"success": False, "error": EXPIRED}
    _unchanged(db_session, crew, ledger_id, "expired")
    assert decisions == []


def test_a_setting_edited_after_the_proposal_records_drifted_and_changes_nothing(db_session, crew, decisions):
    ledger_id, _ = hgx.proposal(db_session, crew)
    hgx.set_interval(db_session, crew.scout, 30)

    assert _apply(db_session, crew) == {"success": False, "error": DRIFTED}
    assert hgx.row(db_session, ledger_id)["state"] == "drifted"
    assert hgx.interval(db_session, crew.scout) == 30
    assert decisions == []


def test_a_double_call_changes_once(db_session, crew, decisions):
    ledger_id, _ = hgx.proposal(db_session, crew)

    first, second = _apply(db_session, crew), _apply(db_session, crew)

    assert first["success"] is True
    assert second == {"success": False, "error": "This change is already applied; nothing changed"}
    assert hgx.row(db_session, ledger_id)["state"] == "applied"
    assert hgx.interval(db_session, crew.scout) == hgx.PROPOSED
    assert len(decisions) == 1  # the inner action ran once


@pytest.mark.parametrize("status", ["pending", "denied", "revoked"])
def test_a_grant_that_was_never_granted_changes_nothing(db_session, crew, decisions, status):
    ledger_id, grant_id = hgx.proposal(db_session, crew, status=status)

    assert _apply(db_session, crew) == {"success": False, "error": NOT_GRANTED}
    _unchanged(db_session, crew, ledger_id)
    assert hgx.grant_status(db_session, grant_id) == status
    assert decisions == []


def test_no_grant_or_another_calls_grant_changes_nothing(db_session, crew, decisions):
    other_call = hgx.grant(db_session, crew.ws, RX, subject_params={"rx_id": "rx-9-009"})
    with_other = hgx.ledger(db_session, crew.ws, RX, crew.scout, other_call)
    without = hgx.ledger(db_session, crew.ws, "rx-1-002", crew.scout, None)

    for rx_id, ledger_id in ((RX, with_other), ("rx-1-002", without)):
        assert _apply(db_session, crew, {"rx_id": rx_id}) == {"success": False, "error": NOT_GRANTED}
        _unchanged(db_session, crew, ledger_id)
    assert decisions == []


def test_another_workspaces_proposal_is_not_found(db_session, crew, seed_workspace, decisions):
    elsewhere = hgx.team(db_session, seed_workspace(name="harness-apply-elsewhere"))
    ledger_id, _ = hgx.proposal(db_session, elsewhere)

    reply = _apply(db_session, crew)

    assert reply == {"success": False, "error": f"This workspace has no HARNESS change {RX}; nothing changed"}
    assert hgx.row(db_session, ledger_id)["state"] == "pending"
    assert hgx.interval(db_session, elsewhere.scout) == hgx.PRIOR


@pytest.mark.parametrize("caller_context", [None, {"driving_user_id": "1"}], ids=["heartbeat", "chat"])
def test_a_chat_or_heartbeat_call_without_a_granted_grant_changes_nothing(db_session, crew, decisions,
                                                                          caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ledger_id, grant_id = hgx.proposal(db_session, crew, status="pending")

    reply = asyncio.run(PlatformActionExecutor(db_session, UUID(crew.ws)).execute(
        "platform_harness_apply", {"rx_id": RX, "_agent_id": crew.auto}, caller_context))

    assert reply == {"success": False, "error": NOT_GRANTED}
    _unchanged(db_session, crew, ledger_id)
    assert hgx.grant_status(db_session, grant_id) == "pending"
    assert decisions == []


def test_an_inner_call_that_fails_records_failed_and_changes_nothing(db_session, crew, monkeypatch):
    from modules.tools.discovery import platform_executor

    async def refuses(db, workspace_id, params):
        return {"success": False, "error": "Agent 'Scout' could not be updated"}

    monkeypatch.setattr(platform_executor, "configure_agent_heartbeat", refuses)
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = _apply(db_session, crew)

    assert reply == {"success": False,
                     "error": "The change could not be made (Agent 'Scout' could not be updated); nothing changed"}
    row = hgx.row(db_session, ledger_id)
    assert row["state"] == "failed" and row["entry"] == {"reason": "Agent 'Scout' could not be updated"}
    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR


def test_a_workspace_with_no_auto_records_failed_and_changes_nothing(db_session, seed_workspace, decisions):
    crew = hgx.team(db_session, seed_workspace(name="harness-apply-no-auto"), with_auto=False)
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = _apply(db_session, crew)

    assert reply == {"success": False, "error": f"The change could not be made ({NO_AUTO}); nothing changed"}
    row = hgx.row(db_session, ledger_id)
    assert row["state"] == "failed" and row["entry"] == {"reason": NO_AUTO}
    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR
    assert decisions == []


def test_an_inner_call_that_raises_records_failed_from_pending_without_its_detail(db_session, crew, monkeypatch):
    """The executor answers a raising handler with its own message and rolls the session
    back, the uncommitted ``pending → applying`` move with it: the row still ends
    ``failed`` (moved from ``pending``), and the exception's text reaches neither the card
    nor the ledger, only the server log."""
    from modules.tools.discovery import platform_executor

    async def raises(db, workspace_id, params):
        raise RuntimeError(INTERNAL_DETAIL)

    ledger_id, _ = hgx.proposal(db_session, crew)

    def rollback():  # what the real rollback does to this row: its applying move is undone
        db_session.execute(text("UPDATE harness_task_ledger SET state = 'pending' WHERE id = :id"),
                           {"id": ledger_id})

    monkeypatch.setattr(platform_executor, "configure_agent_heartbeat", raises)
    monkeypatch.setattr(db_session, "rollback", rollback)

    reply = _apply(db_session, crew)

    raised = "Action 'platform_configure_agent_heartbeat' failed"
    assert reply == {"success": False, "error": f"The change could not be made ({raised}); nothing changed"}
    row = hgx.row(db_session, ledger_id)
    assert row["state"] == "failed" and row["entry"] == {"reason": raised}
    assert INTERNAL_DETAIL not in f"{reply} {row['entry']}"
    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR


def test_a_made_change_whose_row_left_applying_is_never_reported_applied(db_session, crew, monkeypatch):
    """Should the row leave ``applying`` while the inner change runs, the card says the
    change was made, the server log names the row, and nothing records it ``applied``."""
    from services import harness_grants

    real_move = harness_grants.move

    def row_gone(db, row, to_state, **kwargs):
        return False if to_state == "applied" else real_move(db, row, to_state, **kwargs)

    monkeypatch.setattr(harness_grants, "move", row_gone)
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = _apply(db_session, crew)

    assert reply == {"success": False,
                     "error": "The change was made, but its HARNESS ledger entry could not be marked applied"}
    assert hgx.row(db_session, ledger_id)["state"] == "applying"
    assert hgx.interval(db_session, crew.scout) == hgx.PROPOSED


# ── the inner change runs only inside platform_harness_apply ─────────────────


def test_the_inner_change_runs_only_inside_platform_harness_apply(db_session, crew, monkeypatch):
    from modules.tools.discovery import platform_executor

    callers = []

    async def heartbeat(db, workspace_id, params):
        callers.append({frame.function for frame in inspect.stack()})
        return {"success": True}

    monkeypatch.setattr(platform_executor, "configure_agent_heartbeat", heartbeat)
    hgx.proposal(db_session, crew)

    assert _apply(db_session, crew)["success"] is True
    (stack,) = callers
    assert {"harness_apply", "_apply_as_auto", "_run_inner", "_run_cleared"} <= stack
    # Outside the tests, only the registry's own validator and this handler build a change's call.
    builders = sorted(str(path.relative_to(ORCH)) for path in ORCH.rglob("*.py")
                      if "tests" not in path.parts and ".params_builder(" in path.read_text(errors="ignore"))
    assert builders == ["modules/tools/discovery/handlers_harness.py", "services/harness_changes.py"]


# ── registration and the hierarchy audit ─────────────────────────────────────


def _gate():
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def test_the_action_is_a_write_with_no_card_mapped_and_allow_listed_with_its_reason():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.handlers_harness import harness_apply
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    action = get_action_registry().get("platform_harness_apply")
    assert (action.permission_level, action.requires_confirmation, action.admin_only) == ("write", False, False)
    assert action.parameters["required"] == ["rx_id"]
    assert PLATFORM_HANDLERS["platform_harness_apply"] is harness_apply
    assert "platform_harness_apply" in _gate().ALLOW_LIST
    source = (ORCH / "scripts" / "check_hierarchy_gate.py").read_text()
    reason = source[source.index("PRD-263 US-004"):source.index('"platform_harness_apply"')]
    assert "system_actor_bypass" in reason and "rate limit" in reason
    ran = subprocess.run([sys.executable, str(ORCH / "scripts" / "check_hierarchy_gate.py")],
                         capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout + ran.stderr
