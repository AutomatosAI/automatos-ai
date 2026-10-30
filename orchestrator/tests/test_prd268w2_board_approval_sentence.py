"""PRD-268 US-C20b (§6.11.4, K4, K14, K16) — a held board task says its rule in words.

``services.board_approval.evaluate_board_task_approval`` (rewritten short over helpers, its
signature, ``BoardApprovalOutcome`` and the names tests patch kept) gives a task the approval
policy holds the English rule sentence for ``approval_policy.<policy>`` as its reason: on the
outcome, on the grant it creates, and on that grant's bell. The policy key it wrote before
("… under '<policy>' policy") is gone. The other outcome fields are unchanged, a task the policy
lets run still runs with no grant, and who may approve is untouched (only the words change).

``api/board_tasks.py`` still writes ``'Awaiting human approval (grant #N): ' + outcome.reason``,
so ``ticket_reasons`` still reads an approval, and the approval view reads the policy back from
the sentence.

Pure: the grant store, the audit and the bell are captured at their module seams; no DB.
"""
from __future__ import annotations

import inspect
import os
import sys
import uuid
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.services.approval_policy import (  # noqa: E402
    ALWAYS_ASK, AUTO_BELOW_BUDGET, FULL_AUTO, VALID_POLICIES, ApprovalDecision,
)
from modules.policy import rule_sentences as rules  # noqa: E402
from services import board_approval  # noqa: E402

COST = 0.42
GRANT = SimpleNamespace(id=31, subject_id="42")
APPROVAL_PREFIX = "Awaiting human approval (grant #{grant_id}): "


def _sentence(policy: str) -> str:
    """The English rule sentence the table holds for ``policy``."""
    return rules.STRINGS["en"][rules.approval_policy_rule(policy)]


@pytest.fixture
def seams(monkeypatch):
    """The grant store, the audit and the bell, captured; no pending grant waits."""
    seen = SimpleNamespace(created=[], audits=[], bells=[])

    def _create(db, workspace_id, **kwargs):
        seen.created.append(kwargs)
        return GRANT

    async def _bell(workspace_id, grant_id, subject_id, risk_tier, reason, estimated_cost_usd):
        seen.bells.append({"grant_id": grant_id, "subject_id": subject_id, "reason": reason})

    monkeypatch.setattr("core.services.approval_grants.find_pending_grant", lambda *a, **k: None)
    monkeypatch.setattr("core.services.approval_grants.create_grant", _create)
    monkeypatch.setattr(board_approval, "_audit_governance",
                        lambda db, ws, action, **details: seen.audits.append({"action": action, **details}))
    monkeypatch.setattr(board_approval, "_dispatch_approval_pending", _bell)
    return seen


def _held_by(monkeypatch, policy: str) -> None:
    """The workspace's policy, read through the mission primitive, holds the task."""
    decision = ApprovalDecision(False, "asks", policy, 0.10, COST, None)
    monkeypatch.setattr("core.services.approval_policy.evaluate_approval", lambda *a, **k: decision)


@pytest.mark.parametrize("policy", sorted(VALID_POLICIES))
def test_each_policy_that_holds_a_task_gives_its_sentence(monkeypatch, seams, policy):
    _held_by(monkeypatch, policy)

    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=42, estimated_cost_usd=COST, agent_id=7, risk_tier="high")

    assert outcome.reason == _sentence(policy)
    assert [c["reason"] for c in seams.created] == [_sentence(policy)]
    assert [b["reason"] for b in seams.bells] == [_sentence(policy)]


@pytest.mark.parametrize("policy", sorted(VALID_POLICIES))
def test_no_reason_holds_the_policy_key(monkeypatch, seams, policy):
    _held_by(monkeypatch, policy)

    outcome = board_approval.evaluate_board_task_approval(MagicMock(), workspace_id=uuid.uuid4(), task_id=42)

    said = [outcome.reason, *(c["reason"] for c in seams.created), *(b["reason"] for b in seams.bells)]
    assert said and all("policy" not in s.lower() and policy not in s for s in said)


def test_the_other_outcome_fields_are_unchanged(monkeypatch, seams):
    _held_by(monkeypatch, FULL_AUTO)
    ws = uuid.uuid4()

    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=ws, task_id=42, estimated_cost_usd=COST, agent_id=7, risk_tier="high",
        ttl_seconds=600)

    assert (outcome.requires_approval, outcome.grant, outcome.policy, outcome.estimated_cost_usd) == (
        True, GRANT, FULL_AUTO, COST)
    created = seams.created[0]
    assert {k: created[k] for k in ("subject_type", "subject_id", "risk_tier", "agent_id", "estimated_cost_usd",
                                    "ttl_seconds")} == {
        "subject_type": "board_task", "subject_id": "42", "risk_tier": "high", "agent_id": 7,
        "estimated_cost_usd": COST, "ttl_seconds": 600}
    assert seams.audits == [{"action": "approval_grant:created", "grant_id": GRANT.id, "subject_type": "board_task",
                             "subject_id": "42", "policy": FULL_AUTO, "estimated_cost_usd": COST,
                             "risk_tier": "high"}]
    assert [(b["grant_id"], b["subject_id"]) for b in seams.bells] == [(GRANT.id, "42")]


def test_the_test_seam_gives_the_same_sentences(seams):
    always = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=42, _policy_override=ALWAYS_ASK)
    over = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=42, estimated_cost_usd=COST,
        _policy_override=AUTO_BELOW_BUDGET, _ceiling_override=0.10)

    assert (always.reason, over.reason) == (_sentence(ALWAYS_ASK), _sentence(AUTO_BELOW_BUDGET))
    assert all(o.requires_approval and o.grant is GRANT for o in (always, over))


def test_a_reused_grant_gives_the_sentence_and_rings_no_second_bell(monkeypatch, seams):
    existing = SimpleNamespace(id=5, subject_id="42")
    monkeypatch.setattr("core.services.approval_grants.find_pending_grant", lambda *a, **k: existing)
    _held_by(monkeypatch, ALWAYS_ASK)

    outcome = board_approval.evaluate_board_task_approval(MagicMock(), workspace_id=uuid.uuid4(), task_id=42)

    assert (outcome.requires_approval, outcome.grant, outcome.reason) == (True, existing, _sentence(ALWAYS_ASK))
    assert seams.created == [] and seams.bells == [] and seams.audits == []


def test_a_policy_without_a_sentence_reads_the_fallback_never_its_key(monkeypatch, seams):
    _held_by(monkeypatch, "a_policy_from_tomorrow")

    outcome = board_approval.evaluate_board_task_approval(MagicMock(), workspace_id=uuid.uuid4(), task_id=42)

    assert outcome.reason == rules.STRINGS["en"][rules.FALLBACK]
    assert "a_policy_from_tomorrow" not in outcome.reason and outcome.policy == "a_policy_from_tomorrow"


def test_a_task_the_policy_lets_run_gets_no_grant(seams):
    outcome = board_approval.evaluate_board_task_approval(
        MagicMock(), workspace_id=uuid.uuid4(), task_id=42, estimated_cost_usd=0.05,
        _policy_override=AUTO_BELOW_BUDGET, _ceiling_override=0.10)

    assert (outcome.requires_approval, outcome.grant, outcome.reason, outcome.policy) == (
        False, None, board_approval.AUTO_APPROVED_REASON, AUTO_BELOW_BUDGET)
    assert seams.created == [] and seams.bells == [] and seams.audits == []


@pytest.mark.parametrize("policy", sorted(VALID_POLICIES))
def test_the_tickets_blocked_reason_still_reads_as_an_approval(policy):
    from core.services import ticket_reasons as tr

    blocked = APPROVAL_PREFIX.format(grant_id=GRANT.id) + board_approval.held_reason(policy)
    ticket = SimpleNamespace(status="blocked", source_type="user", review_mode="auto", result=None,
                             review_feedback=None, planning_data={}, runtime_ref={}, blocked_reason=blocked,
                             completed_at=None)

    assert tr.blocked_code(ticket) == tr.APPROVAL


@pytest.mark.parametrize("policy", sorted(VALID_POLICIES))
def test_the_approval_view_reads_the_policy_back_from_the_sentence(policy):
    from services.approval_views import _policy_of

    # A reason it matches never reads the workspace (db=None would raise).
    assert _policy_of(None, uuid.uuid4(), board_approval.held_reason(policy)) == policy


def test_the_gate_keeps_its_signature_its_outcome_and_the_names_tests_patch():
    params = inspect.signature(board_approval.evaluate_board_task_approval).parameters

    assert list(params) == ["db", "workspace_id", "task_id", "estimated_cost_usd", "agent_id", "risk_tier",
                            "ttl_seconds", "_policy_override", "_ceiling_override"]
    assert [f.name for f in fields(board_approval.BoardApprovalOutcome)] == [
        "requires_approval", "reason", "grant", "policy", "estimated_cost_usd"]
    assert all(callable(getattr(board_approval, name)) for name in (
        "_audit_governance", "_dispatch_approval_pending", "_notify_approval_pending", "_decide_from_override"))
