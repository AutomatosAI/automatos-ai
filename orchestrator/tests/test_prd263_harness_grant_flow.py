"""PRD-263 US-054 — in propose workspaces, each proposed change becomes an Approvals card with its evidence.

Before PRD-263 a HARNESS change applied with no click (auto-apply), or through the
always-approving ``/approve`` route; US-016 made every run report only, and US-047
deleted ``/approve`` (its cases, ``test_harness_commands.py``, the approve and reject
cases of ``test_harness_api.py`` and ``test_prd179_harness_actuation.py``, map here). Now,
in a workspace whose mode is ``propose`` and while the installation's
HARNESS_SELF_MANAGEMENT_ENABLED is on, PRESCRIBE raises one tool_call grant for
``platform_harness_apply`` on ``{"rx_id": ...}`` per valid apply-kind prescription
(services.harness_grants.raise_cards), with the section 8 text and a ``pending`` ledger
row, all linked to one evidence ticket filed in Review for the owner's decision. The
owner's Approve on the approval API resumes into ``platform_harness_apply``, which makes
the change as Auto (US-052). With the switch off, nothing is raised.

Scripted on the real database (``db_session``, rolled back), no model call: Scout's
enabled heartbeat met its objective in 1 of 25 judged runs of the week, so PRESCRIBE
proposes doubling its interval (US-050). The audit report is faked, the dormancy check
reads "ready", the rate limiter is stubbed and each approval_pending notification is
recorded rather than sent.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_grants as hgx
from tests import helpers_harness_runs as hr

ORCH = Path(__file__).resolve().parents[1]
OWNER = NS(id=7, clerk_user_id=None, email="owner@harness-grants.test")
CARD_PARTS = ("Change: ", "Why: ", "If you approve: ", "If it goes wrong: ", "Expires: ")
BLOCKED_REASON = "evidence card was blocked; nothing changed"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    """No rate limiter, a faked audit report, a 'ready' dormancy check, and the approval
    routes' audit and chat notes off."""
    import api.approval_grants as ag
    from services.report_service import ReportService

    async def _report(self, **_fields):
        return {"success": True, "report_id": "report-1"}

    hr.pin_run(monkeypatch, tmp_path)
    monkeypatch.setattr(ReportService, "create_report", _report)
    monkeypatch.setattr("core.security.rate_limiter.check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(ag, "said_in_the_chat", lambda *a, **k: None)


@pytest.fixture
def notices(monkeypatch):
    """The grant ids each approval_pending notification would announce."""
    seen = []
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending",
                        lambda grant, workspace_id: seen.append(grant.id))
    return seen


@pytest.fixture
def switch(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    return config


@pytest.fixture
def crew(db_session, seed_workspace, switch, notices):
    """A propose workspace with its Auto and Scout, whose heartbeat every 60 minutes rarely meets its objective."""
    ws = seed_workspace(name="harness-grant-flow")
    db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"),
                       {"s": '{"orchestrator": {"harness": {"mode": "propose"}}}', "id": ws})
    team = hgx.team(db_session, ws)
    beat = text("INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
                "objective_met, created_at) VALUES ('agent', :sid, CAST(:ws AS uuid), 'success', 10, :met, :at)")
    for met in [True] + [False] * 24 + [None] * 5:
        db_session.execute(beat, {"sid": str(team.scout), "ws": ws, "met": met, "at": hdx.at(0)})
    return team


GRANTS = text("SELECT * FROM approval_grants WHERE workspace_id = CAST(:ws AS uuid) ORDER BY id")
LEDGER = text("SELECT * FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) ORDER BY id")
EVIDENCE = text("SELECT * FROM board_tasks WHERE workspace_id = CAST(:ws AS uuid) AND source_type = 'harness' "
                "ORDER BY id")


def _rows(db, crew, query):
    return [dict(r) for r in db.execute(query, {"ws": crew.ws}).mappings()]


def _grants(db, crew):
    return _rows(db, crew, GRANTS)


def _ledger(db, crew):
    return _rows(db, crew, LEDGER)


def _evidence(db, crew):
    return _rows(db, crew, EVIDENCE)


def _ctx(crew):
    return NS(workspace_id=UUID(crew.ws), user=OWNER)


def _approve(db, crew, grant_id):
    import api.approval_grants as ag

    return asyncio.run(ag.grant_approval(grant_id, ctx=_ctx(crew), db=db))


def _deny(db, crew, grant_id):
    import api.approval_grants as ag

    return asyncio.run(ag.deny_approval(grant_id, ctx=_ctx(crew), db=db))


def _raised(db, crew):
    """One tick, then its one grant and ledger row."""
    run = hdx.judge(db, crew.ws)
    (grant,), (row,) = _grants(db, crew), _ledger(db, crew)
    return run, grant, row


# ── a tick raises one card per proposed change, with its evidence ────────────


def test_seeded_evidence_raises_exactly_one_grant_with_its_evidence_ticket(db_session, crew, notices):
    run, grant, row = _raised(db_session, crew)

    (rx,) = run["prescriptions"]
    assert (rx["change_type"], rx["proposed_value"]) == ("heartbeat_tune", {"interval_minutes": hgx.PROPOSED})
    assert (grant["status"], grant["tool_name"], grant["subject_type"]) == ("pending", hgx.APPLY, "tool_call")
    assert grant["details"]["params"] == {"rx_id": rx["prescription_id"]}
    (ticket,) = _evidence(db_session, crew)
    assert grant["details"]["board_task_id"] == ticket["id"]
    assert (row["state"], row["rx_id"], row["grant_id"], row["kind"], row["target_agent_id"]) == (
        "pending", rx["prescription_id"], grant["id"], "heartbeat_tune", crew.scout)
    assert (row["prior_value"], row["proposed_value"]) == ({"interval_minutes": hgx.PRIOR},
                                                           {"interval_minutes": hgx.PROPOSED})
    assert row["entry"] == {"evidence_task_id": ticket["id"]} and row["board_task_id"] is None
    assert notices == [grant["id"]]                     # one approval_pending, for the card
    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR     # nothing applied by the tick


def test_the_card_says_its_five_parts_in_order_in_plain_words(db_session, crew):
    _, grant, _ = _raised(db_session, crew)

    lines = grant["question_md"].splitlines()
    assert [line[:len(part)] for line, part in zip(lines, CARD_PARTS)] == list(CARD_PARTS) and len(lines) == 5
    assert lines[0] == "Change: Scout's heartbeat, every 60 minutes → every 120 minutes."
    assert lines[1].startswith("Why: 1 of 25 heartbeats this week met their objective. Even allowing for "
                               "chance, fewer than 20% would. 5 heartbeats reported nothing to judge")
    assert "HARNESS judged 0 of 1 agents this week." in lines[1]
    assert lines[2] == "If you approve: only this setting changes, now, and only if it still reads every 60 minutes."
    assert "the 4 weeks before the change" in lines[3]
    assert lines[4].startswith("Expires: 24 hours from now. After that, approving does nothing")
    assert not any(word in grant["question_md"] for word in ("platform_", "rx_id", "interval_minutes", "rx-"))


def test_the_evidence_ticket_is_the_owners_decision_with_no_agent(db_session, crew):
    _, grant, _ = _raised(db_session, crew)

    (ticket,) = _evidence(db_session, crew)
    assert (ticket["status"], ticket["review_mode"], ticket["assigned_agent_id"], ticket["sla_deadline"]) == (
        "review", "human", None, None)
    assert ticket["planning_data"] == {"owners_decision": True} and ticket["tags"] == ["harness"]
    assert ticket["title"] == "HARNESS proposes 1 change for 2026-W42"
    assert "Scout" in ticket["description"] and f"(Approvals card #{grant['id']})" in ticket["description"]
    assert "1 of 25 heartbeats" in ticket["description"]


# ── the owner's click ────────────────────────────────────────────────────────


def test_approving_through_the_approval_api_applies_with_the_prior_value(db_session, crew):
    _, grant, row = _raised(db_session, crew)

    _approve(db_session, crew, grant["id"])

    assert hgx.interval(db_session, crew.scout) == hgx.PROPOSED
    after = hgx.row(db_session, row["id"])
    assert after["state"] == "applied" and after["applied_at"] is not None
    assert after["value_at_apply"] == {"interval_minutes": hgx.PRIOR}
    (resumed,) = _grants(db_session, crew)
    assert resumed["status"] == "granted" and resumed["details"]["executed_result"]["success"] is True


def test_deny_ends_denied_with_no_change(db_session, crew):
    from services import harness_grants

    _, grant, row = _raised(db_session, crew)

    _deny(db_session, crew, grant["id"])
    harness_grants.reconcile(db_session, crew.ws)

    assert hgx.grant_status(db_session, grant["id"]) == "denied"
    assert hgx.row(db_session, row["id"])["state"] == "denied"
    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR


def test_a_double_click_applies_once(db_session, crew):
    from modules.tools.discovery.handlers_harness import harness_apply

    _, grant, row = _raised(db_session, crew)

    _approve(db_session, crew, grant["id"])
    with pytest.raises(HTTPException) as again:
        _approve(db_session, crew, grant["id"])
    replay = asyncio.run(harness_apply(db_session, UUID(crew.ws), {"rx_id": row["rx_id"]}))

    assert again.value.status_code == 422
    assert replay == {"success": False, "error": "This change is already applied; nothing changed"}
    assert hgx.interval(db_session, crew.scout) == hgx.PROPOSED   # 60 → 120 once, never 240
    assert hgx.row(db_session, row["id"])["state"] == "applied"


# ── no card without the switch, and no second card for the same change ───────


def test_switch_off_raises_no_grant(db_session, crew, switch, notices, monkeypatch):
    monkeypatch.setattr(switch, "HARNESS_SELF_MANAGEMENT_ENABLED", False)

    run = hdx.judge(db_session, crew.ws)

    assert [rx["change_type"] for rx in run["prescriptions"]] == ["heartbeat_tune"]   # still reported
    assert _grants(db_session, crew) == [] and _ledger(db_session, crew) == [] and _evidence(db_session, crew) == []
    assert notices == []


def test_no_second_grant_while_one_waits(db_session, crew):
    hdx.judge(db_session, crew.ws)
    hdx.judge(db_session, crew.ws)   # the evidence still holds, and the first card still waits

    assert len(_grants(db_session, crew)) == 1 and len(_ledger(db_session, crew)) == 1
    assert len(_evidence(db_session, crew)) == 1


def _lapse(db, crew, grant_id):
    db.execute(text("UPDATE approval_grants SET expires_at = now() - interval '1 hour' WHERE id = :id"),
               {"id": grant_id})


def _edit(db, crew, grant_id):
    hgx.set_interval(db, crew.scout, 90)


@pytest.mark.parametrize("end, how, proposed", [("expired", _lapse, 120), ("drifted", _edit, 180)])
def test_after_a_grant_ends_expired_or_drifted_the_next_tick_raises_one_new_grant(db_session, crew, end, how,
                                                                                   proposed):
    from services import harness_grants

    _, grant, row = _raised(db_session, crew)
    how(db_session, crew, grant["id"])
    harness_grants.reconcile(db_session, crew.ws)
    assert hgx.row(db_session, row["id"])["state"] == end

    hdx.judge(db_session, crew.ws)

    pending = [g for g in _grants(db_session, crew) if g["status"] == "pending"]
    fresh = [r for r in _ledger(db_session, crew) if r["state"] == "pending"]
    assert len(_grants(db_session, crew)) == 2 and len(pending) == 1 and len(fresh) == 1
    assert fresh[0]["rx_id"] != row["rx_id"] and fresh[0]["grant_id"] == pending[0]["id"]
    assert fresh[0]["proposed_value"] == {"interval_minutes": proposed}


# ── the evidence ticket on the board ─────────────────────────────────────────


def _claimed(db, crew):
    from services import board_dispatcher

    return [task.id for task in board_dispatcher.claim_tasks(db, worker_id="grant-flow", limit=50,
                                                            lease_seconds=60, workspace_id=crew.ws)]


def _block(db, ticket_id, *, by_the_owner):
    """The ticket moved to Blocked: by the owner's drag on the board (which records their
    stop, services.operator_stop), or parked by the machine (which does not)."""
    from core.models.core import BoardTask
    from services.operator_stop import apply_explicit_status

    task = db.get(BoardTask, ticket_id)
    if by_the_owner:
        apply_explicit_status(task, task.status, "blocked", None, by="operator")
    task.status = "blocked"
    db.flush()


def test_the_dispatcher_never_claims_the_evidence_ticket(db_session, crew):
    _, grant, _ = _raised(db_session, crew)
    (ticket,) = _evidence(db_session, crew)

    assert ticket["id"] not in _claimed(db_session, crew)
    _block(db_session, ticket["id"], by_the_owner=False)
    _approve(db_session, crew, grant["id"])           # the Approve re-queues the parked ticket
    (requeued,) = _evidence(db_session, crew)
    assert (requeued["status"], requeued["assigned_agent_id"]) == ("assigned", None)
    assert ticket["id"] not in _claimed(db_session, crew)


@pytest.mark.parametrize("by_the_owner", [True, False], ids=["owners-drag", "machine-park"])
def test_with_the_evidence_ticket_blocked_approve_changes_nothing_and_the_ledger_ends_failed(
        db_session, crew, by_the_owner):
    """The owner's drag records a stop, so the approval API runs the click's call, which
    refuses (handlers_harness); a machine park is re-queued instead of running the call,
    and reconciliation closes the row (US-053). Either way: failed, with the reason."""
    from services import harness_grants

    _, grant, row = _raised(db_session, crew)
    (ticket,) = _evidence(db_session, crew)
    _block(db_session, ticket["id"], by_the_owner=by_the_owner)

    _approve(db_session, crew, grant["id"])
    harness_grants.reconcile(db_session, crew.ws)

    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR
    after = hgx.row(db_session, row["id"])
    assert after["state"] == "failed" and after["entry"]["reason"] == BLOCKED_REASON


def test_cards_are_raised_with_issue_tool_grant_and_never_attach_an_ask_grant():
    """The harness modules raise a card with issue_tool_grant, never attach_ask_grant (the
    confirmation gate's ask enrichment): grep returns 0."""
    harness = [*sorted((ORCH / "services").glob("harness_*.py")),
               ORCH / "modules/tools/discovery/handlers_harness.py",
               ORCH / "modules/tools/discovery/actions_harness.py"]
    assert [path.name for path in harness if "attach_ask_grant" in path.read_text(encoding="utf-8")] == []
    assert "issue_tool_grant(" in (ORCH / "services/harness_grants.py").read_text(encoding="utf-8")
