"""PRD-263 US-056 — an approved change is checked against the weeks before it, and a rollback comes as its own card.

US-014b deleted the baseline-file rollback check (``_detect_auto_applied_regressions``),
which compared an applied change with last week's baseline; until now nothing judged an
applied change at all. services.harness_post_change judges each ``applied`` change in
harness_task_ledger against its frozen reference, the HARNESS_REF_DAYS before
``applied_at`` that the apply action stores with it, week by week over the ISO weeks after
the week it applied in: its predicted metric, then task success and tool failures. Each
waits for a week of HARNESS_MIN_SAMPLES for up to HARNESS_POST_CHANGE_MAX_WEEKS weeks. Two
consecutive flags, or one below HARNESS_STRONG_P, trigger it; the agent's drop must then
beat the median drop of the unchanged agents judged over the same weeks, or there must be
none ("no control"). The rollback is one more PRESCRIBE suggestion, so in a propose
workspace it is its own Approvals card; once it applies, the change it undoes moves
``applied → rolled_back``. A rollback is never rolled back.

On the real database (``db_session``, rolled back), no model call. Scout's heartbeat went
from every 60 to every 120 minutes on Wednesday 14 Oct 2026 (W42), so its frozen reference
is 16 Sep to 14 Oct and the run judging W44 sees W43 and W44 after it. Analyst is an agent
HARNESS never changed. The audit report is faked and the rate limiter stubbed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy import text

from services import harness_changes as hc
from services import harness_metrics as hm
from services import harness_post_change as hpc
from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_grants as hgx
from tests import helpers_harness_runs as hr

APPLIED_AT = hdx.at(0)            # Wednesday 14 Oct 2026, 10:00 UTC (W42)
ONE_WEEK_ON, TWO_WEEKS_ON = "2026-W43", "2026-W44"
ORIGINAL = "rx-1-001"
HEARTBEAT, REMOVAL = "heartbeat_tune", "tool_assignment_remove"
GMAIL_BEFORE, GMAIL_AFTER = {"app_name": "GMAIL", "assigned": True}, {"app_name": "GMAIL", "assigned": False}
UNDO = {"kind": HEARTBEAT, "value": {"interval_minutes": hgx.PRIOR}, "rx_id": ORIGINAL}
OWNER = NS(id=7, clerk_user_id=None, email="owner@harness-post-change.test")
FELL = ("Task success fell from 94% (30 of 32) in the 28 days before the change to 31% (10 of 32) "
        "in week 1 after it")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    """No rate limiter, a faked audit report, the judged week pinned, the approval routes'
    audit and chat notes off, and no approval_pending notification sent."""
    import api.approval_grants as ag
    from services.report_service import ReportService

    async def _report(self, **_fields):
        return {"success": True, "report_id": "report-1"}

    hr.pin_run(monkeypatch, tmp_path)
    monkeypatch.setattr(ReportService, "create_report", _report)
    monkeypatch.setattr("core.security.rate_limiter.check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(ag, "said_in_the_chat", lambda *a, **k: None)
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending", lambda grant, ws: None)


@pytest.fixture
def crew(db_session, seed_workspace, monkeypatch):
    """A propose workspace, proposals on, with its Auto; Scout, whose heartbeat HARNESS set
    to every 120 minutes; and Analyst."""
    from config import config

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    ws = seed_workspace(name="harness-post-change")
    db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"),
                       {"s": '{"orchestrator": {"harness": {"mode": "propose"}}}', "id": ws})
    auto = hdx.auto(db_session, ws)
    scout = hdx.agent(db_session, ws, "Scout",
                      configuration={"heartbeat": {"enabled": True, "interval_minutes": hgx.PROPOSED}})
    return NS(ws=ws, auto=auto, scout=scout, analyst=hdx.agent(db_session, ws, "Analyst"))


def _applied(db, crew, *, kind=HEARTBEAT, prior=None, proposed=None) -> int:
    """A change HARNESS applied to Scout at APPLIED_AT, as the apply action records it."""
    from core.models.harness import HarnessTaskLedger

    prior = prior or {"interval_minutes": hgx.PRIOR}
    row = HarnessTaskLedger(
        workspace_id=UUID(crew.ws), state="applied", rx_id=ORIGINAL, run_id=1, kind=kind,
        target_agent_id=crew.scout, prior_value=prior, proposed_value=proposed or {"interval_minutes": hgx.PROPOSED},
        applied_at=APPLIED_AT, value_at_apply=prior, frozen_ref_start=APPLIED_AT - timedelta(days=28),
        frozen_ref_end=APPLIED_AT,
    )
    db.add(row)
    db.flush()
    return row.id


def _tasks(db, crew, agent_id, *, after):
    """30 of 32 Tasks done in W41, inside the frozen reference; then ``after`` (done, failed)
    in each of W43 and W44."""
    hdx.tickets(db, crew.ws, agent_id, -1, 30, 2)
    for week in (1, 2):
        hdx.tickets(db, crew.ws, agent_id, week, *after)


COLLAPSED, STEADY = (10, 22), (30, 2)


def _check(db, crew, week=TWO_WEEKS_ON) -> hpc.PostChange:
    return hpc.check_sync(db, crew.ws, week, hpc.settings_from_config())


def _rows(db, crew, sql):
    return [dict(r) for r in db.execute(text(sql), {"ws": crew.ws}).mappings()]


# ── a regression against the frozen reference ────────────────────────────────


def test_a_frozen_reference_regression_raises_one_rollback_grant(db_session, crew):
    _applied(db_session, crew)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    _tasks(db_session, crew, crew.analyst, after=STEADY)

    run = hdx.judge(db_session, crew.ws, TWO_WEEKS_ON)

    (rx,) = run["prescriptions"]
    assert (rx["change_type"], rx["apply_kind"], rx["target_id"]) == ("rollback", True, crew.scout)
    assert (rx["current_value"], rx["proposed_value"]) == ({"interval_minutes": hgx.PROPOSED}, UNDO)
    assert (rx["evidence"]["metric"], rx["evidence"]["control"]) == ("task_success", "EXCEEDS_CONTROL")
    (grant,) = _rows(db_session, crew, "SELECT * FROM approval_grants WHERE workspace_id = CAST(:ws AS uuid)")
    pending = _rows(db_session, crew, "SELECT * FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                                      "AND state = 'pending'")
    assert [(row["kind"], row["grant_id"], row["proposed_value"]) for row in pending] == [
        ("rollback", grant["id"], UNDO)]
    assert grant["details"]["params"] == {"rx_id": rx["prescription_id"]}
    assert FELL in grant["question_md"] and "the 1 unchanged agent(s) judged" in grant["question_md"]
    assert "## Changes HARNESS made" in run["report_md"] and FELL in run["report_md"]


def test_the_change_moves_to_rolled_back_once_its_rollback_applies(db_session, crew):
    import api.approval_grants as ag

    original = _applied(db_session, crew)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    hdx.judge(db_session, crew.ws, TWO_WEEKS_ON)
    (grant,) = _rows(db_session, crew, "SELECT id FROM approval_grants WHERE workspace_id = CAST(:ws AS uuid)")

    asyncio.run(ag.grant_approval(grant["id"], ctx=NS(workspace_id=UUID(crew.ws), user=OWNER), db=db_session))

    assert hgx.interval(db_session, crew.scout) == hgx.PRIOR          # back to every 60 minutes
    assert hgx.row(db_session, original)["state"] == "rolled_back"
    (undo,) = _rows(db_session, crew, "SELECT * FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                                      "AND kind = 'rollback'")
    assert undo["state"] == "applied" and undo["frozen_ref_end"] == undo["applied_at"]


def test_a_tool_removal_whose_app_falls_to_n_0_is_judged_on_task_success(db_session, crew):
    _applied(db_session, crew, kind=REMOVAL, prior=GMAIL_BEFORE, proposed=GMAIL_AFTER)
    hdx.tool_runs(db_session, crew.ws, crew.scout, -1, 30, app="GMAIL", status="error", error_message="Bad reply")
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    _tasks(db_session, crew, crew.analyst, after=STEADY)

    result = _check(db_session, crew)

    (rollback,) = result.rollbacks
    assert rollback.evidence["metric"] == "task_success"
    assert rollback.proposed_value == {"kind": REMOVAL, "value": GMAIL_BEFORE, "rx_id": ORIGINAL}
    assert rollback.current_value == GMAIL_AFTER
    (line,) = result.lines
    assert "GMAIL failures: not judged yet (0 GMAIL calls since the change, 30 in the 28 days before)" in line
    assert "Task success: worse than before the change" in line


def _heartbeats(db, crew, week, *, met, missed):
    beat = text("INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
                "objective_met, created_at) VALUES ('agent', :sid, CAST(:ws AS uuid), 'success', 10, :met, :at)")
    for outcome in [True] * met + [False] * missed:
        db.execute(beat, {"sid": str(crew.scout), "ws": crew.ws, "met": outcome, "at": hdx.at(week)})


def test_a_heartbeat_tune_that_halves_n_waits(db_session, crew):
    """Every 120 minutes Scout runs half as many heartbeats: 12 judged a week is under
    HARNESS_MIN_SAMPLES, so a week where none met its objective is not a flag yet."""
    _applied(db_session, crew)
    _heartbeats(db_session, crew, -1, met=20, missed=4)
    _heartbeats(db_session, crew, 1, met=0, missed=12)

    result = _check(db_session, crew, ONE_WEEK_ON)

    assert result.rollbacks == ()
    (line,) = result.lines
    assert line.startswith("- Scout's heartbeat, every 60 minutes → every 120 minutes, applied 2026-10-14: ")
    assert ("Heartbeat objectives met: not judged yet (12 judged heartbeats since the change, 24 in the 28 days "
            "before)") in line


def test_no_judged_control_raises_the_rollback_labelled_no_control(db_session, crew):
    _applied(db_session, crew)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    hdx.tickets(db_session, crew.ws, crew.analyst, 1, 3, 0)     # too few Tasks to judge

    (rollback,) = _check(db_session, crew).rollbacks

    assert (rollback.evidence["control"], rollback.evidence["control_agents"]) == ("no control", 0)
    assert rollback.rationale.startswith(FELL) and rollback.rationale.endswith("(no control).")


def test_a_drop_no_worse_than_the_unchanged_agents_raises_no_rollback(db_session, crew):
    _applied(db_session, crew)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    _tasks(db_session, crew, crew.analyst, after=COLLAPSED)

    result = _check(db_session, crew)

    assert result.rollbacks == ()
    assert ("Task success: worsened by 62 points, no more than the 1 unchanged agent(s) judged (median 62), so "
            "nothing to undo") in result.lines[0]


# ── a rollback is never rolled back, nor raised twice ───────────────────────


def test_a_rollback_is_never_rolled_back(db_session, crew):
    _applied(db_session, crew, kind="rollback", prior={"interval_minutes": hgx.PROPOSED}, proposed=UNDO)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    _tasks(db_session, crew, crew.analyst, after=STEADY)

    assert _check(db_session, crew) == hpc.PostChange()
    assert "rollback" not in hpc.JUDGED_KINDS
    with pytest.raises(ValueError):
        hc.KINDS["rollback"].rollback_builder(crew.scout, UNDO)


def test_a_declined_rollback_is_not_raised_again(db_session, crew):
    _applied(db_session, crew)
    hgx.ledger(db_session, crew.ws, "rx-2-001", crew.scout, None, state="denied", kind="rollback",
               prior={"interval_minutes": hgx.PROPOSED}, proposed=UNDO)
    _tasks(db_session, crew, crew.scout, after=COLLAPSED)
    _tasks(db_session, crew, crew.analyst, after=STEADY)

    result = _check(db_session, crew)

    assert result.rollbacks == ()
    assert result.lines[0].endswith("An earlier card to undo it is denied.")


# ── the frozen reference and the weeks judged ────────────────────────────────


def test_applying_a_change_stores_its_frozen_reference(db_session, crew):
    from modules.tools.discovery.handlers_harness import harness_apply

    hgx.set_interval(db_session, crew.scout, hgx.PRIOR)
    ledger_id, _ = hgx.proposal(db_session, crew)

    reply = asyncio.run(harness_apply(db_session, UUID(crew.ws), {"rx_id": "rx-1-001"}))

    assert reply["success"] is True, reply
    row = hgx.row(db_session, ledger_id)
    assert row["frozen_ref_end"] == row["applied_at"]
    assert row["frozen_ref_start"] == row["applied_at"] - timedelta(days=28)


def test_a_change_is_judged_on_the_weeks_after_the_week_it_applied_in_for_up_to_four():
    assert hpc.post_weeks(APPLIED_AT, "2026-W42", 4) == ()
    assert hpc.post_weeks(APPLIED_AT, TWO_WEEKS_ON, 4) == (ONE_WEEK_ON, TWO_WEEKS_ON)
    assert hpc.post_weeks(APPLIED_AT, "2026-W46", 4) == (ONE_WEEK_ON, TWO_WEEKS_ON, "2026-W45", "2026-W46")
    assert hpc.post_weeks(APPLIED_AT, "2026-W47", 4) is None        # the check is over


def test_the_frozen_reference_is_read_between_its_exact_moments():
    window = hm.span_window(APPLIED_AT - timedelta(days=28), APPLIED_AT)

    assert window.weeks == ("2026-W38", "2026-W39", "2026-W40", "2026-W41", "2026-W42")
    assert (window.start, window.end) == (datetime(2026, 9, 16, 10), datetime(2026, 10, 14, 10))
    with pytest.raises(ValueError):
        hm.span_window(APPLIED_AT, APPLIED_AT)
