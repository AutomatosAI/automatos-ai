"""PRD-263 US-057 — a tool is proposed for removal only when it truly fails and nothing depends on it.

At 697ca7d6e HARNESS proposed removing a tool from the routing graph's lifetime
``fails_for_intent`` count (``_prescribe_tool_removals``, deleted by US-019), whatever the
failures were and whatever still used the tool. Now ``tool_assignment_remove``
(services.harness_changes) is suggested only for an app assigned to the agent whose judged
week's production calls fail with a Wilson lower bound of at least
HARNESS_TOOL_FAIL_WILSON_MIN, not counting sign-in and connection failures (those stay
"reconnect <app>" lines) and never reading asked or refused rows. It is an apply kind, an
Approvals card in a propose workspace, only while no Playbook or active Mission that routes
to the agent needs the app; otherwise it stays a line of the report (section 11-Q17
default: the check is built from existing tables, harness_metrics' app links).

Each case runs the real pipeline (``helpers_harness_diagnose.judge``) for W42 on the
transactional ``db_session``, in a propose workspace with proposals switched on, the audit
report faked.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

REMOVAL = "tool_assignment_remove"
# A failure the server does not put down to the app's sign-in or connection.
BAD_REPLY = "Unexpected response shape from the provider"
GRANTS = text("SELECT COUNT(*) FROM approval_grants WHERE workspace_id = CAST(:ws AS uuid)")
LEDGER = text("SELECT kind FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid)")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    hr.fake_platform(monkeypatch, tmp_path)
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending", lambda grant, ws: None)


@pytest.fixture
def crew(db_session, seed_workspace, monkeypatch):
    """A propose workspace, proposals on, with its Auto and Scout, who holds GMAIL."""
    from config import config
    from core.models.composio_cache import AgentAppAssignment

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    ws = seed_workspace(name="harness-tool-removal")
    db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:id AS uuid)"),
                       {"s": '{"orchestrator": {"harness": {"mode": "propose"}}}', "id": ws})
    hdx.auto(db_session, ws)
    scout = hdx.agent(db_session, ws, "Scout")
    db_session.add(AgentAppAssignment(agent_id=scout, app_name="GMAIL", is_active=True))
    db_session.flush()
    return NS(ws=ws, scout=scout)


def _failing(db, crew, count=20):
    """``count`` GMAIL calls by Scout this week, every one failing on a bad reply."""
    hdx.tool_runs(db, crew.ws, crew.scout, 0, count, app="GMAIL", status="error", error_message=BAD_REPLY)


def _removals(run):
    return [rx for rx in run["prescriptions"] if rx["change_type"] == REMOVAL]


def _grants(db, crew):
    return db.execute(GRANTS, {"ws": crew.ws}).scalar()


def _ledger(db, crew):
    return list(db.execute(LEDGER, {"ws": crew.ws}).scalars())


def _playbook(db, crew, *, steps, required_tools):
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"p263-{uuid.uuid4().hex[:8]}", name="Weekly digest",
                                description="US-057", workspace_id=UUID(crew.ws), template_definition={"steps": []},
                                steps=steps, required_tools=required_tools, created_by="us057")
    db.add(playbook)
    db.flush()
    return playbook


def _step_routed_to_scout(crew):
    return {"step_id": "s1", "order": 1, "agent_id": crew.scout, "prompt_template": "Send the weekly digest"}


def _listed_in_required_tools(db, crew):
    _playbook(db, crew, steps=[_step_routed_to_scout(crew)], required_tools=["GMAIL"])


def _called_in_a_recorded_run(db, crew):
    """A Playbook that lists no tools, whose last run's step by Scout sent an email."""
    from core.models.core import RecipeExecution

    playbook = _playbook(db, crew, steps=[_step_routed_to_scout(crew)], required_tools=[])
    db.add(RecipeExecution(execution_id=f"exec-{uuid.uuid4().hex[:8]}", recipe_id=playbook.id,
                           workspace_id=UUID(crew.ws), status="completed", triggered_by="us057",
                           step_results=[{"step_id": "s1", "agent_id": crew.scout,
                                          "tool_calls_summary": ["GMAIL_SEND_EMAIL (success)"]}]))
    db.flush()


# ── a tool that truly fails, and nothing depends on it ───────────────────────


def test_a_failing_app_nothing_depends_on_becomes_one_removal_card(db_session, crew):
    _failing(db_session, crew)

    run = hdx.judge(db_session, crew.ws)

    (rx,) = _removals(run)
    assert (rx["apply_kind"], rx["target_id"]) == (True, crew.scout) and "report_only" not in rx
    assert (rx["current_value"], rx["proposed_value"]) == ({"app_name": "GMAIL", "assigned": True},
                                                           {"app_name": "GMAIL", "assigned": False})
    assert rx["rationale"].startswith("20 of Scout's 20 GMAIL calls this week failed")
    assert _grants(db_session, crew) == 1 and _ledger(db_session, crew) == [REMOVAL]


# ── something depends on it: a report line, not a grant ──────────────────────


@pytest.mark.parametrize("needs", [_listed_in_required_tools, _called_in_a_recorded_run],
                         ids=["required-tools", "recorded-run"])
def test_an_app_used_by_an_active_playbook_step_routed_to_the_agent_yields_a_report_line_not_a_grant(
        db_session, crew, needs):
    _failing(db_session, crew)
    needs(db_session, crew)

    run = hdx.judge(db_session, crew.ws)

    (rx,) = _removals(run)
    kept = 'GMAIL is needed by the Playbook "Weekly digest", which routes to Scout'
    assert (rx["apply_kind"], rx["report_only"]) == (False, kept)
    assert f"Reported only: {kept}." in run["report_md"]
    assert _grants(db_session, crew) == 0 and _ledger(db_session, crew) == []


@pytest.mark.parametrize("state, blocks", [("running", True), ("completed", False)])
def test_only_a_mission_still_running_that_routes_to_the_agent_keeps_the_app(db_session, crew, state, blocks):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    _failing(db_session, crew)
    mission = OrchestrationRun(workspace_id=UUID(crew.ws), goal="Answer the October enquiries", state=state,
                               created_by="user_us057", config={})
    db_session.add(mission)
    db_session.flush()
    db_session.add_all([
        OrchestrationTask(run_id=mission.id, title="Draft the replies", sequence_number=1,
                          assigned_agent_id=crew.scout, input_context={}),
        OrchestrationTask(run_id=mission.id, title="Send the replies", sequence_number=2,
                          input_context={"required_tools": ["gmail"]}),
    ])
    db_session.flush()

    (rx,) = _removals(hdx.judge(db_session, crew.ws))

    assert rx["apply_kind"] is not blocks
    assert rx.get("report_only") == (
        'GMAIL is needed by the Mission "Answer the October enquiries", which routes to Scout' if blocks else None)
    assert _grants(db_session, crew) == (0 if blocks else 1)


# ── what does not count as a failure ─────────────────────────────────────────


def test_refused_and_asked_rows_do_not_count_toward_the_failure_rate(db_session, crew):
    """Counted as failures, 60 refused or asked calls beside 20 that worked would flag
    (a Wilson lower bound of 0.64 on 60 of 80); they are not tool runs."""
    hdx.tool_runs(db_session, crew.ws, crew.scout, 0, 20, app="GMAIL")
    hdx.tool_runs(db_session, crew.ws, crew.scout, 0, 30, app="GMAIL", status="refused", error_message=BAD_REPLY)
    hdx.tool_runs(db_session, crew.ws, crew.scout, 0, 30, app="GMAIL", status="asked")

    run = hdx.judge(db_session, crew.ws)

    assert _removals(run) == [] and _grants(db_session, crew) == 0
    assert "GMAIL failed" not in run["report_md"]


def test_sign_in_failures_stay_a_reconnect_line(db_session, crew):
    hdx.tool_runs(db_session, crew.ws, crew.scout, 0, 20, app="GMAIL", status="error",
                  error_message="401 Unauthorized: the access token has expired")

    run = hdx.judge(db_session, crew.ws)

    assert _removals(run) == []
    assert [rx["change_type"] for rx in run["prescriptions"]] == ["reconnect_app"]


def test_an_app_the_agent_does_not_hold_is_never_proposed_for_removal(db_session, crew):
    hdx.tool_runs(db_session, crew.ws, crew.scout, 0, 20, app="NOTION", status="error", error_message=BAD_REPLY)

    run = hdx.judge(db_session, crew.ws)

    assert _removals(run) == []
    assert "NOTION failed 20 of 20 calls this week" in run["report_md"]     # still reported as an issue
