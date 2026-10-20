"""PRD-263 US-019 — tool failures are judged on this week's production logs; a broken connection says "reconnect".

At 697ca7d6e HARNESS read the routing graph's agent-keyed ``fails_for_intent``
affinities with ``sample_count >= 5``. That count only grows (signal_recorder), so a tool
that failed once in a bad month "kept failing" for ever, and a broken connection was
proposed for removal like a bad tool. DIAGNOSE now counts each agent's production calls
of each app in the judged ISO week (services.harness_metrics) and flags a pair only with
n of at least HARNESS_MIN_SAMPLES and a Wilson lower bound of the failure rate of at
least HARNESS_TOOL_FAIL_WILSON_MIN (services.harness_diagnose.tool_failure_issues, FR-8).
Failures ``classify_tool_error`` reads as auth or connection give "reconnect <app>",
never a removal; tool_assignment_remove is a registry kind (US-049) that PRESCRIBE
detects only with its dependency gate (US-057).

Replaces test_harness_tool_failure_signal.py. Each case runs the real pipeline on the
transactional ``db_session`` with the audit report faked
(helpers_harness_diagnose.judge); the judged week is 2026-W42.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from services import harness_diagnose as hd
from services import harness_metrics as hm
from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

APP = "GMAIL"
REMOVAL = "tool_assignment_remove"
EXPIRED = "Error executing action: Connected account ca_8Hk2 has expired. Reconnect it to continue."
NOT_FOUND = "channel_not_found"


@pytest.fixture
def calls(monkeypatch, tmp_path):
    return hr.fake_platform(monkeypatch, tmp_path)


@pytest.fixture
def ws(db_session, seed_workspace, calls):
    """A workspace with its Auto."""
    a = seed_workspace(name="harness-tool-failures")
    hdx.auto(db_session, a)
    return a


def _issues(db_session, ws):
    """DIAGNOSE's tool issues for the judged week, straight from the read model."""
    metrics = hm.read_sync(db_session, ws, hdx.WEEK, ref_weeks=hd.reference_weeks())
    return hd.tool_failure_issues(metrics, hd.settings_from_config())


def _no_removal(run):
    assert REMOVAL not in run["report_md"]
    assert all(rx["change_type"] != REMOVAL for rx in run["prescriptions"] or [])


def test_the_thresholds_are_the_prd_defaults():
    settings = hd.settings_from_config()
    assert (settings.rates.min_samples, settings.tool_fail_wilson_min) == (20, 0.5)


def test_a_high_lifetime_failure_count_with_no_failure_this_week_is_not_flagged(db_session, ws):
    scout = hdx.agent(db_session, ws, "Scout")
    db_session.execute(text(
        "INSERT INTO tool_routing_affinities (action_name, affinity_type, workspace_id, agent_id, weight, "
        "confidence, sample_count, last_updated) VALUES ('GMAIL_SEND_EMAIL', 'fails_for_intent', "
        "CAST(:ws AS uuid), :agent, 1.0, 0.9, 500, now())"), {"ws": str(ws), "agent": scout})
    hdx.tool_runs(db_session, ws, scout, -3, 20, app=APP, status="error", error_message=NOT_FOUND)  # a bad week, long gone
    hdx.tool_runs(db_session, ws, scout, 0, 20, app=APP)                                             # this week: all fine

    assert _issues(db_session, ws) == []
    run = hdx.judge(db_session, ws)
    assert "Issues Detected" not in run["report_md"] and run["prescriptions"] == []


def test_18_of_20_non_auth_errors_flag_as_a_tool_failure(db_session, ws):
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 2, app=APP)
    hdx.tool_runs(db_session, ws, scout, 0, 18, app=APP, status="error", error_message=NOT_FOUND)

    (issue,) = _issues(db_session, ws)
    assert (issue["root_cause"], issue["app_name"], issue["calls"], issue["failed_calls"]) == (
        hd.TOOL_FAILURE, APP, 20, 18)
    run = hdx.judge(db_session, ws)
    assert ("- **Scout**: GMAIL failed 18 of 20 calls this week (a failure rate of at least 70%) "
            "(root cause: tool_failure)") in run["report_md"]
    assert "reconnect" not in run["report_md"].lower()
    assert run["prescriptions"] == []  # Scout holds no GMAIL assignment: the removal gate (US-057) proposes none
    _no_removal(run)


def test_fewer_calls_than_the_minimum_are_not_judged(db_session, ws):
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 19, app=APP, status="error", error_message=NOT_FOUND)
    assert _issues(db_session, ws) == []


def test_20_of_20_errors_with_an_auth_code_give_a_reconnect_line_and_no_removal(db_session, ws):
    """The code is an HTTP 401: no writer of an auth code exists yet, so the patterns class it."""
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 20, app=APP, status="error", error_code="401")

    (issue,) = _issues(db_session, ws)
    assert (issue["root_cause"], issue["reconnect_calls"]) == (hd.RECONNECT_APP, 20)
    run = hdx.judge(db_session, ws)
    report = run["report_md"]
    assert "- **Scout**: reconnect GMAIL: 20 of its 20 calls this week failed on GMAIL's sign-in or connection" in report
    (rx,) = run["prescriptions"]
    assert (rx["change_type"], rx["target_id"], rx["proposed_value"]) == (hd.RECONNECT_APP, scout, {"app_name": APP})
    assert "GMAIL failed" not in report
    _no_removal(run)


def test_20_of_20_errors_with_no_code_and_an_expired_auth_message_give_a_reconnect_line(db_session, ws):
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 20, app=APP, status="error", error_message=EXPIRED)

    run = hdx.judge(db_session, ws)

    assert "reconnect GMAIL: 20 of its 20 calls this week" in run["report_md"]
    assert [rx["change_type"] for rx in run["prescriptions"]] == [hd.RECONNECT_APP]
    _no_removal(run)


def test_a_flag_the_other_failures_carry_alone_stays_a_tool_failure(db_session, ws):
    """5 expired sign-ins and 15 other failures of 20: the 15 alone flag (Wilson 0.53)."""
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 5, app=APP, status="error", error_message=EXPIRED)
    hdx.tool_runs(db_session, ws, scout, 0, 15, app=APP, status="error", error_message=NOT_FOUND)

    (issue,) = _issues(db_session, ws)
    assert (issue["root_cause"], issue["failed_calls"], issue["reconnect_calls"]) == (hd.TOOL_FAILURE, 20, 5)


def test_a_platform_tools_refusals_are_never_a_reconnect(db_session, ws):
    """PLATFORM and WORKSPACE are the platform's own tools: there is no connection to renew."""
    scout = hdx.agent(db_session, ws, "Scout")
    hdx.tool_runs(db_session, ws, scout, 0, 20, app="PLATFORM", status="error", error_message="403 Forbidden")

    (issue,) = _issues(db_session, ws)
    assert (issue["root_cause"], issue["app_name"]) == (hd.TOOL_FAILURE, "PLATFORM")


def test_the_lifetime_read_and_the_removal_prescriber_are_gone():
    """US-019 deletes the agent-keyed fails_for_intent read and what served it (grep returns 0);
    the recorder that writes fails_for_intent for the GraphRouter is unchanged."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "services" / "harness_service.py").read_text(encoding="utf-8")
    for name in ("fails_for_intent", "ToolRoutingAffinity", "_TOOL_FAILURE_MIN_SAMPLES", "_derive_app_name",
                 "_prescribe_tool_removals", "_diagnose_tool_failures"):
        assert name not in source, name
