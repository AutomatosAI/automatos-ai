"""PRD-263 US-050 — HARNESS proposes a longer interval only for a heartbeat that rarely meets its objective.

At 697ca7d6e the heartbeat rule read list_agents fields that never exist
(``heartbeat_cost_7d`` and a week-over-week task volume delta), so it never fired;
PR-1D deleted it with those inputs. ``heartbeat_tune`` (services.harness_changes) reads
the judged week's heartbeat outcomes from COLLECT instead: only an enabled heartbeat of
a non-Auto agent, with runs whose ``objective_met`` is None left out of the count. It
proposes doubling the interval (clamped to 5-1440, as configure_agent_heartbeat clamps
it) when at least HARNESS_HEARTBEAT_MIN_RUNS (20) runs have an outcome and the Wilson
upper bound of the share that met their objective is below HARNESS_HEARTBEAT_MAX_MET
(0.2), section 11-Q10's defaults. The prior value is
``configuration.heartbeat.interval_minutes``.

The detector cases are pure. The reader and the full run use the real database
(``db_session``; ``helpers_harness_diagnose.judge`` with the audit report faked).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from services import harness_changes as hc
from tests import helpers_harness_changes as hcx
from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

SCOUT = 7


def _proposals(*rows):
    return hc.KINDS[hc.HEARTBEAT_TUNE].detector(hcx.evidence(*rows))


def test_the_thresholds_are_the_prd_defaults():
    assert hc.settings_from_config() == hc.ChangeSettings(heartbeat_min_runs=20, heartbeat_max_met=0.2)


def test_25_runs_with_an_outcome_and_1_met_propose_twice_the_interval():
    (proposal,) = _proposals(hcx.agent_row(SCOUT, "Scout", interval=60, met=1, missed=24))

    assert (proposal.target_id, proposal.current_value, proposal.proposed_value) == (
        SCOUT, {"interval_minutes": 60}, {"interval_minutes": 120})
    evidence = proposal.evidence
    assert (evidence["met"], evidence["judged"], evidence["unknown"]) == (1, 25, 0)
    assert evidence["met_upper_bound"] == pytest.approx(0.1955, abs=1e-4)   # below 0.2
    assert "Heartbeat every 60 minutes → every 120 minutes: 1 of 25 heartbeats" in proposal.rationale


def test_25_runs_of_which_15_have_no_outcome_propose_nothing():
    """10 runs with an outcome is INSUFFICIENT_DATA: the 15 silent ones are not counted."""
    assert _proposals(hcx.agent_row(SCOUT, "Scout", met=1, missed=9, unknown=15)) == ()


def test_a_disabled_heartbeat_proposes_nothing():
    assert _proposals(hcx.agent_row(SCOUT, "Scout", enabled=False, met=1, missed=24)) == ()


def test_the_upper_bound_decides_not_the_raw_share():
    """2 of 25 is 8%, but chance alone allows up to 25%; 0 of 20 allows at most 16%."""
    assert _proposals(hcx.agent_row(SCOUT, "Scout", met=2, missed=23)) == ()
    (proposal,) = _proposals(hcx.agent_row(SCOUT, "Scout", met=0, missed=20))
    assert proposal.evidence["met_upper_bound"] == pytest.approx(0.1611, abs=1e-4)


@pytest.mark.parametrize("interval, proposed", [(2, 5), (60, 120), (1000, 1440), (1440, None), (10080, None)])
def test_the_interval_doubles_inside_the_handlers_clamp(interval, proposed):
    """A doubling the 5-1440 clamp would not lengthen (already at 1440, or a weekly 10080
    the clamp would shorten) proposes nothing."""
    found = _proposals(hcx.agent_row(SCOUT, "Scout", interval=interval, met=1, missed=24))
    assert [p.proposed_value["interval_minutes"] for p in found] == ([proposed] if proposed else [])


def test_only_the_judged_week_counts():
    rows = (hcx.agent_row(SCOUT, "Scout", week=hcx.REF_WEEK, met=1, missed=24), hcx.agent_row(SCOUT, "Scout"))
    assert _proposals(*rows) == ()


def test_prescribe_raises_it_as_an_apply_kind_with_its_metrics():
    prescribed = hc.prescribe(hcx.evidence(hcx.agent_row(SCOUT, "Scout", met=1, missed=24)), 3, auto_agent_id=1)

    (rx,) = prescribed.valid
    assert (rx["prescription_id"], rx["change_type"], rx["apply_kind"], rx["risk_score"]) == (
        "rx-3-001", "heartbeat_tune", True, 2)
    assert (rx["predicted_metric"], rx["guard_metrics"]) == ("heartbeat_objective", ["task_success", "tool_failure"])
    assert prescribed.dropped == ()


# ── the real database ────────────────────────────────────────────────────────

@pytest.fixture
def calls(monkeypatch, tmp_path):
    return hr.fake_platform(monkeypatch, tmp_path)


@pytest.fixture
def ws(db_session, seed_workspace, calls):
    """A workspace with its Auto."""
    a = seed_workspace(name="harness-heartbeat-tune")
    hdx.auto(db_session, a)
    return a


def test_the_prior_value_is_read_from_the_heartbeat_configuration(db_session, ws, seed_workspace):
    read = hc.KINDS[hc.HEARTBEAT_TUNE].prior_value_reader
    tuned = hdx.agent(db_session, ws, "Tuned", configuration={"heartbeat": {"enabled": True, "interval_minutes": 90}})
    unset = hdx.agent(db_session, ws, "Unset")
    other_ws = seed_workspace(name="harness-heartbeat-other")
    elsewhere = hdx.agent(db_session, other_ws, "Elsewhere",
                          configuration={"heartbeat": {"enabled": True, "interval_minutes": 30}})

    assert read(db_session, ws, tuned, {}) == {"interval_minutes": 90}
    assert read(db_session, ws, unset, {}) == {"interval_minutes": 60}   # the heartbeat service's default
    assert read(db_session, ws, elsewhere, {}) is None                   # another workspace's agent


def test_a_run_reports_the_suggestion_and_changes_nothing(db_session, ws):
    from services.harness_service import SUGGESTIONS_HEADING

    configuration = {"heartbeat": {"enabled": True, "interval_minutes": 60}}
    scout = hdx.agent(db_session, ws, "Scout", configuration=configuration)
    beat = text("INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
                "objective_met, created_at) VALUES ('agent', :sid, CAST(:ws AS uuid), 'success', 10, :met, :at)")
    for met in [True] + [False] * 24 + [None] * 5:
        db_session.execute(beat, {"sid": str(scout), "ws": ws, "met": met, "at": hdx.at(0)})

    run = hdx.judge(db_session, ws)

    (rx,) = run["prescriptions"]
    assert (rx["change_type"], rx["target_id"], rx["current_value"], rx["proposed_value"]) == (
        "heartbeat_tune", scout, {"interval_minutes": 60}, {"interval_minutes": 120})
    assert (rx["evidence"]["judged"], rx["evidence"]["unknown"]) == (25, 5)
    assert run["counts"]["prescriptions"] == 1
    assert SUGGESTIONS_HEADING in run["report_md"]
    assert f"Scout: heartbeat_tune ({rx['prescription_id']}, risk 2/5)" in run["report_md"]
    stored = db_session.execute(text("SELECT configuration FROM agents WHERE id = :id"), {"id": scout}).scalar()
    assert stored == configuration                                       # report only: nothing applied
