"""PRD-263 US-049 — one registry defines every change kind.

At 697ca7d6e PRESCRIBE emitted four change kinds, two of them with a placeholder for a
value, and the apply helper knew ten, so what HARNESS proposed and what it could apply
never matched. services.harness_changes now holds every kind (FR-13): the apply kinds
(heartbeat_tune, tool_assignment_remove, rollback), each with its detector, inner
platform action, params builder, risk, predicted metric, guard metrics, prior-value
reader and rollback builder; and the report kinds, which become report lines only.
PRESCRIBE is a loop over it, and its validator drops an invalid suggestion and keeps
the reason on the run's row (harness_runs.prescriptions). No suggestion targets the
workspace's Auto.

Replaces the _auto_apply_prescription cases of test_harness_power_mode.py and
test_harness_routing_rule.py (deleted with US-048): routing and power mode are report
kinds, so nothing can apply them.

Most cases are pure, on in-memory evidence (helpers_harness_changes). The tool
assignment reader and the full run use the real database (``db_session``;
``helpers_harness_diagnose.judge`` with the audit report faked).
"""
from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import text

from services import harness_changes as hc
from tests import helpers_harness_changes as hcx
from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

APPLY = {"heartbeat_tune", "tool_assignment_remove", "rollback"}
REPORT = {"reconnect_app", "model", "description", "temperature", "tags", "routing", "power_mode"}
EIGHT_FIELDS = {"detector", "inner_actions", "params_builder", "risk", "predicted_metric", "guard_metrics",
                "prior_value_reader", "rollback_builder"}
AUTO, SCOUT, ANALYST = 1, 7, 8
# A heartbeat that met its objective once in 25 runs: heartbeat_tune suggests doubling it.
RARELY_MET = {"met": 1, "missed": 24}


def test_the_registry_holds_the_apply_and_the_report_kinds():
    assert (set(hc.APPLY_KINDS), set(hc.REPORT_KINDS)) == (APPLY, REPORT)
    assert set(hc.KINDS) == APPLY | REPORT


def test_every_apply_kind_has_all_eight_fields():
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    assert {f.name for f in dataclasses.fields(hc.ChangeKind)} == EIGHT_FIELDS | {"name"}
    for name in hc.APPLY_KINDS:
        kind = hc.KINDS[name]
        assert all(getattr(kind, field) not in (None, ()) for field in EIGHT_FIELDS), name
        assert all(callable(fn) for fn in (kind.detector, kind.params_builder, kind.prior_value_reader,
                                           kind.rollback_builder)), name
        assert 1 <= kind.risk <= 5 and kind.guard_metrics == ("task_success", "tool_failure"), name
        assert set(kind.inner_actions) <= set(PLATFORM_HANDLERS), name   # each is a real platform action
    assert hc.KINDS["heartbeat_tune"].inner_actions == ("platform_configure_agent_heartbeat",)
    assert hc.KINDS["tool_assignment_remove"].inner_actions == ("platform_unassign_tool_from_agent",)


def test_report_kinds_yield_report_lines_only():
    for name in hc.REPORT_KINDS:
        kind = hc.KINDS[name]
        assert not kind.applies and kind.params_builder is None and kind.prior_value_reader is None, name
    # A routing or power-mode suggestion is valid only as a line of the report.
    for name in ("routing", "power_mode"):
        assert hc.drop_reason({"change_type": name, "target_id": SCOUT}, AUTO) is None

    prescribed = hc.prescribe(hcx.evidence(issues=[hcx.reconnect_issue(SCOUT, "Scout")]), 4, AUTO)

    (rx,) = prescribed.valid
    assert (rx["prescription_id"], rx["change_type"], rx["apply_kind"], rx["risk_score"]) == (
        "rx-4-001", "reconnect_app", False, 1)
    assert (rx["predicted_metric"], rx["guard_metrics"], rx["proposed_value"]) == (None, [], {"app_name": "GMAIL"})


def test_a_prescription_for_the_workspaces_auto_is_dropped():
    evidence = hcx.evidence(hcx.agent_row(AUTO, "Auto", **RARELY_MET), hcx.agent_row(SCOUT, "Scout", **RARELY_MET),
                            issues=[hcx.reconnect_issue(AUTO, "Auto")])

    prescribed = hc.prescribe(evidence, 5, auto_agent_id=AUTO)

    assert [(rx["prescription_id"], rx["target_id"]) for rx in prescribed.valid] == [("rx-5-002", SCOUT)]
    assert prescribed.dropped == (
        {"prescription_id": "rx-5-001", "change_type": "heartbeat_tune", "target_id": AUTO, "target_name": "Auto",
         "dropped": hc.DROP_AUTO},
        {"prescription_id": "rx-5-003", "change_type": "reconnect_app", "target_id": AUTO, "target_name": "Auto",
         "dropped": hc.DROP_AUTO},
    )
    assert prescribed.recorded() == [*prescribed.valid, *prescribed.dropped]


def _heartbeat_rx(**over):
    rx = {"change_type": "heartbeat_tune", "target_id": SCOUT, "current_value": {"interval_minutes": 60},
          "proposed_value": {"interval_minutes": 120}}
    return {**rx, **over}


@pytest.mark.parametrize("rx, reason", [
    (_heartbeat_rx(), None),
    (_heartbeat_rx(change_type="model_change"), hc.DROP_UNKNOWN_KIND),
    (_heartbeat_rx(target_id="Scout"), hc.DROP_NO_TARGET),
    (_heartbeat_rx(target_id=None), hc.DROP_NO_TARGET),
    (_heartbeat_rx(current_value={}), hc.DROP_NO_PRIOR),
    (_heartbeat_rx(proposed_value={"interval_minutes": 60}), hc.DROP_NO_CHANGE),
    (_heartbeat_rx(proposed_value={"interval_minutes": "placeholder"}), "invalid proposed value"),
    (_heartbeat_rx(proposed_value={"interval_minutes": 2880}), "invalid proposed value"),
    ({"change_type": "tool_assignment_remove", "target_id": SCOUT, "current_value": {"app_name": "GMAIL"},
      "proposed_value": {"app_name": " "}}, "invalid proposed value"),
    ({"change_type": "rollback", "target_id": SCOUT, "current_value": {"interval_minutes": 120},
      "proposed_value": {"kind": "rollback", "value": {}}}, "invalid proposed value"),
])
def test_the_validator_drops_an_invalid_prescription(rx, reason):
    found = hc.drop_reason(rx, AUTO)
    assert found == reason if reason is None else found.startswith(reason)


def test_params_and_rollback_builders_name_the_inner_call():
    heartbeat, removal, rollback = (hc.KINDS[name] for name in ("heartbeat_tune", "tool_assignment_remove",
                                                                 "rollback"))
    assert heartbeat.params_builder(SCOUT, {"interval_minutes": 120}) == hc.InnerCall(
        "platform_configure_agent_heartbeat", {"agent_id": SCOUT, "interval_minutes": 120})
    assert heartbeat.rollback_builder(SCOUT, {"interval_minutes": 60}).params["interval_minutes"] == 60
    assert removal.params_builder(SCOUT, {"app_name": "gmail"}) == hc.InnerCall(
        "platform_unassign_tool_from_agent", {"agent_id": SCOUT, "app_name": "GMAIL"})
    assert removal.rollback_builder(SCOUT, {"app_name": "GMAIL", "assigned": True}) == hc.InnerCall(
        "platform_assign_tool_to_agent", {"agent_id": SCOUT, "app_name": "GMAIL"})
    # A rollback runs the undo call of the change it reverses, and is never rolled back itself.
    undo = {"kind": "heartbeat_tune", "value": {"interval_minutes": 60}}
    assert rollback.params_builder(SCOUT, undo) == heartbeat.rollback_builder(SCOUT, undo["value"])
    with pytest.raises(ValueError):
        rollback.params_builder(SCOUT, {"kind": "rollback", "value": undo})
    with pytest.raises(ValueError):
        rollback.rollback_builder(SCOUT, undo)
    undo_actions = {heartbeat.rollback_builder(SCOUT, {"interval_minutes": 60}).action,
                    removal.rollback_builder(SCOUT, {"app_name": "GMAIL"}).action}
    assert set(rollback.inner_actions) == undo_actions


def test_kinds_with_no_evidence_wired_detect_nothing():
    """With no tool failure (US-057) and no rollback from the post-change check (US-056) in
    the evidence, neither kind detects anything; the report kinds that were placeholders
    have no detector."""
    evidence = hcx.evidence(hcx.agent_row(SCOUT, "Scout", **RARELY_MET), issues=[hcx.reconnect_issue(SCOUT, "Scout")])
    detected = {name for name, kind in hc.KINDS.items() if kind.detector(evidence)}
    assert detected == {"heartbeat_tune", "reconnect_app"}


# ── the real database ────────────────────────────────────────────────────────

@pytest.fixture
def calls(monkeypatch, tmp_path):
    return hr.fake_platform(monkeypatch, tmp_path)


@pytest.fixture
def ws(db_session, seed_workspace, calls):
    """A workspace with its Auto."""
    a = seed_workspace(name="harness-changes-registry")
    hdx.auto(db_session, a)
    return a


def test_the_tool_assignment_reader_reads_this_workspaces_assignment(db_session, ws, seed_workspace):
    from core.models.composio_cache import AgentAppAssignment

    read = hc.KINDS["tool_assignment_remove"].prior_value_reader
    scout = hdx.agent(db_session, ws, "Scout")
    db_session.add_all([AgentAppAssignment(agent_id=scout, app_name="GMAIL", is_active=True),
                        AgentAppAssignment(agent_id=scout, app_name="SLACK", is_active=False)])
    elsewhere = hdx.agent(db_session, seed_workspace(name="harness-changes-other"), "Elsewhere")
    db_session.flush()

    assert read(db_session, ws, scout, {"app_name": "gmail"}) == {"app_name": "GMAIL", "assigned": True}
    assert read(db_session, ws, scout, {"app_name": "SLACK"}) == {"app_name": "SLACK", "assigned": False}
    assert read(db_session, ws, scout, {"app_name": "NOTION"}) == {"app_name": "NOTION", "assigned": False}
    assert read(db_session, ws, elsewhere, {"app_name": "GMAIL"}) is None


def test_a_run_keeps_a_dropped_suggestion_and_its_reason_on_its_row(db_session, ws, monkeypatch):
    """Scout stands in for the workspace's Auto: its heartbeat suggestion is dropped, kept
    on the run's row with the reason, and not reported."""
    from services import harness_grants
    from services.harness_service import SUGGESTIONS_HEADING

    scout = hdx.agent(db_session, ws, "Scout", configuration={"heartbeat": {"enabled": True, "interval_minutes": 60}})
    beat = text("INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
                "objective_met, created_at) VALUES ('agent', :sid, CAST(:ws AS uuid), 'success', 10, :met, :at)")
    for met in [True] + [False] * 24:
        db_session.execute(beat, {"sid": str(scout), "ws": ws, "met": met, "at": hdx.at(0)})
    monkeypatch.setattr(harness_grants, "resolve_auto_agent", lambda db, workspace_id: {
        "id": scout, "name": "Scout"})

    run = hdx.judge(db_session, ws)

    (entry,) = run["prescriptions"]
    assert entry["prescription_id"].endswith("-001")
    assert {key: entry[key] for key in ("change_type", "target_id", "dropped")} == {
        "change_type": "heartbeat_tune", "target_id": scout, "dropped": hc.DROP_AUTO}
    assert run["counts"]["prescriptions"] == 0 and SUGGESTIONS_HEADING not in run["report_md"]
