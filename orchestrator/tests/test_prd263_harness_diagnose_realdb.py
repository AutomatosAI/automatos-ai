"""PRD-263 US-014a — HARNESS judges each agent on real counts and says when it cannot judge.

At 697ca7d6e DIAGNOSE compared four ``platform_list_agents`` fields that are never
returned against last week's baseline file with a flat 10% rule: every agent came out
STABLE and the report said "No changes needed — organization is stable." DIAGNOSE now
judges each agent x metric of the ``services.harness_metrics`` rows with the
statistics in ``core.stats.harness_verdicts`` (``services.harness_diagnose``), and
reports coverage, INSUFFICIENT_DATA, WATCH and REGRESSION.

Most cases run the real pipeline on the transactional ``db_session`` with the audit
report faked (``helpers_harness_diagnose.judge``). The judged week is 2026-W42 with
reference weeks W38-W41. The Benjamini-Hochberg case is pure. The dormancy case calls
the real check, which the pipeline cases stub. The last case runs a full tick on
committed rows and checks the workspace volume.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from core.stats import harness_verdicts as hv
from services import harness_diagnose as hd
from services import harness_metrics as hm
from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it (the full-tick case)
    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def calls(monkeypatch, tmp_path):
    return hr.fake_platform(monkeypatch, tmp_path)


@pytest.fixture
def ws(db_session, seed_workspace, calls):
    """Workspace A, with its Auto."""
    a = seed_workspace(name="harness-diagnose-a")
    hdx.auto(db_session, a)
    return a


# ── verdicts on real counts ───────────────────────────────────────────────────

def test_a_first_flag_is_watch(db_session, ws, calls):
    """16 of 20 a week for four weeks, then 12 of 20 (p=0.031): flagged, first week, WATCH."""
    barista = hdx.agent(db_session, ws, "Barista")
    for week in (-4, -3, -2, -1):
        hdx.tickets(db_session, ws, barista, week, 16, 4)
    hdx.tickets(db_session, ws, barista, 0, 12, 8)

    run = hdx.judge(db_session, ws)

    # COLLECT is one SQL read and the run calls no platform action: its one call is the audit report.
    assert calls == [(ws, hr.REPORT)]
    task = hdx.metric(run, barista)
    assert (task["verdict"], task["reason"]) == ("WATCH", None)
    assert (task["x_cur"], task["n_cur"], task["x_ref"], task["n_ref"]) == (12, 20, 64, 80)
    assert run["verdicts"]["agents"][str(barista)]["verdict"] == "WATCH"
    assert run["verdicts"]["convergence"] == "watching"
    assert "- **Barista**: Watching. Task success fell from 80% (64 of 80) to 60% (12 of 20), p=0.031" in (
        run["report_md"])


def test_a_collapse_from_30_of_32_to_10_of_32_is_flagged_on_its_first_judged_week(db_session, ws):
    """At 697ca7d6e this agent was STABLE. Its p (about 1e-7) is below HARNESS_STRONG_P,
    so FR-6 makes this first flag REGRESSION at once rather than WATCH."""
    barista = hdx.agent(db_session, ws, "Barista")
    hdx.tickets(db_session, ws, barista, -1, 30, 2)
    hdx.tickets(db_session, ws, barista, 0, 10, 22)

    task = hdx.metric(hdx.judge(db_session, ws), barista)

    assert (task["verdict"], task["reason"]) == ("REGRESSION", "strong_p")
    assert task["p_value"] < hd.settings_from_config().strong_p


def test_an_agent_with_3_tickets_is_insufficient_data(db_session, ws):
    baker = hdx.agent(db_session, ws, "Baker")
    hdx.tickets(db_session, ws, baker, 0, 2, 1)

    run = hdx.judge(db_session, ws)

    entry = run["verdicts"]["agents"][str(baker)]
    assert entry["verdict"] == "INSUFFICIENT_DATA"
    assert {m["verdict"] for m in entry["metrics"].values()} == {"INSUFFICIENT_DATA"}
    assert (hdx.metric(run, baker)["n_cur"], hdx.metric(run, baker)["p_value"]) == (3, None)
    assert "- **Baker**: Not enough data (3 Tasks this week, 0 in the reference; each needs 20" in run["report_md"]


def test_16_of_20_against_14_of_20_is_stable(db_session, ws):
    roaster = hdx.agent(db_session, ws, "Roaster")
    hdx.tickets(db_session, ws, roaster, -1, 16, 4)
    hdx.tickets(db_session, ws, roaster, 0, 14, 6)

    run = hdx.judge(db_session, ws)

    assert hdx.metric(run, roaster)["verdict"] == "STABLE"
    assert run["verdicts"]["agents"][str(roaster)]["verdict"] == "STABLE"
    assert run["verdicts"]["convergence"] == "steady"


def test_the_heartbeat_interval_comes_from_configuration_and_its_cost_from_whitelisted_usage(db_session, ws):
    beating = hdx.agent(db_session, ws, "Night watch", model_config={"model_id": "anthropic/claude-haiku-5.5"},
                        configuration={"heartbeat": {"enabled": True, "interval_minutes": 30}})
    quiet = hdx.agent(db_session, ws, "Quiet")
    for _ in range(3):
        hdx.usage(db_session, ws, beating, hdx.at(0), request_type="heartbeat", cost=0.01, tokens=100)
    hdx.usage(db_session, ws, beating, hdx.at(0), request_type="heartbeat", status="reserved", cost=9.0, tokens=9000)
    hdx.usage(db_session, ws, beating, hdx.at(0), request_type="memory_distill", cost=9.0, tokens=9000)
    db_session.flush()

    agents = hdx.judge(db_session, ws)["verdicts"]["agents"]

    night = agents[str(beating)]
    assert (night["heartbeat_enabled"], night["heartbeat_interval_minutes"]) == (True, 30)
    assert night["heartbeat_cost_usd"] == pytest.approx(0.03)
    assert night["model_id"] == "anthropic/claude-haiku-5.5"   # model_id, not the absent "model"
    assert (agents[str(quiet)]["heartbeat_enabled"], agents[str(quiet)]["heartbeat_interval_minutes"]) == (False, 60)


def test_an_orchestrator_heartbeat_row_with_a_uuid_source_id_does_not_break_the_run(db_session, ws):
    watcher = hdx.agent(db_session, ws, "Watcher")
    beat = text("INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
                "objective_met, created_at) VALUES (:st, :sid, CAST(:ws AS uuid), 'success', 10, :met, :at)")
    db_session.execute(beat, {"st": "orchestrator", "sid": ws, "ws": ws, "met": False, "at": hdx.at(0)})
    for met in (True, True, False, None):
        db_session.execute(beat, {"st": "agent", "sid": str(watcher), "ws": ws, "met": met, "at": hdx.at(0)})

    objective = hdx.metric(hdx.judge(db_session, ws), watcher, "heartbeat_objective")

    assert (objective["x_cur"], objective["n_cur"]) == (2, 3)   # agent rows only; None is not in the denominator


# ── cost and tokens per run ───────────────────────────────────────────────────

def test_a_30_percent_median_cost_rise_of_at_least_half_a_cent_flags(db_session, ws):
    spender = hdx.agent(db_session, ws, "Spender")
    hdx.runs(db_session, ws, spender, -1, 24, cost=0.020)
    hdx.runs(db_session, ws, spender, 0, 24, cost=0.026)

    cost = hdx.metric(hdx.judge(db_session, ws), spender, "run_cost")

    assert cost["unit"] == "usd" and cost["verdict"] in ("WATCH", "REGRESSION")
    assert cost["p_value"] < hd.settings_from_config().amounts.alpha


def test_a_30_percent_rise_of_three_tenths_of_a_cent_does_not_flag(db_session, ws):
    saver = hdx.agent(db_session, ws, "Saver")
    hdx.runs(db_session, ws, saver, -1, 24, cost=0.010)
    hdx.runs(db_session, ws, saver, 0, 24, cost=0.013)

    cost = hdx.metric(hdx.judge(db_session, ws), saver, "run_cost")

    assert (cost["unit"], cost["verdict"]) == ("usd", "STABLE")
    assert cost["p_value"] < hd.settings_from_config().amounts.alpha   # significant, but under $0.005


def test_an_agent_with_no_billed_runs_is_judged_on_tokens(db_session, ws):
    cli = hdx.agent(db_session, ws, "Claude Code agent")
    hdx.runs(db_session, ws, cli, -1, 24, tokens=1000, tier=hdx.PLAN_TIER)
    hdx.runs(db_session, ws, cli, 0, 24, tokens=1300, tier=hdx.PLAN_TIER)

    run = hdx.judge(db_session, ws)

    tokens = hdx.metric(run, cli, "run_cost")
    assert tokens["unit"] == "tokens" and tokens["verdict"] in ("WATCH", "REGRESSION")
    assert (tokens["ref_value"], tokens["cur_value"]) == (1000, 1300)
    assert "Tokens per run rose from 1,000 to 1,300" in run["report_md"]


# ── Benjamini-Hochberg across the run (pure) ──────────────────────────────────

def _row(agent_id: int, week: str, done: int, total: int) -> Dict[str, Any]:
    """An agent-week with only tickets: ``done`` of ``total`` succeeded."""
    blank = {field: () if "_run_" in field else 0 for field in hm.AGENT_ROW_FIELDS}
    return {**blank, "agent_id": agent_id, "agent_name": f"Agent {agent_id}", "model_id": None,
            "heartbeat_enabled": False, "heartbeat_interval_minutes": 60, "iso_week": week,
            "tasks_done": done, "tasks_failed": total - done}


def _lone_verdict(lone_ref_done: int) -> str:
    """Ten agents with one judgeable metric each: agent 1 at 12 of 20 this week against
    ``lone_ref_done`` of 20, nine others steady at 16 of 20. Agent 1's verdict."""
    window = hm.build_window(hdx.WEEK, 1)
    (ref_week,) = window.ref_weeks
    rows = [_row(1, ref_week, lone_ref_done, 20), _row(1, hdx.WEEK, 12, 20)]
    for agent_id in range(2, 11):
        rows += [_row(agent_id, ref_week, 16, 20), _row(agent_id, hdx.WEEK, 16, 20)]
    metrics = hm.HarnessMetrics(workspace_id="00000000-0000-0000-0000-000000000001", window=window,
                                agent_rows=tuple(rows), tool_rows=())
    diagnosis = hd.diagnose(metrics, {}, {}, hd.settings_from_config())
    assert diagnosis["counts"]["judged_agents"] == 10
    return diagnosis["agents"]["1"]["metrics"]["task_success"]["verdict"]


def test_a_lone_p_of_0_04_among_ten_tests_is_not_flagged_after_benjamini_hochberg():
    """Rank 1 of 10 at q=0.10 must clear 0.01: 12 of 20 against 17 of 20 (p=0.038) flags
    on its own and not after BH; 12 of 20 against 19 of 20 (p=0.004) flags after BH too."""
    settings = hd.settings_from_config()
    assert (settings.fdr_q, settings.rates.min_samples) == (0.10, 20)
    alone = hv.proportion_flag(12, 20, 17, 20, settings.rates)
    assert alone.flagged and 0.03 < alone.p_value < 0.05
    strong = hv.proportion_flag(12, 20, 19, 20, settings.rates)
    assert strong.flagged and 0.001 < strong.p_value < 0.01

    assert _lone_verdict(17) == "STABLE"
    assert _lone_verdict(19) == "WATCH"


# ── coverage, the report and the status ───────────────────────────────────────

def test_coverage_below_half_reads_insufficient_data_and_the_report_never_says_stable_or_converged(db_session, ws):
    from modules.tools.discovery.handlers_harness import harness_status

    judged = hdx.agent(db_session, ws, "Roaster")
    hdx.tickets(db_session, ws, judged, -1, 16, 4)
    hdx.tickets(db_session, ws, judged, 0, 16, 4)
    thin = hdx.agent(db_session, ws, "Baker")
    hdx.tickets(db_session, ws, thin, 0, 3, 0)
    hdx.agent(db_session, ws, "Idle")

    run = hdx.judge(db_session, ws)

    assert run["coverage"] == pytest.approx(1 / 3)
    assert run["verdicts"]["convergence"] == "insufficient_data"
    report = run["report_md"]
    assert "HARNESS judged 1 of 3 agents (33%)" in report
    assert "stable" not in report.lower() and "converged" not in report.lower()
    lines = report.splitlines()
    assert any(line.startswith("- **Baker**: Not enough data (3 Tasks this week") for line in lines)
    assert any(line.startswith("- **Idle**: Not enough data (nothing recorded;") for line in lines)
    assert "- **Roaster**: Steady. Judged on Task success (n=20, reference 20)." in lines

    status = asyncio.run(harness_status(db_session, UUID(ws), {}))["data"]
    assert status["status"] == "completed" and status["coverage"] == pytest.approx(1 / 3)
    assert (status["convergence"], status["insufficient_data"], status["watch"], status["regression"]) == (
        "insufficient_data", 2, 0, 0)
    assert (status["judged_agents"], status["active_agents"]) == (1, 3)


def test_workspace_b_never_appears(db_session, ws, seed_workspace):
    mine = hdx.agent(db_session, ws, "Barista")
    hdx.tickets(db_session, ws, mine, -1, 16, 4)
    hdx.tickets(db_session, ws, mine, 0, 16, 4)
    b = seed_workspace(name="harness-diagnose-b")
    theirs = hdx.agent(db_session, b, "Their barista")
    for agent_id in (theirs, mine):   # B's tickets, one set carrying A's agent id
        hdx.tickets(db_session, b, agent_id, 0, 5, 15)

    run = hdx.judge(db_session, ws)

    assert set(run["verdicts"]["agents"]) == {str(mine)}
    assert (hdx.metric(run, mine)["x_cur"], hdx.metric(run, mine)["n_cur"]) == (16, 20)
    assert "Their barista" not in run["report_md"]


def test_a_workspace_with_no_heartbeat_history_is_not_dormant(db_session, seed_workspace):
    """Dormancy is the active-agent count alone (Auto included). Before PR-1D a workspace
    with no heartbeat_results rows read dormant_insufficient_data whatever its agents."""
    from services.harness_service import HarnessService

    workspace = seed_workspace(name="harness-dormancy")
    hdx.auto(db_session, workspace)
    for name in ("Barista", "Baker", "Roaster"):
        hdx.agent(db_session, workspace, name)
    service = HarnessService()

    breakdown = service._sufficiency_breakdown(db_session, UUID(workspace))

    assert breakdown == {"active_agents": 4, "min_required_agents": 3, "agents_ok": True}
    assert service._dormancy(breakdown) is None


# ── the full tick writes nothing to the volume ────────────────────────────────

def test_a_full_tick_writes_no_file_under_the_workspace_harness_folder(new_session, calls, tmp_path):
    from config import config
    from services.harness_service import HarnessService

    workspace = hr.seed_workspace(new_session)
    try:
        s = new_session()
        barista = hdx.agent(s, workspace, "Barista")
        hdx.tickets(s, workspace, barista, -1, 16, 4)
        hdx.tickets(s, workspace, barista, 0, 14, 6)
        s.commit()
        s.close()

        asyncio.run(HarnessService()._harness_tick(workspace))

        (run,) = hr.runs(new_session, workspace)
        assert (run["status"], run["cur_window"], run["report_status"]) == ("completed", hr.WINDOW, "ok")
        assert run["coverage"] == pytest.approx(1.0)
        assert run["verdicts"]["agents"][str(barista)]["metrics"]["task_success"]["verdict"] == "STABLE"
        assert config.WORKSPACE_VOLUME_PATH == str(tmp_path)
        assert not (tmp_path / str(workspace) / "harness").exists()
        assert not any(tmp_path.rglob("*"))
    finally:
        hr.sweep_workspace(new_session, workspace)
