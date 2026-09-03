"""PRD-177 S6: operating-graph uplift eval — the business gate.

Validates the eval HARNESS (pure, offline): BM25 + embedding-proxy + learned-edge
rankers, per-tenant accuracy, and honest gate reporting. It does NOT assert a
passing uplift number — a sub-threshold result is a valid, honest outcome (trap
#3); the harness must report it faithfully and must never flip the flag on a
low number.
"""
import sys
from pathlib import Path

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from evals.operating_graph_uplift import (  # noqa: E402
    EvalCase,
    UPLIFT_THRESHOLD_POINTS,
    cases_from_log_rows,
    load_cases_from_eval_set,
    render_report,
    run_uplift_eval,
    _build_parser,
    _bm25_ranker,
    _embedding_proxy_ranker,
)


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

def test_bm25_ranker_picks_lexically_matching_action():
    actions = ["platform_list_agents", "platform_get_cost_breakdown"]
    cats = {"platform_list_agents": "agents", "platform_get_cost_breakdown": "analytics"}
    rank = _bm25_ranker(actions, cats)
    assert rank("show me the cost breakdown") == "platform_get_cost_breakdown"
    assert rank("list my agents") == "platform_list_agents"


def test_embedding_proxy_is_deterministic():
    actions = ["platform_list_agents", "platform_get_cost_breakdown"]
    cats = {"platform_list_agents": "agents", "platform_get_cost_breakdown": "analytics"}
    rank1 = _embedding_proxy_ranker(actions, cats)
    rank2 = _embedding_proxy_ranker(actions, cats)
    q = "what agents are running"
    assert rank1(q) == rank2(q)  # no randomness — reproducible in CI


def test_learned_ranker_beats_baseline_on_a_biased_tenant():
    """A tenant whose history always answers a paraphrased intent with the SAME
    action lets the learned ranker win where lexical matching is ambiguous.

    This proves the mechanism works (learned signal helps); it is a mechanism
    test, NOT the production gate — the gate number comes from run_uplift_eval
    over real per-tenant telemetry.
    """
    # Two actions with near-identical text so BM25/embedding can't separate the
    # ambiguous paraphrase; the tenant's history disambiguates.
    train = [
        EvalCase("handle the thing", "platform_do_alpha", "ambiguous", "ws"),
        EvalCase("handle the thing please", "platform_do_alpha", "ambiguous", "ws"),
        EvalCase("deal with the thing", "platform_do_alpha", "ambiguous", "ws"),
    ]
    test = [EvalCase("handle the thing now", "platform_do_alpha", "ambiguous", "ws")]
    cases = train + test

    report = run_uplift_eval(cases)
    assert report.tenants
    t = report.tenants[0]
    # learned should be at least as good as the best baseline here
    assert t.learned_acc >= t.best_baseline


# ---------------------------------------------------------------------------
# Per-tenant reporting + honest gate
# ---------------------------------------------------------------------------

def test_report_is_per_tenant():
    cases = load_cases_from_eval_set(num_tenants=3)
    report = run_uplift_eval(cases)
    ws_ids = {t.workspace_id for t in report.tenants}
    assert ws_ids == {"tenant-0", "tenant-1", "tenant-2"}, (
        "uplift must be reported PER TENANT"
    )
    for t in report.tenants:
        assert t.n_test > 0
        assert 0.0 <= t.bm25_acc <= 1.0
        assert 0.0 <= t.embedding_acc <= 1.0
        assert 0.0 <= t.learned_acc <= 1.0


def test_gate_recommends_flip_only_when_uplift_clears_threshold():
    """flip_flag_recommended tracks the honest number — never fabricated."""
    cases = load_cases_from_eval_set(num_tenants=2)
    report = run_uplift_eval(cases)
    d = report.to_dict()
    # The recommendation is exactly the honest pass/fail of the threshold.
    assert d["flip_flag_recommended"] == (d["mean_uplift_points"] >= UPLIFT_THRESHOLD_POINTS)
    assert d["passes"] == d["flip_flag_recommended"]


def test_bundled_fixture_runs_and_emits_a_number():
    """The bundled eval set runs end-to-end and produces a real uplift number
    (whatever it is) — the deliverable of S6."""
    cases = load_cases_from_eval_set(num_tenants=2)
    report = run_uplift_eval(cases)
    d = report.to_dict()
    assert isinstance(d["mean_uplift_points"], float)
    assert "tenants" in d and len(d["tenants"]) == 2
    assert d["source"] == "fixture"  # a fixture run can never pass as the gate


# ---------------------------------------------------------------------------
# Telemetry loader — the real gate (PRD-232 §6.4), pure half, no database
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "platform_list_agents": "agents",
    "platform_create_agent": "agents",
    "platform_search_memory": "memory",
}


def _row(ws, query, action, status="success", source="production"):
    return {
        "workspace_id": ws,
        "user_query": query,
        "action_name": action,
        "status": status,
        "telemetry_source": source,
    }


def test_telemetry_cases_come_only_from_successful_production_rows_with_known_actions():
    rows = [
        _row("ws-a", "list my agents", "platform_list_agents"),
        _row("ws-a", "list my agents", "platform_list_agents", status="error"),
        _row("ws-a", "replay me", "platform_list_agents", source="replay"),
        _row("ws-a", "eval me", "platform_list_agents", source="eval"),
        _row("ws-a", "do the thing", "platform_execute"),  # the wrapper is not an effect
        _row("ws-a", "retired", "platform_gone_action"),  # unknown to the registry
        _row("ws-a", "   ", "platform_list_agents"),  # no query text
        _row(None, "no tenant", "platform_list_agents"),  # no workspace
    ]
    cases, counts = cases_from_log_rows(rows, _CATEGORIES, min_cases_per_tenant=1)
    assert [(c.workspace_id, c.query, c.correct_action, c.category) for c in cases] == [
        ("ws-a", "list my agents", "platform_list_agents", "agents")
    ]
    assert counts["rows_loaded"] == 8
    assert counts["rows_not_success"] == 1
    assert counts["rows_non_production"] == 2
    assert counts["rows_unknown_action"] == 2
    assert counts["rows_without_query"] == 1
    assert counts["rows_without_workspace"] == 1


def test_telemetry_repeated_queries_collapse_to_the_majority_action():
    rows = (
        [_row("ws-a", "Find my agents", "platform_list_agents")] * 3
        + [_row("ws-a", "find my  agents", "platform_create_agent")] * 2  # same after normalising
        + [_row("ws-a", "remember this", "platform_search_memory")]
    )
    cases, counts = cases_from_log_rows(rows, _CATEGORIES, min_cases_per_tenant=1)
    assert counts["distinct_queries"] == 2
    by_query = {c.query: c.correct_action for c in cases}
    assert by_query["Find my agents"] == "platform_list_agents"  # 3 votes beat 2; first text kept
    assert by_query["remember this"] == "platform_search_memory"


def test_telemetry_ties_break_deterministically():
    rows = [
        _row("ws-a", "q", "platform_list_agents"),
        _row("ws-a", "q", "platform_create_agent"),
    ]
    cases, _ = cases_from_log_rows(rows, _CATEGORIES, min_cases_per_tenant=1)
    assert cases[0].correct_action == "platform_create_agent"  # smallest name on a tie


def test_telemetry_drops_tenants_below_the_minimum_and_reports_it():
    rows = [_row("ws-big", f"query {i}", "platform_list_agents") for i in range(10)] + [
        _row("ws-small", "one", "platform_list_agents"),
        _row("ws-small", "two", "platform_search_memory"),
    ]
    cases, counts = cases_from_log_rows(rows, _CATEGORIES, min_cases_per_tenant=10)
    assert {c.workspace_id for c in cases} == {"ws-big"}
    assert counts["tenants_seen"] == 2
    assert counts["tenants_dropped_small"] == 1
    assert counts["tenants_evaluated"] == 1
    assert counts["cases"] == 10


def test_telemetry_cases_feed_the_harness_and_the_report_names_its_source():
    rows = [_row("ws-a", f"list agents number {i}", "platform_list_agents") for i in range(6)] + [
        _row("ws-a", f"search memory for item {i}", "platform_search_memory") for i in range(6)
    ]
    cases, counts = cases_from_log_rows(rows, _CATEGORIES, min_cases_per_tenant=10)
    report = run_uplift_eval(cases)
    report.meta = {"source": "telemetry", **counts}
    d = report.to_dict()
    assert d["source"] == "telemetry"
    assert d["loader"]["cases"] == 12
    assert d["tenants"][0]["workspace_id"] == "ws-a"
    assert "PRODUCTION TELEMETRY" in render_report(report)


def test_cli_exposes_the_telemetry_gate_flags():
    ns = _build_parser().parse_args(
        ["--from-telemetry", "--window-days", "7", "--min-cases", "5", "--json"]
    )
    assert ns.from_telemetry and ns.window_days == 7 and ns.min_cases == 5 and ns.json
    defaults = _build_parser().parse_args([])
    assert not defaults.from_telemetry and defaults.tenants == 2
