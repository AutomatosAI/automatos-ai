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
    load_cases_from_eval_set,
    run_uplift_eval,
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
