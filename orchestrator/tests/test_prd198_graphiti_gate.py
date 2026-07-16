"""PRD-198 S1 — the Graphiti-vs-baseline A/B gate.

The instrument lands before any Graphiti code: the gate must read an
honest PENDING until both baselines are frozen and a treatment run
exists, and its arithmetic must be the operating_graph_uplift honest
shape (treatment − best_baseline, in points). Pure — bundled/fabricated
artifacts + stdlib; no Postgres, no LLM, no graph store.
"""
import sys
from pathlib import Path

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from evals.graphiti_vs_baseline import (  # noqa: E402
    DEFAULT_MEMORY_BASELINE,
    DEFAULT_RETRIEVAL_BASELINE,
    TREATMENT_VARIANT,
    compute_gate,
    load_artifact,
    uplift_points,
)


def _artifact(alias: str, variants: dict) -> dict:
    return {alias: {"variants": {k: {"mean_recall_at_5": v} for k, v in variants.items()}}}


def test_graphiti_variant_registered():
    """The treatment variant is 'graphiti' and the gate consumes artifacts
    in the exact frozen-baseline shape — freezing once serves the wave."""
    assert TREATMENT_VARIANT == "graphiti"
    gate = compute_gate(
        _artifact("pilot-a", {"baseline": 0.65, "hybrid_rrf": 0.692}),
        {"any": "memory-baseline"},
        _artifact("pilot-a", {"graphiti": 0.80}),
    )
    assert gate["verdict"] in ("ADOPT_UNBLOCKED", "DO_NOT_ADOPT")
    assert gate["tenants"][0]["graphiti_recall_at_5"] == 0.80


def test_uplift_gate_pending_without_frozen_baseline():
    """Absent inputs ⇒ PENDING, naming what is missing — never a false
    green, and never a verdict computed from a strawman."""
    gate = compute_gate(None, None, None)
    assert gate["verdict"] == "PENDING"
    assert any("retrieval_baseline" in m for m in gate["missing"])
    assert any("memory_baseline_s10" in m for m in gate["missing"])
    assert any("graphiti_treatment" in m for m in gate["missing"])

    # A frozen retrieval baseline alone is still PENDING (S10 + treatment).
    gate = compute_gate(
        _artifact("pilot-a", {"hybrid_rrf": 0.692}), None, None
    )
    assert gate["verdict"] == "PENDING"
    assert not any("retrieval_baseline" in m for m in gate["missing"])


def test_uplift_points_treatment_minus_baseline():
    """The honest-gate arithmetic: points = (treatment − BEST baseline)×100;
    the margin decides adopt-vs-no-op."""
    assert uplift_points(0.75, 0.692) == pytest.approx(5.8)

    retrieval = _artifact("pilot-a", {"baseline": 0.65, "rerank": 0.769, "hybrid_rrf": 0.692})
    memory = {"frozen": True}

    winning = compute_gate(retrieval, memory, _artifact("pilot-a", {"graphiti": 0.83}))
    # best baseline is rerank (0.769), not the plain baseline — no strawman
    assert winning["tenants"][0]["best_baseline_recall_at_5"] == 0.769
    assert winning["tenants"][0]["uplift_points"] == pytest.approx(6.1)
    assert winning["verdict"] == "ADOPT_UNBLOCKED"
    assert winning["pending_slices"]  # a recall win alone never claims the slices

    losing = compute_gate(retrieval, memory, _artifact("pilot-a", {"graphiti": 0.78}))
    assert losing["verdict"] == "DO_NOT_ADOPT"
    assert losing["tenants"][0]["uplift_points"] == pytest.approx(1.1)


def test_repo_state_is_honest_today():
    """Pins the current truth: the retrieval baseline is FROZEN on main
    (#547), the S10 memory baseline and the treatment are not — so the
    committed default state of the gate is PENDING."""
    retrieval = load_artifact(DEFAULT_RETRIEVAL_BASELINE)
    assert retrieval is not None, "the #547 frozen retrieval baseline should exist"

    gate = compute_gate(retrieval, load_artifact(DEFAULT_MEMORY_BASELINE), None)
    assert gate["verdict"] == "PENDING"
    assert any("graphiti_treatment" in m for m in gate["missing"])
