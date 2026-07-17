"""PRD-198 S1 — the Graphiti-vs-baseline A/B gate (the trial's instrument).

Built BEFORE any Graphiti code by design (§5: S1 first and blocking). The
whole PRD is eval-gated: Graphiti is adopted only if it beats the repaired
baseline by the stated margin, so the gate must exist first and must read
an honest **PENDING** — never a false green — until every input it needs
exists:

- **retrieval baseline** — ``evals/baseline/kg_retrieval_2026-07.json``:
  the pilot-a live freeze (FROZEN, landed with PRD-186 #547).
- **memory baseline (S10)** — ``evals/baseline/memory_recall_2026-07.json``:
  NOT yet frozen; Gerard's live run per
  ``docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md``. Its absence is the
  ⏸ PENDING BASELINE gate that blocks S2–S6.
- **graphiti treatment** — ``evals/results/graphiti_recall.json``: produced
  by a live retrieval-recall run with the graphiti lever once S2 stands the
  trial up. Same artifact shape as the frozen baseline, so freezing once
  serves the whole wave.

Verdict shape reuses the ``operating_graph_uplift`` honest gate:
``uplift_points = (treatment − best_baseline) × 100``, mean across tenant
aliases, published, **exit 0 always** — a sub-margin uplift is a valid,
honest outcome (the flag stays OFF and the trial is a documented no-op).

Bar (§8-Q1 CONFIRMED by Gerard 2026-07-17): ≥ +5.0 points recall@5 mean
across tenants **AND** the contradicted-fact slice resolves to one live
fact. Margin alone never adopts — temporal invalidation is the capability
with no in-house substitute. The other slices (dup-rate, multi-hop) are
reported, not gating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent

TREATMENT_VARIANT = "graphiti"
UPLIFT_MARGIN_POINTS = 5.0  # §8-Q1 proposal — confirm before the verdict binds
DEFAULT_TENANT_ALIASES: Sequence[str] = ("pilot-a",)

DEFAULT_RETRIEVAL_BASELINE = _HERE / "baseline" / "kg_retrieval_2026-07.json"
DEFAULT_MEMORY_BASELINE = _HERE / "baseline" / "memory_recall_2026-07.json"
DEFAULT_TREATMENT = _HERE / "results" / "graphiti_recall.json"

# Capability slices (measured by S3/S4 once the trial runs). The
# contradiction slice GATES adoption (Gerard 2026-07-17); the others are
# reported. Treatment artifacts carry them as
# {"slices": {"contradicted_fact_resolution": {"passed": bool}, ...}}.
GATING_SLICE = "contradicted_fact_resolution"
REPORTED_SLICES = (
    "duplicate_node_rate",
    "multi_hop_answer_quality",
)
PENDING_SLICES = (GATING_SLICE,) + REPORTED_SLICES


def load_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Load a frozen/treatment artifact; None when absent (→ PENDING)."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def best_baseline_recall(artifact: Dict[str, Any], tenant_alias: str) -> Optional[float]:
    """Best shipped-variant recall@5 for one tenant alias — the honest-gate
    ``best_baseline`` (the treatment must beat the best of what already
    ships, not a strawman). The graphiti variant itself is excluded."""
    variants = ((artifact.get(tenant_alias) or {}).get("variants") or {})
    scores = [
        float(row.get("mean_recall_at_5", 0.0))
        for name, row in variants.items()
        if name != TREATMENT_VARIANT and row
    ]
    return max(scores) if scores else None


def treatment_recall(artifact: Dict[str, Any], tenant_alias: str) -> Optional[float]:
    """The graphiti variant's recall@5 for one tenant alias, if measured."""
    variants = ((artifact.get(tenant_alias) or {}).get("variants") or {})
    row = variants.get(TREATMENT_VARIANT)
    if not row:
        return None
    return float(row.get("mean_recall_at_5", 0.0))


def uplift_points(treatment: float, baseline: float) -> float:
    """treatment − baseline, in points (the operating_graph_uplift shape)."""
    return round((treatment - baseline) * 100.0, 2)


def compute_gate(
    retrieval_baseline: Optional[Dict[str, Any]],
    memory_baseline: Optional[Dict[str, Any]],
    treatment: Optional[Dict[str, Any]],
    tenant_aliases: Sequence[str] = DEFAULT_TENANT_ALIASES,
    margin_points: float = UPLIFT_MARGIN_POINTS,
) -> Dict[str, Any]:
    """The gate. PENDING while any input is missing; otherwise the verdict.

    A PENDING gate is not a failure — it is the ⏸ state the PRD's own §8
    box defines: the build (S2–S6) does not start until the baselines are
    frozen, and the adopt decision does not exist until the treatment ran.
    """
    missing: List[str] = []
    if retrieval_baseline is None:
        missing.append("retrieval_baseline (evals/baseline/kg_retrieval_2026-07.json)")
    if memory_baseline is None:
        missing.append(
            "memory_baseline_s10 (evals/baseline/memory_recall_2026-07.json — "
            "docs/runbooks/S10-MEMORY-BASELINE-FREEZE.md)"
        )
    if treatment is None:
        missing.append("graphiti_treatment (evals/results/graphiti_recall.json — S2 gated)")

    if missing:
        return {
            "verdict": "PENDING",
            "reason": "baseline not frozen / treatment not run — no false greens",
            "missing": missing,
            "margin_points": margin_points,
            "pending_slices": list(PENDING_SLICES),
        }

    tenants: List[Dict[str, Any]] = []
    for alias in tenant_aliases:
        baseline_r5 = best_baseline_recall(retrieval_baseline, alias)
        treat_r5 = treatment_recall(treatment, alias)
        if baseline_r5 is None or treat_r5 is None:
            return {
                "verdict": "PENDING",
                "reason": f"tenant '{alias}' lacks a baseline or treatment number",
                "missing": [alias],
                "margin_points": margin_points,
                "pending_slices": list(PENDING_SLICES),
            }
        tenants.append(
            {
                "tenant": alias,
                "best_baseline_recall_at_5": round(baseline_r5, 4),
                "graphiti_recall_at_5": round(treat_r5, 4),
                "uplift_points": uplift_points(treat_r5, baseline_r5),
            }
        )

    mean_uplift = round(sum(t["uplift_points"] for t in tenants) / len(tenants), 2)
    beats_margin = mean_uplift >= margin_points

    # §8-Q1 (CONFIRMED 2026-07-17): adoption requires the margin AND the
    # contradiction slice. Margin met with the slice unmeasured is PENDING
    # (the trial hasn't produced its gating evidence), never an adopt.
    slices = (treatment or {}).get("slices") or {}
    gating = slices.get(GATING_SLICE)

    if beats_margin and gating is None:
        verdict = "PENDING"
        note = (
            "recall margin met but the gating contradiction slice is "
            "unmeasured (S3) — adoption cannot fire on the aggregate alone"
        )
    elif beats_margin and bool(gating.get("passed")):
        verdict = "ADOPT_UNBLOCKED"
        note = (
            "margin AND contradiction slice met — the adopt follow-through "
            "(retire the in-house merge path) is unblocked as a §8 decision"
        )
    elif beats_margin:
        verdict = "DO_NOT_ADOPT"
        note = (
            "recall margin met but the contradiction slice FAILED — temporal "
            "invalidation is the point; flag stays OFF"
        )
    else:
        verdict = "DO_NOT_ADOPT"
        note = "below margin — flag stays OFF; a documented no-op is a valid outcome"

    return {
        "verdict": verdict,
        "mean_uplift_points": mean_uplift,
        "margin_points": margin_points,
        "gating_slice": {GATING_SLICE: gating},
        "tenants": tenants,
        "pending_slices": [name for name in PENDING_SLICES if name not in slices],
        "note": note,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Graphiti-vs-baseline A/B gate (PRD-198 S1 — exit 0 always)"
    )
    parser.add_argument("--retrieval-baseline", type=Path, default=DEFAULT_RETRIEVAL_BASELINE)
    parser.add_argument("--memory-baseline", type=Path, default=DEFAULT_MEMORY_BASELINE)
    parser.add_argument("--treatment", type=Path, default=DEFAULT_TREATMENT)
    parser.add_argument("--margin", type=float, default=UPLIFT_MARGIN_POINTS)
    args = parser.parse_args(argv)

    gate = compute_gate(
        load_artifact(args.retrieval_baseline),
        load_artifact(args.memory_baseline),
        load_artifact(args.treatment),
        margin_points=args.margin,
    )
    print(json.dumps(gate, indent=2))
    return 0  # the number (or the PENDING) is the deliverable


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
