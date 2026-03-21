"""
A/B Experiment Report — PRD-108
================================
Compares metrics from a vector_field run vs a redis run.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from modules.context.instrumentation import ExperimentMetrics


@dataclass(frozen=True)
class ComparisonResult:
    """Immutable comparison between two backend runs."""
    vector_field_metrics: dict[str, Any]
    redis_metrics: dict[str, Any]

    # Deltas (positive = vector_field wins)
    avg_results_delta: float          # More results per query = better coverage
    avg_latency_delta_ms: float       # Negative = vector_field faster
    total_injections_match: bool      # Should be equal (same mission)
    total_queries_match: bool         # Should be equal (same mission)

    verdict: str  # One-line summary


def compare_runs(
    vf_metrics: ExperimentMetrics,
    redis_metrics: ExperimentMetrics,
) -> ComparisonResult:
    """Compare a vector_field run against a redis baseline run."""

    vf = vf_metrics.to_dict()
    rd = redis_metrics.to_dict()

    avg_results_delta = vf["avg_results_per_query"] - rd["avg_results_per_query"]
    avg_latency_delta = vf["avg_query_latency_ms"] - rd["avg_query_latency_ms"]

    # Verdict
    wins = []
    losses = []
    if avg_results_delta > 0:
        wins.append(f"+{avg_results_delta:.1f} results/query")
    else:
        losses.append(f"{avg_results_delta:.1f} results/query")

    if avg_latency_delta < 0:
        wins.append(f"{avg_latency_delta:.1f}ms faster")
    else:
        losses.append(f"+{avg_latency_delta:.1f}ms slower")

    if wins and not losses:
        verdict = f"Vector field wins: {', '.join(wins)}"
    elif losses and not wins:
        verdict = f"Redis wins: {', '.join(losses)}"
    else:
        verdict = f"Mixed: vector field {', '.join(wins)}; redis {', '.join(losses)}"

    return ComparisonResult(
        vector_field_metrics=vf,
        redis_metrics=rd,
        avg_results_delta=avg_results_delta,
        avg_latency_delta_ms=avg_latency_delta,
        total_injections_match=vf["total_injections"] == rd["total_injections"],
        total_queries_match=vf["total_queries"] == rd["total_queries"],
        verdict=verdict,
    )


def format_report(result: ComparisonResult) -> str:
    """Format comparison as markdown for logging or display."""
    vf = result.vector_field_metrics
    rd = result.redis_metrics

    lines = [
        "# PRD-108 A/B Experiment Results",
        "",
        "| Metric | Vector Field | Redis Baseline | Delta |",
        "|--------|-------------|----------------|-------|",
        f"| Injections | {vf['total_injections']} | {rd['total_injections']} | {'MATCH' if result.total_injections_match else 'MISMATCH'} |",
        f"| Queries | {vf['total_queries']} | {rd['total_queries']} | {'MATCH' if result.total_queries_match else 'MISMATCH'} |",
        f"| Avg results/query | {vf['avg_results_per_query']:.1f} | {rd['avg_results_per_query']:.1f} | {result.avg_results_delta:+.1f} |",
        f"| Avg query latency | {vf['avg_query_latency_ms']:.1f}ms | {rd['avg_query_latency_ms']:.1f}ms | {result.avg_latency_delta_ms:+.1f}ms |",
        "",
        f"**Verdict:** {result.verdict}",
    ]
    return "\n".join(lines)
