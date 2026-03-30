#!/usr/bin/env python3
"""
Compare benchmark results from two different backend runs.

Usage:
  python tools/compare_benchmarks.py tools/benchmark_results/
"""

import json
import sys
from pathlib import Path


def load_latest_per_label(results_dir: Path) -> dict[str, dict]:
    """Load the most recent result file per label."""
    by_label: dict[str, tuple[str, dict]] = {}
    for f in results_dir.glob("benchmark_*.json"):
        data = json.loads(f.read_text())
        label = data["label"]
        ts = data["timestamp"]
        if label not in by_label or ts > by_label[label][0]:
            by_label[label] = (ts, data)
    return {label: data for label, (_, data) in by_label.items()}


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/compare_benchmarks.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    runs = load_latest_per_label(results_dir)

    if len(runs) < 2:
        print(f"Need at least 2 different backends. Found: {list(runs.keys())}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("FIELD MEMORY BENCHMARK — COMPARISON")
    print(f"{'='*70}\n")

    # Header
    labels = sorted(runs.keys())
    header = f"{'Metric':<25s}"
    for label in labels:
        header += f"  {label:>15s}"
    if len(labels) == 2:
        header += f"  {'Delta':>10s}"
    print(header)
    print("-" * len(header))

    # Coverage
    coverages = {}
    for label in labels:
        coverages[label] = runs[label]["summary"]["avg_coverage"]

    row = f"{'Coverage (avg)':<25s}"
    for label in labels:
        row += f"  {coverages[label]:>14.0%}"
    if len(labels) == 2:
        delta = coverages[labels[1]] - coverages[labels[0]]
        row += f"  {delta:>+9.0%}p"
    print(row)

    # Coverage range
    row = f"{'Coverage (range)':<25s}"
    for label in labels:
        s = runs[label]["summary"]
        row += f"  {s['min_coverage']:.0%}-{s['max_coverage']:.0%}".rjust(15)
    print(row)

    # Per difficulty
    for diff in ("easy", "medium", "hard"):
        row = f"  {diff:<23s}"
        diffs = {}
        for label in labels:
            avg = runs[label]["summary"]["per_difficulty"][diff]["avg_coverage"]
            diffs[label] = avg
            row += f"  {avg:>14.0%}"
        if len(labels) == 2:
            delta = diffs[labels[1]] - diffs[labels[0]]
            row += f"  {delta:>+9.0%}p"
        print(row)

    # Trials
    row = f"{'Trials (success/total)':<25s}"
    for label in labels:
        s = runs[label]["summary"]
        row += f"  {s['successful_trials']}/{s['total_trials']}".rjust(15)
    print(row)

    # Tokens
    row = f"{'Avg tokens':<25s}"
    for label in labels:
        successful = [
            t for t in runs[label]["trials"] if t.get("coverage", 0) > 0
        ]
        if successful:
            avg_tok = sum(t.get("tokens_used", 0) for t in successful) / len(successful)
            row += f"  {avg_tok:>13,.0f}"
        else:
            row += f"  {'n/a':>15s}"
    print(row)

    # Scoring method
    row = f"{'Scoring method':<25s}"
    for label in labels:
        row += f"  {runs[label]['config']['scoring_method']:>15s}"
    print(row)

    print(f"\n{'='*70}")

    # Per-fact comparison (if both have per_fact data)
    if all(
        runs[l]["trials"][0].get("per_fact")
        for l in labels
        if runs[l]["summary"]["successful_trials"] > 0
    ):
        print("\nPer-fact recovery (across all trials):\n")
        print(f"  {'Fact':<8s} {'Diff':<8s} {'Domain':<20s}", end="")
        for label in labels:
            print(f"  {label:>12s}", end="")
        print()
        print("  " + "-" * 60)

        # Aggregate per-fact across trials
        for fact in sorted(
            runs[labels[0]]["trials"][0]["per_fact"].keys(),
            key=lambda x: (
                {"easy": 0, "medium": 1, "hard": 2}.get(
                    next(
                        (f["difficulty"] for f in _get_facts() if f["id"] == x), "?"
                    ),
                    3,
                ),
                x,
            ),
        ):
            fact_info = next((f for f in _get_facts() if f["id"] == fact), None)
            if not fact_info:
                continue
            print(
                f"  {fact:<8s} {fact_info['difficulty']:<8s} {fact_info['domain']:<20s}",
                end="",
            )
            for label in labels:
                successful = [
                    t
                    for t in runs[label]["trials"]
                    if t.get("per_fact", {}).get(fact)
                ]
                if successful:
                    found_count = sum(
                        1
                        for t in successful
                        if t["per_fact"][fact].get("found")
                    )
                    print(f"  {found_count}/{len(successful):>10s}", end="")
                else:
                    print(f"  {'n/a':>12s}", end="")
            print()

    print()


def _get_facts():
    """Import seed facts from the benchmark module."""
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_field_memory import SEED_FACTS
    return SEED_FACTS


if __name__ == "__main__":
    main()
