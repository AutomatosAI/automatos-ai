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
    header = f"{'Metric':<30s}"
    for label in labels:
        mode = runs[label].get("mode", "?")
        header += f"  {label:>15s}"
    if len(labels) == 2:
        header += f"  {'Delta':>10s}"
    print(header)
    print("-" * len(header))

    # Mode
    row = f"{'Mode':<30s}"
    for label in labels:
        row += f"  {runs[label].get('mode', '?'):>15s}"
    print(row)

    # Coverage
    coverages = {}
    for label in labels:
        coverages[label] = runs[label]["summary"]["avg_coverage"]

    row = f"{'Coverage (avg)':<30s}"
    for label in labels:
        row += f"  {coverages[label]:>14.0%}"
    if len(labels) == 2:
        delta = coverages[labels[1]] - coverages[labels[0]]
        row += f"  {delta:>+9.0%}p"
    print(row)

    # Coverage range
    row = f"{'Coverage (range)':<30s}"
    for label in labels:
        s = runs[label]["summary"]
        row += f"  {s['min_coverage']:.0%}-{s['max_coverage']:.0%}".rjust(15)
    print(row)

    # Per difficulty
    for diff in ("easy", "medium", "hard"):
        row = f"  {diff:<28s}"
        diffs = {}
        for label in labels:
            avg = runs[label]["summary"]["per_difficulty"][diff]["avg_coverage"]
            diffs[label] = avg
            row += f"  {avg:>14.0%}"
        if len(labels) == 2:
            delta = diffs[labels[1]] - diffs[labels[0]]
            row += f"  {delta:>+9.0%}p"
        print(row)

    # Per domain (if available)
    first_run = runs[labels[0]]
    if first_run.get("summary", {}).get("per_domain"):
        print()
        all_domains = set()
        for label in labels:
            all_domains.update(runs[label].get("summary", {}).get("per_domain", {}).keys())
        for domain in sorted(all_domains):
            row = f"  {domain:<28s}"
            diffs = {}
            for label in labels:
                avg = runs[label].get("summary", {}).get("per_domain", {}).get(domain, {}).get("avg_coverage", 0)
                diffs[label] = avg
                row += f"  {avg:>14.0%}"
            if len(labels) == 2:
                delta = diffs[labels[1]] - diffs[labels[0]]
                row += f"  {delta:>+9.0%}p"
            print(row)

    print()

    # Trials
    row = f"{'Trials (success/total)':<30s}"
    for label in labels:
        s = runs[label]["summary"]
        row += f"  {s['successful_trials']}/{s['total_trials']}".rjust(15)
    print(row)

    # Facts count
    row = f"{'Facts':<30s}"
    for label in labels:
        row += f"  {runs[label]['config']['facts_count']:>15d}"
    print(row)

    # Domains count
    row = f"{'Domains':<30s}"
    for label in labels:
        row += f"  {runs[label]['config'].get('domains', '?'):>15}"
    print(row)

    # Tokens
    row = f"{'Avg tokens':<30s}"
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

    # Telemetry
    row = f"{'Field queries (total)':<30s}"
    for label in labels:
        successful = [t for t in runs[label]["trials"] if t.get("coverage", 0) > 0]
        total_q = sum(t.get("telemetry", {}).get("field_queries", 0) for t in successful)
        row += f"  {total_q:>15d}"
    print(row)

    row = f"{'Field injects (total)':<30s}"
    for label in labels:
        successful = [t for t in runs[label]["trials"] if t.get("coverage", 0) > 0]
        total_i = sum(t.get("telemetry", {}).get("field_injects", 0) for t in successful)
        row += f"  {total_i:>15d}"
    print(row)

    # Scoring method
    row = f"{'Scoring method':<30s}"
    for label in labels:
        row += f"  {runs[label]['config']['scoring_method']:>15s}"
    print(row)

    print(f"\n{'='*70}")
    print()


if __name__ == "__main__":
    main()
