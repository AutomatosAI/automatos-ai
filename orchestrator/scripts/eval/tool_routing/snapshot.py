"""
Promote a scratch run in `results/` into a committed benchmark snapshot
under `benchmarks/YYYY-MM-DD-<label>/`.

Captures the eval set, model matrix, and all result/report files so the
benchmark stays reproducible even after queries or prices change.

Usage:

    cd orchestrator
    python -m scripts.eval.tool_routing.snapshot \
        --label "full-matrix" \
        --notes "First full sweep across all 9 models."

If `--label` is omitted, just the date is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
BENCHMARKS_DIR = HERE / "benchmarks"


_FILES_TO_SNAPSHOT: List[str] = [
    "eval_set.jsonl",
    "models.yaml",
]


def _slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="",
        help="Short slug appended to the date (e.g. 'smoke', 'full-matrix').",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes for the benchmark notes.md.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing snapshot dir at the same path.",
    )
    args = parser.parse_args()

    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.iterdir()):
        print("results/ is empty — nothing to snapshot", file=sys.stderr)
        return 1

    today = dt.date.today().isoformat()
    label_part = f"-{_slugify(args.label)}" if args.label else ""
    target = BENCHMARKS_DIR / f"{today}{label_part}"

    if target.exists():
        if not args.force:
            print(
                f"{target} already exists. Pass --force to overwrite or pick a different label.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=False)

    # Copy results contents
    for child in RESULTS_DIR.iterdir():
        if child.is_file():
            shutil.copy2(child, target / child.name)

    # Snapshot the eval set + model matrix at run time.
    for fname in _FILES_TO_SNAPSHOT:
        src = HERE / fname
        if src.exists():
            shutil.copy2(src, target / fname)

    notes_path = target / "notes.md"
    notes_path.write_text(
        f"# Benchmark — {today}{label_part}\n\n"
        f"{args.notes or '(no notes provided)'}\n\n"
        f"Snapshotted {len(list(target.iterdir())) - 1} files from results/.\n"
    )

    print(f"[snapshot] wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
