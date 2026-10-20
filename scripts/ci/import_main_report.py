#!/usr/bin/env python3
"""What `import main` loads, and how long it takes (PRD-262 US-032, FR-6).

Every API process imports ``main`` before it serves, so a heavy package imported at
module level anywhere on that path is paid by every worker at every boot. test.yml's
backend job runs this once per run, from ``orchestrator/``, with the packages CI
installs:

    python ../scripts/ci/import_main_report.py --budget-seconds "$BUDGET" --summary "$GITHUB_STEP_SUMMARY"

It imports ``main`` in a fresh interpreter under ``python -X importtime`` and:

* fails when a module FR-6 names (``FR6_MODULES``), or a submodule of one, is in
  ``sys.modules`` afterwards, and names the import chain that first loaded it, read
  from importtime's tree;
* writes the ``--top`` imports with the largest cumulative time to the step summary,
  with the wall time of ``import main`` itself (measured inside the child, so
  interpreter start-up is not in it; importtime's own overhead is);
* fails when that wall time is over ``--budget-seconds``. An empty budget only
  reports the time: PRD-262 US-032 sets the budget from the first week's numbers.

Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

# PRD-262 FR-6: `import main` loads none of these.
FR6_MODULES = ("camelot", "cv2", "matplotlib", "pandas", "nltk", "google.generativeai", "googleapiclient")
TOP_IMPORTS = 25
IMPORT_TIMEOUT_SECONDS = 600
STDERR_TAIL_LINES = 15
US_PER_MS = 1000
# The child times `import main` alone and writes what it loaded to argv[1]. os._exit:
# a thread the import started must not hold the interpreter open.
CHILD = (
    "import json, os, sys, time\n"
    "start = time.perf_counter()\n"
    "import main\n"
    "seconds = time.perf_counter() - start\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as out:\n"
    "    json.dump({'seconds': seconds, 'modules': sorted(sys.modules)}, out)\n"
    "sys.stdout.flush()\n"
    "sys.stderr.flush()\n"
    "os._exit(0)\n"
)
# `import time: <self us> | <cumulative us> | <2 spaces per level><module>` (CPython's format).
_ROW = re.compile(r"^import time:\s*(\d+) \|\s*(\d+) \| ( *)(\S+)\s*$")


class Row(NamedTuple):
    """One importtime line: a module's own and cumulative microseconds, and its depth in the tree."""

    self_us: int
    cumulative_us: int
    depth: int
    name: str


class Imported(NamedTuple):
    """One run of the child: the wall time of `import main`, sys.modules after it, importtime's rows."""

    seconds: float
    modules: List[str]
    rows: List[Row]


def parse_importtime(text: str) -> List[Row]:
    """importtime's rows, in the order printed: a module comes after everything it imported."""
    rows = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match:
            self_us, cumulative_us, indent, name = match.groups()
            rows.append(Row(int(self_us), int(cumulative_us), len(indent) // 2, name))
    return rows


def import_chain(rows: Sequence[Row], name: str) -> List[str]:
    """The chain of imports that first loaded ``name``, outermost first; [] when it is not in the tree.

    importtime prints a module once its import finishes, so its importer is the first
    later row one level up.
    """
    at = next((i for i, row in enumerate(rows) if row.name == name), None)
    if at is None:
        return []
    chain, depth = [name], rows[at].depth
    for row in rows[at + 1:]:
        if depth == 0:
            break
        if row.depth < depth:
            chain.append(row.name)
            depth = row.depth
    return chain[::-1]


def loaded(modules: Sequence[str], forbidden: Sequence[str]) -> Dict[str, str]:
    """Each forbidden name that is loaded, with the first loaded module that matched it."""
    found = {}
    for name in forbidden:
        hit = next((module for module in sorted(modules) if module == name or module.startswith(name + ".")), None)
        if hit:
            found[name] = hit
    return found


def top(rows: Sequence[Row], count: int) -> List[Row]:
    """The ``count`` rows with the largest cumulative time, largest first."""
    return sorted(rows, key=lambda row: row.cumulative_us, reverse=True)[:count]


def run_import(python: str, cwd: Path, timeout: float) -> Imported:
    """``import main`` in a fresh interpreter under ``-X importtime``. Raises RuntimeError when it fails."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "imported.json"
        try:
            done = subprocess.run([python, "-X", "importtime", "-c", CHILD, str(out)], cwd=cwd,
                                  capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"`import main` did not finish in {timeout:.0f} s") from error
        if done.returncode != 0 or not out.is_file():
            tail = " | ".join(done.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
            raise RuntimeError(f"`import main` failed with exit {done.returncode}: {tail}")
        record = json.loads(out.read_text(encoding="utf-8"))
    return Imported(float(record["seconds"]), list(record["modules"]), parse_importtime(done.stderr))


def budget_line(seconds: float, budget: Optional[float]) -> str:
    if budget is None:
        return "no budget yet: it is set from the first week's numbers (PRD-262 US-032)"
    verdict = "within it" if seconds <= budget else f"**{seconds - budget:.1f} s over it**"
    return f"budget {budget:.1f} s, {verdict}"


def report_markdown(imported: Imported, found: Dict[str, str], budget: Optional[float], count: int) -> str:
    """The step-summary section."""
    lines = [
        "### `import main` (PRD-262 US-032)",
        "",
        f"Wall time **{imported.seconds:.2f} s** (under `-X importtime`), {len(imported.modules)} modules loaded; "
        f"{budget_line(imported.seconds, budget)}.",
        "",
    ]
    if found:
        lines.append("**FR-6 modules loaded by `import main`:**")
        lines += [f"- `{hit}`: {' → '.join(import_chain(imported.rows, hit)) or '(not in the importtime tree)'}"
                  for hit in found.values()]
    else:
        lines.append("FR-6 modules loaded by `import main`: none.")
    lines += ["", f"| # | Cumulative | Self | Module (top {count} by cumulative time) |", "|--:|--:|--:|---|"]
    lines += [f"| {position} | {row.cumulative_us / US_PER_MS:,.1f} ms | {row.self_us / US_PER_MS:,.1f} ms | "
              f"`{row.name}` |" for position, row in enumerate(top(imported.rows, count), 1)]
    return "\n".join(lines) + "\n"


def problems(imported: Imported, found: Dict[str, str], budget: Optional[float]) -> List[str]:
    """What fails the step: an FR-6 module loaded, or a wall time over the budget."""
    failed = [f"`import main` loads {hit} ({' → '.join(import_chain(imported.rows, hit)) or 'chain unknown'}); "
              f"FR-6 says it must not" for hit in found.values()]
    if budget is not None and imported.seconds > budget:
        failed.append(f"`import main` took {imported.seconds:.2f} s, over its {budget:.1f} s budget")
    return failed


def _budget(text: str) -> Optional[float]:
    return float(text) if text.strip() else None


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as summary:
            summary.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="What `import main` loads, and how long it takes (PRD-262 US-032).")
    parser.add_argument("--python", default=sys.executable, help="the interpreter with the app's packages")
    parser.add_argument("--cwd", type=Path, default=Path("."), help="the directory main.py is in")
    parser.add_argument("--forbid", action="append", help=f"a module that must not load (default: FR-6, {FR6_MODULES})")
    parser.add_argument("--top", type=int, default=TOP_IMPORTS)
    parser.add_argument("--budget-seconds", type=_budget, default=None, help="empty: report only")
    parser.add_argument("--timeout", type=float, default=IMPORT_TIMEOUT_SECONDS)
    parser.add_argument("--summary", help="Markdown is appended here (the step summary)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        imported = run_import(args.python, args.cwd, args.timeout)
    except RuntimeError as error:
        print(f"::error title=import main (PRD-262 US-032)::{error}")
        _append(args.summary, f"### `import main` (PRD-262 US-032)\n\n**Failed:** {error}\n")
        return 1
    found = loaded(imported.modules, args.forbid or FR6_MODULES)
    _append(args.summary, report_markdown(imported, found, args.budget_seconds, args.top))
    failed = problems(imported, found, args.budget_seconds)
    for problem in failed:
        print(f"::error title=import main (PRD-262 US-032)::{problem}")
    print(f"import main: {imported.seconds:.2f} s, {len(imported.modules)} modules, "
          f"{len(found)} FR-6 module(s) loaded; {budget_line(imported.seconds, args.budget_seconds)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
