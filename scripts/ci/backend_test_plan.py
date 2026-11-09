"""Which orchestrator tests a change needs, and each CI shard's share (scripts/ci).

Gerard, 7-8 Oct ("it's getting so so slow to pass a PR"): ``orchestrator-tests``
ran all 14,181 tests in one process, for every PR, a README included. It took
24-28 minutes against a 30-minute timeout and timed out on a slow runner. Now:

* ``scope`` reads the PR's changed paths (one per line, stdin) and prints what the
  change needs, as JSON. ``full``: every test, when anything outside the
  selectable paths changed. ``select``: only the test files that name what
  changed, when every path is in ``frontend/``, ``docs/``, ``graphify-out/``,
  ``e2e/``, ``.github/`` (not test.yml) or a top-level ``*.md``. A test that
  reads such a file names its folder as a path in its source (``REPO_ROOT /
  "docs" / "PRDS"``), or a parent folder it reads below, so a search for the
  changed file's folder finds every test that can see it. A test that lists or
  greps the whole tracked tree can see any file, so it is always selected.
  ``none``: no test names it.
* ``plan --shard i/N`` prints shard i's test files, one per line, relative to
  ``orchestrator/``. Files are balanced by their recorded seconds (largest first,
  each to the lightest shard), so every file lands in exactly one shard and the N
  shards together run every selected file once.
* ``durations`` rebuilds ``backend_test_durations.json`` from a ``pytest -v`` CI
  log on stdin (``gh api repos/<repo>/actions/jobs/<id>/logs``). A file missing
  from the record weighs the median, so a stale record only skews the balance.

Push to main and a manual run always take ``full``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parents[2]
ORCH = REPO / "orchestrator"
DURATIONS = Path(__file__).resolve().with_name("backend_test_durations.json")

# Top-level folders whose files only some tests read: those that name the file's
# folder (``select``). Anything else changed means the full suite.
SELECTABLE_DIRS = {"frontend", "docs", "graphify-out", "e2e"}
FULL_SUITE_WORKFLOW = ".github/workflows/test.yml"
# How a test's source spells a path: a string, maybe after ./ or ../, whose folders
# are joined by / inside it or by Path's / (or os.path.join's comma) between strings.
OPEN = r"""["'](?:\.{1,2}/)*"""
JOIN = r"""(?:/|["']\)?\s*[/,]\s*["'])"""
ENDS = r"""/?["'](?!\)?\s*[/,]\s*["'])"""
# A test that lists or greps every tracked file (git ls-files, git grep) can see any file.
WHOLE_TREE = re.compile(r"""ls-files|git_grep|["']git["'],\s*["']grep["']""")
LOG_LINE = re.compile(r"^﻿?(\S+?)Z (tests/\S+?)::\S+ (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)")


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load this script by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse_shard = _sibling("template_shards").parse


def keyword_for(path: str) -> str | None:
    """What a test that reads ``path`` names in its source, or None: ``path`` needs every test.

    A file in a folder gives the folder, with a trailing ``/`` (``docs/PRDS/x.md`` ->
    ``docs/PRDS/``); a top-level ``X.md`` gives ``X``.
    """
    top, _, rest = path.partition("/")
    if not rest:
        return path[: -len(".md")] if path.endswith(".md") else None
    if top in SELECTABLE_DIRS or (top == ".github" and path != FULL_SUITE_WORKFLOW):
        return path.rpartition("/")[0] + "/"
    return None


def scope(paths: Iterable[str], test_files: Dict[str, str]) -> dict:
    """``{"mode": "full" | "select" | "none", "keywords": [...]}`` for the changed ``paths``."""
    keywords = set()
    for path in (p.strip() for p in paths):
        if not path:
            continue
        keyword = keyword_for(path)
        if keyword is None:
            return {"mode": "full", "keywords": []}
        keywords.add(keyword)
    if not keywords or not select(test_files, sorted(keywords)):
        return {"mode": "none", "keywords": sorted(keywords)}
    return {"mode": "select", "keywords": sorted(keywords)}


def select(test_files: Dict[str, str], keywords: List[str]) -> List[str]:
    """The test files whose source names any keyword or reads the whole tree; every file when there is none."""
    if not keywords:
        return sorted(test_files)
    patterns = [named_by(keyword) for keyword in keywords] + [WHOLE_TREE]
    return sorted(name for name, source in test_files.items() if any(p.search(source) for p in patterns))


def named_by(keyword: str) -> "re.Pattern[str]":
    """What a test's source says when it names ``keyword``.

    A folder (``docs/PRDS/``) is named by a path to it or into it, ``"docs/PRDS/x.md"``
    or ``ROOT / "docs" / "PRDS"``, or by a path that ends at one of its parents
    (``DOCS = ROOT / "docs"``: the test reads below it). Not by the bare word: ``docs``
    is in a hundred test files for ``/docs`` routes and docstrings (#1455). Any other
    keyword (``README``) is named by its text.
    """
    if not keyword.endswith("/"):
        return re.compile(re.escape(keyword))
    parts = [re.escape(part) for part in keyword.rstrip("/").split("/")]
    into = OPEN + JOIN.join(parts) + r"""(?=["'/])"""
    parents = [OPEN + JOIN.join(parts[:depth]) + ENDS for depth in range(1, len(parts))]
    return re.compile("|".join([into, *parents]))


def balance(files: List[str], seconds: Dict[str, float], count: int) -> List[List[str]]:
    """``files`` in ``count`` shards of near-equal recorded seconds, each shard sorted."""
    default = statistics.median(seconds.values()) if seconds else 1.0
    weight = {name: seconds.get(name, default) for name in files}
    shards: List[List[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for name in sorted(files, key=lambda n: (-weight[n], n)):
        lightest = min(range(count), key=lambda i: (loads[i], i))
        shards[lightest].append(name)
        loads[lightest] += weight[name]
    return [sorted(shard) for shard in shards]


def read_test_files(root: Path = ORCH) -> Dict[str, str]:
    """Every ``tests/**/test_*.py`` under ``root``, relative to it, with its source."""
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "tests").rglob("test_*.py"))
    }


def durations_from_log(lines: Iterable[str]) -> Dict[str, float]:
    """Seconds per test file from a ``pytest -v`` log: each result line's gap to the one before."""
    seconds: Counter = Counter()
    previous = None
    for line in lines:
        match = LOG_LINE.match(line)
        if not match:
            continue
        stamp = datetime.fromisoformat(match.group(1)[:26])
        if previous is not None:
            seconds[match.group(2)] += (stamp - previous).total_seconds()
        previous = stamp
    return {name: round(value, 2) for name, value in sorted(seconds.items())}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scope", help="changed paths on stdin -> JSON scope")
    plan = sub.add_parser("plan", help="this shard's test files")
    plan.add_argument("--shard", default="0/1", help="i/N, 0 <= i < N")
    plan.add_argument("--keywords", default="", help="comma-separated; empty = every test file")
    sub.add_parser("durations", help="pytest -v CI log on stdin -> durations JSON")
    args = parser.parse_args(argv)

    if args.command == "scope":
        print(json.dumps(scope(sys.stdin, read_test_files())))
    elif args.command == "plan":
        index, count = parse_shard(args.shard)
        keywords = [k for k in args.keywords.split(",") if k]
        seconds = json.loads(DURATIONS.read_text(encoding="utf-8"))
        print("\n".join(balance(select(read_test_files(), keywords), seconds, count)[index]))
    else:
        print(json.dumps(durations_from_log(sys.stdin), indent=0, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
