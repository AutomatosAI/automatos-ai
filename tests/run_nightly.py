#!/usr/bin/env python3
"""Nightly API Test Runner

Runs pytest with JSON reporting. Produces two output files:
  - test-report.json  — Full pytest-json-report (for archival)
  - test-summary.json — Compact summary for recipe agents (~2KB)

Output location (in priority order):
  1. AUTOMATOS_RESULTS_DIR env var (absolute path)
  2. {workspace_root}/artifacts/results  (workspace-relative)
  3. tests/api/reports/                   (fallback, in-repo)

The summary file is what recipe agents should read. It contains:
  totals, duration, pass rate, and a failures array with nodeids +
  truncated error messages. Small enough for an LLM to parse reliably.

Usage:
    python3 tests/run_nightly.py

Exit codes match pytest: 0=pass, 1=failures, 2=runner error.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
API_TESTS = TESTS_DIR / "api"
EXTRA_REGRESSION_TESTS = [
    REPO_ROOT / "orchestrator" / "tests" / "test_memory_fixes.py",
]


def _resolve_results_dir() -> Path:
    """Resolve results directory — outside the repo when in a workspace."""
    # Explicit override
    env_dir = os.environ.get("AUTOMATOS_RESULTS_DIR")
    if env_dir:
        return Path(env_dir)

    # Workspace layout: /workspaces/{id}/repos/automatos-ai/tests/
    # Results go to:    /workspaces/{id}/artifacts/results/
    workspace_root = REPO_ROOT.parent.parent  # up from repos/automatos-ai
    artifacts_dir = workspace_root / "artifacts" / "results"
    if workspace_root.name != REPO_ROOT.name and (workspace_root / "artifacts").exists():
        return artifacts_dir

    # Fallback: in-repo (local dev)
    return API_TESTS / "reports"


REPORT_DIR = _resolve_results_dir()
REPORT_FILE = REPORT_DIR / "test-report.json"
SUMMARY_FILE = REPORT_DIR / "test-summary.json"

# Load tests/.env for API_URL, API_KEY, WORKSPACE_ID
load_dotenv(TESTS_DIR / ".env", override=True)


def run_pytest() -> int:
    """Run pytest and return the exit code."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_targets = [str(API_TESTS)]
    missing_regressions = [str(path) for path in EXTRA_REGRESSION_TESTS if not path.exists()]
    if missing_regressions:
        print("[runner] ERROR: required regression tests missing:")
        for path in missing_regressions:
            print(f"[runner]   - {path}")
        return 2
    test_targets.extend(str(path) for path in EXTRA_REGRESSION_TESTS)

    cmd = [
        sys.executable, "-m", "pytest",
        *test_targets,
        "--json-report",
        f"--json-report-file={REPORT_FILE}",
        "-v",
        "--tb=short",
    ]
    print(f"[runner] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def load_report() -> dict | None:
    """Load the JSON report file."""
    if not REPORT_FILE.exists():
        print("[runner] WARNING: report file not found")
        return None
    with open(REPORT_FILE) as f:
        return json.load(f)


def _extract_source_files(longrepr: str) -> list[str]:
    """Extract source file paths from a pytest traceback string."""
    import re
    # Match patterns like "orchestrator/api/channels.py:326" in tracebacks
    paths = re.findall(r'([\w/]+\.py):(\d+)', longrepr)
    # Deduplicate, keep order, skip test infrastructure files
    seen = set()
    result = []
    for fpath, lineno in paths:
        key = f"{fpath}:{lineno}"
        if key not in seen and "site-packages" not in fpath:
            seen.add(key)
            result.append(key)
    return result


def _extract_assertion_message(longrepr: str) -> str:
    """Extract the AssertionError message from a traceback — this often
    contains the fix hint embedded by test authors."""
    import re
    # Look for "AssertionError: <message>" or "assert ... , '<message>'"
    m = re.search(r'(?:AssertionError|assert\w*Error):\s*(.+?)(?:\n|$)', longrepr, re.DOTALL)
    if m:
        return m.group(1).strip()[:2000]
    # Fallback: last non-empty line (often the assertion)
    lines = [ln.strip() for ln in longrepr.strip().splitlines() if ln.strip()]
    return lines[-1][:2000] if lines else ""


def build_summary(report: dict) -> dict:
    """Build a structured summary for recipe agents.

    Each failure includes:
    - nodeid: test path for the Bug Fixer to locate the test
    - error: full traceback (up to 3000 chars) so the agent can see
      the exact file/line that raised and the assertion message
    - assertion_message: extracted assertion text (often contains fix hints)
    - source_files: list of source file:line references from the traceback
    - severity: P0/P1/P2 if the test docstring contains it
    """
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    duration = round(report.get("duration", 0), 2)

    failures = []
    for test in report.get("tests", []):
        if test.get("outcome") == "failed":
            nodeid = test.get("nodeid", "unknown")
            dur = round(test.get("duration", 0), 3)

            # Full error — 3000 chars gives enough traceback for the
            # Bug Fixer to identify the exact source file and line.
            call = test.get("call", {})
            longrepr = ""
            if isinstance(call, dict) and call.get("longrepr"):
                longrepr = call["longrepr"]

            failure_entry = {
                "nodeid": nodeid,
                "duration_seconds": dur,
                "error": longrepr[:3000],
                "assertion_message": _extract_assertion_message(longrepr),
                "source_files": _extract_source_files(longrepr),
            }

            failures.append(failure_entry)

    return {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_seconds": duration,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        "status": "PASS" if failed == 0 else "FAIL",
        "failures": failures,
    }


def print_summary(compact: dict):
    """Print a human-readable summary to stdout."""
    print(
        f"[runner] Results: {compact['total_tests']} total, "
        f"{compact['passed']} passed, {compact['failed']} failed, "
        f"{compact['skipped']} skipped ({compact['duration_seconds']}s)"
    )

    if compact["failures"]:
        print("[runner] Failures:")
        for f in compact["failures"]:
            print(f"  FAIL  {f['nodeid']}  ({f['duration_seconds']}s)")
            if f["error"]:
                for line in f["error"].split("\n")[:3]:
                    print(f"        {line}")


def main():
    print(f"[runner] Nightly API test run — {datetime.now(timezone.utc).isoformat()}")
    print(f"[runner] Results dir: {REPORT_DIR}")

    exit_code = run_pytest()

    report = load_report()
    if report:
        compact = build_summary(report)
        print_summary(compact)

        # Write compact summary for recipe agents
        with open(SUMMARY_FILE, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"[runner] Summary written to {SUMMARY_FILE}")
    else:
        print("[runner] No report to process")

    print(f"[runner] Done — exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
