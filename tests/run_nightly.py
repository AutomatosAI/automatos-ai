#!/usr/bin/env python3
"""Nightly API Test Runner

Runs pytest with JSON reporting. Produces a structured report that
the recipe's LLM agents read to file Jira tickets, send Slack messages,
and email summaries using their own tools.

Usage:
    python tests/run_nightly.py          # from repo root
    python run_nightly.py                # from tests/ dir

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
REPORT_DIR = API_TESTS / "reports"
REPORT_FILE = REPORT_DIR / "test-report.json"

# Load tests/.env for API_URL, API_KEY, WORKSPACE_ID
load_dotenv(TESTS_DIR / ".env", override=True)


def run_pytest() -> int:
    """Run pytest and return the exit code."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        str(API_TESTS),
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


def print_summary(report: dict):
    """Print a human-readable summary to stdout."""
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    duration = round(report.get("duration", 0), 1)

    print(f"[runner] Results: {total} total, {passed} passed, {failed} failed, {skipped} skipped ({duration}s)")

    if failed > 0:
        print("[runner] Failures:")
        for test in report.get("tests", []):
            if test.get("outcome") == "failed":
                nodeid = test.get("nodeid", "unknown")
                dur = round(test.get("duration", 0), 3)
                print(f"  FAIL  {nodeid}  ({dur}s)")
                call = test.get("call", {})
                if isinstance(call, dict) and call.get("longrepr"):
                    for line in call["longrepr"].split("\n")[:3]:
                        print(f"        {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[runner] Nightly API test run — {datetime.now(timezone.utc).isoformat()}")

    # 1. Run pytest
    exit_code = run_pytest()

    # 2. Load and summarize report
    report = load_report()
    if report:
        print_summary(report)
    else:
        print("[runner] No report to process")

    print(f"[runner] Done — exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
