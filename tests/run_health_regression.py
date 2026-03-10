#!/usr/bin/env python3
"""API Health Check & Regression Detector

Runs a curated subset of high-signal API and regression tests and writes:
  - health-regression-report.json
  - health-regression-summary.json

This is intended to be the single entrypoint for the
"API Health Check & Regression Detector" recipe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from run_nightly import _resolve_results_dir, build_summary, load_report


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

TARGETS = [
    TESTS_DIR / "api" / "test_health.py",
    TESTS_DIR / "api" / "test_chat.py",
    TESTS_DIR / "api" / "test_agents.py",
    TESTS_DIR / "api" / "test_memory.py",
    TESTS_DIR / "api" / "test_workflows.py",
    TESTS_DIR / "api" / "test_heartbeat.py",
    TESTS_DIR / "api" / "test_user_journeys.py",
    REPO_ROOT / "orchestrator" / "tests" / "test_memory_fixes.py",
]

REPORT_DIR = _resolve_results_dir()
REPORT_FILE = REPORT_DIR / "health-regression-report.json"
SUMMARY_FILE = REPORT_DIR / "health-regression-summary.json"

load_dotenv(TESTS_DIR / ".env", override=True)


def run_pytest() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in TARGETS if not path.exists()]
    if missing:
        print("[health-regression] ERROR: required test targets missing:")
        for path in missing:
            print(f"[health-regression]   - {path}")
        return 2

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *(str(path) for path in TARGETS),
        "--json-report",
        f"--json-report-file={REPORT_FILE}",
        "-v",
        "--tb=short",
    ]
    print(f"[health-regression] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def main():
    print(f"[health-regression] Run started — {datetime.now(timezone.utc).isoformat()}")
    print(f"[health-regression] Results dir: {REPORT_DIR}")

    exit_code = run_pytest()
    report = load_report()

    if report:
        compact = build_summary(report)
        compact["suite"] = "API Health Check & Regression Detector"
        compact["targets"] = [str(path.relative_to(REPO_ROOT)) for path in TARGETS]
        with open(SUMMARY_FILE, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"[health-regression] Summary written to {SUMMARY_FILE}")
    else:
        print("[health-regression] No report to process")

    print(f"[health-regression] Done — exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
