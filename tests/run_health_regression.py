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
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from run_nightly import _resolve_results_dir, build_summary, load_report


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

TARGETS = [
    # Core API tests
    TESTS_DIR / "api" / "test_health.py",
    TESTS_DIR / "api" / "test_chat.py",
    TESTS_DIR / "api" / "test_agents.py",
    TESTS_DIR / "api" / "test_memory.py",
    TESTS_DIR / "api" / "test_workflows.py",
    TESTS_DIR / "api" / "test_heartbeat.py",
    TESTS_DIR / "api" / "test_user_journeys.py",
    # Deepened journey tests
    TESTS_DIR / "api" / "test_chat_errors.py",
    TESTS_DIR / "api" / "test_memory_journeys.py",
    TESTS_DIR / "api" / "test_workflow_journeys.py",
    TESTS_DIR / "api" / "test_heartbeat_journeys.py",
    # Error path tests (guard against 500s)
    TESTS_DIR / "api" / "test_agent_errors.py",
    TESTS_DIR / "api" / "test_document_errors.py",
    TESTS_DIR / "api" / "test_memory_errors.py",
    TESTS_DIR / "api" / "test_workspace_errors.py",
    # PRD-123 harness features
    TESTS_DIR / "api" / "test_missions.py",
    TESTS_DIR / "api" / "test_permissions.py",
    TESTS_DIR / "api" / "test_health_bootstrap.py",
    # Mission lifecycle
    TESTS_DIR / "api" / "test_mission_journeys.py",
    # Performance baselines
    TESTS_DIR / "api" / "test_performance_baselines.py",
    # User journey tests (cross-domain)
    TESTS_DIR / "api" / "test_onboarding_journey.py",
    TESTS_DIR / "api" / "test_daily_workflow_journey.py",
    # Regression pins (highest signal)
    TESTS_DIR / "regressions" / "test_memory_regressions.py",
    TESTS_DIR / "regressions" / "test_workspace_isolation.py",
    TESTS_DIR / "regressions" / "test_agent_factory_regressions.py",
    TESTS_DIR / "regressions" / "test_document_sync_regressions.py",
    # Orchestrator-level regressions
    REPO_ROOT / "orchestrator" / "tests" / "test_memory_fixes.py",
]

REPORT_DIR = _resolve_results_dir()
REPORT_FILE = REPORT_DIR / "health-regression-report.json"
SUMMARY_FILE = REPORT_DIR / "health-regression-summary.json"
QA_REPORT_FILE = REPORT_DIR / "qa-report.json"

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


def classify_severity(nodeid: str, longrepr: str) -> str:
    text = f"{nodeid}\n{longrepr}".lower()
    if any(term in text for term in ["auth", "token", "login", "permission", "security", "jwt"]):
        return "P0"
    if any(term in text for term in ["chat", "memory", "workflow", "agent", "heartbeat", "webhook", "500"]):
        return "P1"
    if any(term in text for term in ["timeout", "pagination", "analytics", "stats", "rate limit"]):
        return "P2"
    return "P3"


def classify_category(nodeid: str, longrepr: str) -> str:
    text = f"{nodeid}\n{longrepr}".lower()
    category_map = [
        ("auth", ["auth", "jwt", "token", "permission"]),
        ("chat", ["chat", "conversation", "sse"]),
        ("memory", ["memory", "mem0", "context"]),
        ("workflow", ["workflow", "recipe", "execution"]),
        ("agents", ["agent", "model-config"]),
        ("missions", ["mission", "checkpoint", "stop_reason"]),
        ("heartbeat", ["heartbeat"]),
        ("routing", ["routing", "route"]),
        ("channels", ["channel", "slack", "discord", "telegram"]),
        ("documents", ["document", "upload", "pdf"]),
        ("knowledge", ["knowledge", "rag", "search"]),
        ("performance", ["latency", "slo", "performance", "baseline"]),
        ("permissions", ["permission", "tier", "assign", "revoke"]),
    ]
    for category, keywords in category_map:
        if any(keyword in text for keyword in keywords):
            return category
    return "api"


def build_ticket_title(nodeid: str, assertion_message: str, category: str) -> str:
    node = nodeid.split("::")[-1].replace("test_", "").replace("_", " ").strip()
    message = (assertion_message or "").strip().splitlines()[0] if assertion_message else ""
    message = re.sub(r"\s+", " ", message)
    base = message[:90] if message else node[:90]
    return f"[{category}] {base or node or 'Regression failure'}"


def build_qa_report(report: dict) -> dict:
    compact = build_summary(report)
    bugs = []
    full_platform_logs = None

    for test in report.get("tests", []):
        if test.get("outcome") != "failed":
            continue

        nodeid = test.get("nodeid", "unknown")
        call = test.get("call", {})
        longrepr = call.get("longrepr", "") if isinstance(call, dict) else ""
        assertion = compact_failure_assertion(longrepr)
        severity = classify_severity(nodeid, longrepr)
        category = classify_category(nodeid, longrepr)

        bugs.append(
            {
                "test": nodeid,
                "severity": severity,
                "title": build_ticket_title(nodeid, assertion, category),
                "error": assertion[:200],
                "traceback": longrepr[:3000],
                "server_log": None,
                "source_files": compact_failure_sources(longrepr),
                "category": category,
            }
        )

    return {
        "run_date": compact["run_date"],
        "total": compact["total_tests"],
        "passed": compact["passed"],
        "failed": compact["failed"],
        "skipped": compact["skipped"],
        "pass_rate": f"{compact['pass_rate']}%",
        "status": compact["status"],
        "platform_logs": full_platform_logs,
        "log_fetch_required": compact["failed"] > 0,
        "bugs": bugs,
    }


def compact_failure_sources(longrepr: str) -> list[str]:
    from run_nightly import _extract_source_files

    return _extract_source_files(longrepr)


def compact_failure_assertion(longrepr: str) -> str:
    from run_nightly import _extract_assertion_message

    return _extract_assertion_message(longrepr)


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

        qa_report = build_qa_report(report)
        with open(QA_REPORT_FILE, "w") as f:
            json.dump(qa_report, f, indent=2)
        print(f"[health-regression] QA report written to {QA_REPORT_FILE}")
    else:
        print("[health-regression] No report to process")

    print(f"[health-regression] Done — exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
