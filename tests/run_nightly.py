#!/usr/bin/env python3
"""Nightly API Test Runner

Runs pytest with JSON reporting, files Jira tickets for failures,
and emails a summary — all via Composio REST API.

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

# Load tests/.env so Composio vars are available for Jira + email
load_dotenv(TESTS_DIR / ".env", override=True)

# ---------------------------------------------------------------------------
# Composio config (optional — gracefully skipped if missing)
# ---------------------------------------------------------------------------
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY", "")
COMPOSIO_ENTITY_ID = os.environ.get("COMPOSIO_ENTITY_ID", "")
COMPOSIO_BASE = os.environ.get("COMPOSIO_BASE_URL", "https://backend.composio.dev/api/v2")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "")
REPORT_EMAIL_TO = os.environ.get("REPORT_EMAIL_TO", "")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "")


def run_pytest() -> int:
    """Run pytest and return the exit code."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        str(API_TESTS),
        f"--json-report",
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


def extract_failures(report: dict) -> list[dict]:
    """Pull failure details from the pytest-json-report output."""
    failures = []
    for test in report.get("tests", []):
        if test.get("outcome") == "failed":
            longrepr = ""
            call = test.get("call", {})
            if isinstance(call, dict):
                longrepr = call.get("longrepr", "")
            failures.append({
                "nodeid": test.get("nodeid", "unknown"),
                "outcome": test.get("outcome"),
                "duration": round(test.get("duration", 0), 3),
                "longrepr": longrepr,
            })
    return failures


# ---------------------------------------------------------------------------
# Composio helpers
# ---------------------------------------------------------------------------
def _composio_headers() -> dict:
    return {
        "X-API-Key": COMPOSIO_API_KEY,
        "Content-Type": "application/json",
    }


def _composio_post(action: str, params: dict, app_name: str = "") -> dict | None:
    """Execute a Composio action via REST. Returns response JSON or None."""
    import httpx

    url = f"{COMPOSIO_BASE}/actions/{action}/execute"
    body = {"entityId": COMPOSIO_ENTITY_ID, "input": params}
    if app_name:
        body["appName"] = app_name
    try:
        r = httpx.post(url, headers=_composio_headers(), json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"[runner] WARNING: Composio {action} failed: {exc}")
        return None


def file_jira_ticket(failure: dict) -> str | None:
    """Create a Jira Bug for a single test failure. Returns issue key or None."""
    if not COMPOSIO_API_KEY:
        return None

    summary = f"[Nightly] {failure['nodeid'].split('::')[-1]} FAILED"
    description = (
        f"*Test:* `{failure['nodeid']}`\n"
        f"*Duration:* {failure['duration']}s\n\n"
        f"{{code}}\n{failure['longrepr'][:3000]}\n{{code}}"
    )

    result = _composio_post("JIRA_CREATE_ISSUE", {
        "project_key": JIRA_PROJECT_KEY,
        "summary": summary[:255],
        "description": description,
        "issue_type": "Bug",
        "labels": ["nightly-test"],
    }, app_name="JIRA")
    if result and result.get("data"):
        key = result["data"].get("key") or result["data"].get("id")
        print(f"[runner] Jira ticket created: {key}")
        return key
    return None


def send_email_summary(report: dict, failures: list[dict], jira_keys: list[str]):
    """Send a summary email via Composio GMAIL_SEND_EMAIL."""
    if not COMPOSIO_API_KEY or not REPORT_EMAIL_TO:
        print("[runner] Skipping email (COMPOSIO_API_KEY or REPORT_EMAIL_TO not set)")
        return

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    duration = round(report.get("duration", 0), 1)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    icon = "✅" if failed == 0 else "❌"

    subject = f"{icon} Nightly API Test Report — {now} — {passed}/{total} passed"

    lines = [
        f"Nightly API Test Report — {now}",
        f"{'=' * 50}",
        f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Duration: {duration}s",
        "",
    ]

    if failures:
        lines.append("FAILURES:")
        lines.append("-" * 40)
        for f in failures:
            lines.append(f"  • {f['nodeid']}  ({f['duration']}s)")
            if f["longrepr"]:
                # First 3 lines of traceback
                for tb_line in f["longrepr"].split("\n")[:3]:
                    lines.append(f"    {tb_line}")
            lines.append("")
        if jira_keys:
            lines.append(f"Jira tickets filed: {', '.join(k for k in jira_keys if k)}")
    else:
        lines.append("All tests passed! No issues found.")

    body = "\n".join(lines)

    _composio_post("GMAIL_SEND_EMAIL", {
        "recipient_email": REPORT_EMAIL_TO,
        "subject": subject,
        "body": body,
    }, app_name="GMAIL")
    print(f"[runner] Email sent to {REPORT_EMAIL_TO}")


def send_slack_summary(report: dict, failures: list[dict], jira_keys: list[str]):
    """Post a summary to Slack via Composio SLACK_SEND_MESSAGE."""
    if not COMPOSIO_API_KEY or not SLACK_CHANNEL:
        print("[runner] Skipping Slack (COMPOSIO_API_KEY or SLACK_CHANNEL not set)")
        return

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    duration = round(report.get("duration", 0), 1)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = "PASS" if failed == 0 else "FAIL"
    icon = ":white_check_mark:" if failed == 0 else ":x:"

    lines = [
        f"{icon} *Nightly Test Report — {now}*",
        f"Tests: *{passed}/{total}* passed ({round(passed / total * 100) if total else 0}%) | Duration: {duration}s",
        f"Verdict: *{verdict}*",
    ]

    if failures:
        lines.append("")
        lines.append("*Failures:*")
        for f in failures:
            name = f["nodeid"].split("::")[-1]
            lines.append(f"  • `{name}` ({f['duration']}s)")
        if jira_keys:
            lines.append(f"\nJira tickets: {', '.join(k for k in jira_keys if k)}")

    _composio_post("SLACK_SEND_MESSAGE", {
        "channel": SLACK_CHANNEL,
        "text": "\n".join(lines),
    }, app_name="SLACK")
    print(f"[runner] Slack message sent to {SLACK_CHANNEL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[runner] Nightly API test run — {datetime.now(timezone.utc).isoformat()}")

    # 1. Run pytest
    exit_code = run_pytest()

    # 2. Load report
    report = load_report()
    if not report:
        print("[runner] No report to process, exiting")
        sys.exit(exit_code)

    # 3. Extract failures
    failures = extract_failures(report)
    summary = report.get("summary", {})
    print(
        f"[runner] Results: {summary.get('total', 0)} total, "
        f"{summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed"
    )

    # 4. File Jira tickets for failures
    jira_keys = []
    for failure in failures:
        key = file_jira_ticket(failure)
        if key:
            jira_keys.append(key)

    # 5. Send email summary
    send_email_summary(report, failures, jira_keys)

    # 6. Send Slack summary
    send_slack_summary(report, failures, jira_keys)

    print(f"[runner] Done — exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
