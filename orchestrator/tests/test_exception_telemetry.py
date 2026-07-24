"""Tests for the structured error telemetry util and the bare-except gate."""
import logging
import subprocess
from pathlib import Path
from uuid import uuid4

from core.utils.exception_telemetry import record_error

_ORCHESTRATOR_ROOT = Path(__file__).resolve().parent.parent


def test_record_error_logs_structured(caplog):
    ws = uuid4()
    with caplog.at_level(logging.ERROR, logger="automatos.errors"):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            record_error(
                subsystem="memory",
                operation="add_memory",
                error=exc,
                workspace_id=ws,
                agent_id=42,
                action_name="platform_add_memory",
            )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == "automatos.errors"
    assert record.levelno == logging.ERROR

    se = record.structured_error
    assert se["subsystem"] == "memory"
    assert se["operation"] == "add_memory"
    assert se["error_type"] == "ValueError"
    assert se["error_message"] == "boom"
    assert se["workspace_id"] == str(ws)
    assert se["agent_id"] == 42
    assert se["action_name"] == "platform_add_memory"

    # Logged inside the except block, so the traceback is captured.
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError


def test_record_error_truncates_message(caplog):
    long_message = "x" * 1000
    with caplog.at_level(logging.ERROR, logger="automatos.errors"):
        try:
            raise RuntimeError(long_message)
        except RuntimeError as exc:
            record_error(subsystem="tools", operation="dispatch", error=exc)

    se = caplog.records[0].structured_error
    assert len(se["error_message"]) == 500
    assert se["error_message"] == "x" * 500


def test_record_error_handles_none_workspace(caplog):
    with caplog.at_level(logging.ERROR, logger="automatos.errors"):
        try:
            raise KeyError("missing")
        except KeyError as exc:
            record_error(
                subsystem="harness",
                operation="apply_prescription",
                error=exc,
                workspace_id=None,
            )

    se = caplog.records[0].structured_error
    assert se["workspace_id"] is None
    assert se["subsystem"] == "harness"
    assert se["error_type"] == "KeyError"


def test_no_bare_except_in_codebase():
    """Gate: no bare ``except:`` anywhere under orchestrator/ (PRD-141 Phase 0).

    The repo has no GitHub Actions workflow, so this pytest test — which runs
    the same scripts/ci/check-no-bare-except.sh gate — is the enforcement
    point. It fails the suite if anyone reintroduces a bare except.
    """
    script = _ORCHESTRATOR_ROOT.parent / "scripts" / "ci" / "check-no-bare-except.sh"
    assert script.exists(), f"gate script missing: {script}"

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "bare-except gate failed:\n" + result.stdout + result.stderr
    )
