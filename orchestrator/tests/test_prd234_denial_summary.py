"""PRD-234: a refused tool call is kept on the ticket as reason + subject, capped."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.cli_host_service import MAX_DENIALS_KEPT, _denial_summary  # noqa: E402


def test_summary_keeps_reason_and_the_command_only():
    out = _denial_summary({"tool": "Bash", "stage": "PreToolUse", "reason": "outside the allowlist",
                           "input": {"command": "python3 hello.py", "timeout": 5, "description": "x"}})
    assert out == {"tool": "Bash", "stage": "PreToolUse", "reason": "outside the allowlist", "subject": "python3 hello.py"}


def test_summary_uses_the_path_for_file_tools_and_tolerates_junk():
    out = _denial_summary({"tool": "Write", "reason": "outside the session directory", "input": {"file_path": "/etc/x"}})
    assert out["subject"] == "/etc/x" and out["stage"] == ""
    assert _denial_summary("weird")["reason"] == "weird"
    long = _denial_summary({"tool": "Bash", "reason": "r" * 1000, "input": {"command": "c" * 1000}})
    assert len(long["reason"]) == 300 and len(long["subject"]) == 300
    assert MAX_DENIALS_KEPT == 20
