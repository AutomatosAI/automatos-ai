"""PRD-234 S2: a task report for a Claude Code session says what ran and what it produced."""
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

from services.session_report import session_report_lines  # noqa: E402


def test_api_runs_get_no_session_section():
    assert session_report_lines({"runtime": "api", "result": "x"}) == []
    assert session_report_lines({}) == []


def test_session_section_names_model_files_deliverables_refusals_and_takeover():
    lines = session_report_lines({
        "runtime": "cli",
        "usage": {"model": "claude-fable-5", "input_tokens": 38, "output_tokens": 9518},
        "files_touched": ["/w/ws/sessions/68/hello.py"],
        "deliverables": [{"title": "hello.py", "file_path": "sessions/68/hello.py"}],
        "session": {
            "session_id": "bc258043", "cwd": "/w/ws/sessions/68", "exit_reason": "completed",
            "transcript_path": "/home/me/.claude/projects/x/bc258043.jsonl",
            "recent_tools": [{"at": "2026-09-03T10:28:00Z", "tool": "Bash", "subject": "python3 hello.py"}],
            "permission_denials": [{"tool": "Bash", "reason": "'cd /tmp' is outside this ticket's Bash allowlist"}],
        },
    })
    text = "\n".join(lines)
    assert text.startswith("## Claude Code session")
    assert "claude-fable-5" in text and "38 / 9518" in text and "no cost" in text
    assert "### Deliverables" in text and "`sessions/68/hello.py`" in text
    assert "### Refused tool calls" in text and "'cd /tmp'" in text
    assert "### Tool calls (last 1)" in text and "`python3 hello.py`" in text
    assert "cd /w/ws/sessions/68 && claude --resume bc258043" in text
