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


def test_refused_calls_are_grouped_by_what_they_mean_holds_first():
    """PRD-245 S0.3: the report says which refusals put the ticket in review (holds)
    and which were the guardrail doing its job; an older summary without a kind
    is listed as unclassified."""
    from services.session_report import refused_calls_lines

    lines = refused_calls_lines([
        {"tool": "Read", "reason": "Read outside the session directory: /x/host.json", "kind": "read_outside"},
        {"tool": "Bash", "reason": "'pip --version' is outside this ticket's Bash allowlist — no answer from the operator within 120 s", "kind": "hold"},
        {"tool": "ToolSearch", "reason": "tool 'ToolSearch' is not enabled for session tickets", "kind": "unknown_tool"},
        {"tool": "Bash", "reason": "an older summary without a kind"},
    ])
    text = "\n".join(lines)
    assert lines[0] == "### Refused tool calls"
    assert text.index("**Held for the operator") < text.index("**Refused (unclassified") \
        < text.index("**Reads outside the session directory") < text.index("**Tools a session does not have")
    assert "- Bash: 'pip --version'" in text and "- Read: Read outside" in text and "- Bash: an older summary" in text
    assert "- ToolSearch: tool 'ToolSearch'" in text
