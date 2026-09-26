"""F168 — the ticket card names the session's last tool.

The chat's ticket card and the ticket's ending line read ``recent_tools[-1]["name"]``.
The backend writes each entry as ``{"at", "tool", "subject"}`` (``_absorb_hook_event``),
so "last tool" was always empty. The tests that covered it built entries by hand
in a shape the host never sends. Both readers now take ``tool``, and still read
``name`` from older rows.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS

from api.board_tasks import ending_summary
from modules.tools.discovery.handlers_board_tasks import task_card
from services import cli_host_service as svc


def _ticket(ref):
    return NS(id=999, title="Print the report", status="in_progress", runtime_ref=ref,
              started_at=None, completed_at=None)


def _as_the_host_reports_it():
    ref = {"runtime": "cli"}
    for tool, subject in (("Read", "report.html"), ("Bash", "google-chrome --headless --print-to-pdf=report.pdf")):
        svc._absorb_hook_event(ref, NS(id=999, workspace_id=uuid.uuid4()),
                               {"event": "PreToolUse", "tool_name": tool, "subject": subject})
    return ref


def test_the_card_and_the_ending_line_name_the_last_tool_the_host_reported():
    ref = _as_the_host_reports_it()
    assert task_card(_ticket(ref))["last_tool"] == "Bash"
    assert "last tool: Bash" in ending_summary(_ticket(ref))


def test_an_older_entry_still_reads():
    ref = {"runtime": "cli", "recent_tools": [{"name": "Read"}, {"name": "Edit"}]}
    assert task_card(_ticket(ref))["last_tool"] == "Edit"
    assert "last tool: Edit" in ending_summary(_ticket(ref))
