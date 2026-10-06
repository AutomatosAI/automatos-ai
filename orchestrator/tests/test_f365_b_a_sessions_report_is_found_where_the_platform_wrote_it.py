"""F365 (b) (night 10c) — a session's report is found where the platform wrote it.

Brand Designer ticket #2140 ran in ``sessions/2140`` and filed its report with
``submit_report``. The platform wrote it at the workspace root, as every report is
(``reports/<agent>/<when>_<title>.md``, services/report_service.py), and the tool
answered with that path. The result named it, the close check (F014) joined the
name onto the session's folder only, looked for ``sessions/2140/reports/…``, and
wrote "Not found when this ticket closed … the workspace has no such file. Sent to
review instead of done." The file was in ``deliverables/reports/brand-designer/``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from services import result_files

WS = "00000000-0000-0000-0000-0000000000c1"
HOST_ROOT = "/Users/me/Development/deliverables"
REPORT_2140 = ("reports/brand-designer/2026-10-06_162300_5bc953_brand-kit-report-ticket-2140-highlights-moved-"
               "from-orange-to-navy.md")
RESULT_2140 = (f"Highlights moved from orange to navy; nothing else changed. The report is filed as "
               f"`{REPORT_2140}`, with the board in `brand-board-proposal-2a1f.png`.")
SESSION_2140 = {"host_id": "h1", "cwd": f"{HOST_ROOT}/sessions/2140"}


class _Worker:
    """The workspace worker's directory listing, from a set of existing paths."""

    def __init__(self, files=()):
        self.files = set(files)

    async def list_dir(self, path="."):
        names = {f.rpartition("/")[2] for f in self.files if (f.rpartition("/")[0] or ".") == path}
        return {"path": path, "entries": [{"name": n, "type": "file"} for n in sorted(names)]}


@pytest.fixture(autouse=True)
def deliverables_root(monkeypatch):
    monkeypatch.setattr("services.cli_host_service.configured_workspace_dir", lambda: HOST_ROOT)


def _note(text, worker):
    check = asyncio.run(result_files.check_named_files(NS(id=2140, runtime_ref=SESSION_2140), text, WS,
                                                       client=worker))
    return None if check is None else check.note


def test_2140s_report_at_the_workspace_root_is_found():
    assert _note(RESULT_2140, _Worker({REPORT_2140})) is None


def test_the_report_name_is_looked_for_in_the_session_folder_and_at_the_root():
    assert result_files.worker_paths([REPORT_2140], workspace_id=WS, runtime_ref=SESSION_2140,
                                     projects_dir=None) == [(REPORT_2140, f"sessions/2140/{REPORT_2140}"),
                                                            (REPORT_2140, REPORT_2140)]


def test_a_reports_folder_the_session_made_itself_is_still_found():
    assert _note(RESULT_2140, _Worker({f"sessions/2140/{REPORT_2140}"})) is None


def test_a_report_that_is_nowhere_still_sends_the_ticket_to_review():
    note = _note(RESULT_2140, _Worker({"reports/brand-designer/another-report.md"}))
    assert f"`{REPORT_2140}`" in note and "Sent to review instead of done" in note


def test_other_folders_are_still_looked_for_in_the_session_folder_only():
    assert result_files.worker_paths(["previews/board.png"], workspace_id=WS, runtime_ref=SESSION_2140,
                                     projects_dir=None) == [("previews/board.png", "sessions/2140/previews/board.png")]
