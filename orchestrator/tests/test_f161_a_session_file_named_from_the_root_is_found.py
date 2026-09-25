"""F161 (night 5) — a session's file named from the deliverables root is found.

Ticket #980 ran in ``sessions/980``, and its result named
``deliverables/sessions/980/christmas-box-cafe-offer-overview.md``, a file that
was there (10,458 bytes). The close check (F014) joined that name onto the
session's own folder, looked for ``sessions/980/deliverables/sessions/980/…``, and
wrote "Not found when this ticket closed … the workspace has no such file",
sending the ticket to review; later steps took the note as fact ("the original
drafts were not saved"). The same happened on #982, #985, #994 and #996.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from services import result_files

WS = "00000000-0000-0000-0000-0000000000c1"
HOST_ROOT = "/Users/me/Development/deliverables"
FILE_980 = "sessions/980/christmas-box-cafe-offer-overview.md"
RESULT_980 = ("The offer overview is saved as `deliverables/sessions/980/christmas-box-cafe-offer-overview.md` "
              "(10,458 bytes), ready for the tasting visits.")
SESSION_980 = {"host_id": "h1", "cwd": f"{HOST_ROOT}/sessions/980"}


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
    check = asyncio.run(result_files.check_named_files(NS(id=980, runtime_ref=SESSION_980), text, WS,
                                                       client=worker))
    return None if check is None else check.note


def test_980s_file_named_from_the_deliverables_root_is_found():
    assert _note(RESULT_980, _Worker({FILE_980})) is None


def test_a_name_from_the_root_without_its_folder_name_is_found_too():
    assert _note(f"Saved: `{FILE_980}`", _Worker({FILE_980})) is None


def test_the_same_name_is_still_missing_when_the_file_is_nowhere():
    note = _note(RESULT_980, _Worker())
    assert "`deliverables/sessions/980/christmas-box-cafe-offer-overview.md`" in note
    assert "Sent to review instead of done" in note


def test_a_name_relative_to_the_session_folder_is_looked_for_there_only():
    assert result_files.worker_paths(["out/summary.md"], workspace_id=WS, runtime_ref=SESSION_980,
                                     projects_dir=None) == [("out/summary.md", "sessions/980/out/summary.md")]
