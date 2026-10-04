"""F297 (night 8): a file named with the workspace's own folder in front is found.

#0234's Analyst saved ``decaf_colombia_margin.csv`` (Deliverables had it at
00:27:20, before the card closed) and named it on the card as
``workspace/decaf_colombia_margin.csv``. The close check (F014) looked only for
``workspace/decaf_colombia_margin.csv``, wrote "the workspace has no such file",
and the owner's reject said the file was not there.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from services import result_files

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
RESULT_0234 = "| Margin | £6.47 |\n\nThe file is saved as `workspace/decaf_colombia_margin.csv`."


class _Worker:
    """The workspace worker's directory listing, from a set of existing paths."""

    def __init__(self, files=()):
        self.files = set(files)

    async def list_dir(self, path="."):
        names = {f.rpartition("/")[2] for f in self.files if (f.rpartition("/")[0] or ".") == path}
        return {"path": path, "entries": [{"name": n, "type": "file"} for n in sorted(names)]}


def _note(text, worker):
    check = asyncio.run(result_files.check_named_files(NS(id=1492, runtime_ref=None), text, WS, client=worker))
    return None if check is None else check.note


def test_0234s_file_is_found_at_the_workspace_root():
    assert _note(RESULT_0234, _Worker({"decaf_colombia_margin.csv"})) is None


def test_a_real_folder_called_workspace_is_still_looked_in_first():
    assert _note(RESULT_0234, _Worker({"workspace/decaf_colombia_margin.csv"})) is None


def test_a_file_that_is_nowhere_is_still_missing():
    note = _note(RESULT_0234, _Worker({"other.csv"}))
    assert "`workspace/decaf_colombia_margin.csv`" in note and "Sent to review instead of done" in note


def test_only_the_leading_folder_is_taken_off():
    assert result_files.worker_paths(["reports/workspace/a.csv"], workspace_id=WS, runtime_ref=None,
                                     projects_dir=None) == [("reports/workspace/a.csv", "reports/workspace/a.csv")]
