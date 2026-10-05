"""#839 — a session's result that names a file relative to a subfolder is
not sent to review.

A session that ran across a monorepo closed with "I wrote `api/src/index.ts`
and `commercial/README.md`"; both files existed, at
``packages/api/src/index.ts`` and ``docs/commercial/README.md``. The close
check (F014) looked for each name joined onto the session's own folder only,
found neither, and sent real work to review.

The fix: a name still not found by the exact lookup is checked once more
against the session's folder, walked once, for a path whose trailing
components match the name exactly (a path-component boundary, not a string
suffix — ``src/index.ts`` must not match ``xsrc/index.ts``). A name that
matches two or more real paths is too ambiguous to pick, and stays missing.

The walk has its own time budget, separate from the exact pass (a review
note said: a slow walk that burns the whole check's budget must not let a
genuinely missing name through), and never descends into a dependency or
build folder (``node_modules`` and the like), which could otherwise exhaust
the whole walk before it reaches a real subfolder — or turn a unique match
into an ambiguous one.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from services import result_files

WS = "00000000-0000-0000-0000-0000000000c1"
# The workspace-id segment in the host path is the anchor `workspace_relative_path`
# matches on (services/cli_host_service.py) — it maps this cwd to `sessions/501`.
SESSION = {"host_id": "h1", "cwd": f"/Users/me/Development/{WS}/sessions/501"}
RESULT = "Done — I wrote `api/src/index.ts` and `commercial/README.md`, both ready for review."


class _Worker:
    """The workspace worker's directory listing, one level at a time (as the
    real worker does), from a set of real file paths. Intermediate folders are
    implied by the paths, not listed separately. ``hang_for`` makes a listing
    sleep past its caller's timeout instead of answering."""

    def __init__(self, files=(), *, down_for=(), hang_for=(), hang_seconds=1.0):
        self.files = set(files)
        self.down_for = set(down_for)
        self.hang_for = set(hang_for)
        self.hang_seconds = hang_seconds
        self.listed = []

    def _children(self, path):
        """(name, is_dir) for each direct child of ``path`` implied by the files."""
        prefix = "" if path == "." else path + "/"
        direct = set()
        for f in self.files:
            if not f.startswith(prefix):
                continue
            rest = f[len(prefix):]
            name, _, deeper = rest.partition("/")
            direct.add((name, bool(deeper)))
        return direct

    async def list_dir(self, path="."):
        self.listed.append(path)
        if path in self.hang_for:
            await asyncio.sleep(self.hang_seconds)
        if path in self.down_for:
            raise ConnectionError("worker unreachable")
        entries = [{"name": name, "type": "dir" if is_dir else "file"}
                   for name, is_dir in sorted(self._children(path))]
        return {"path": path, "entries": entries, "truncated": False}


def _note(text, worker, session=SESSION):
    check = asyncio.run(result_files.check_named_files(NS(id=501, runtime_ref=session), text, WS, client=worker))
    return None if check is None else check.note


def test_a_name_relative_to_a_subfolder_is_found_by_its_unique_suffix():
    worker = _Worker({"sessions/501/packages/api/src/index.ts",
                      "sessions/501/docs/commercial/README.md"})
    assert _note(RESULT, worker) is None


def test_a_component_boundary_is_kept_not_a_bare_string_suffix():
    """``src/index.ts`` must not match a real ``xsrc/index.ts``: the name's
    path components must line up exactly, not just the trailing characters."""
    worker = _Worker({"sessions/501/packages/xsrc/index.ts",
                      "sessions/501/docs/commercial/README.md"})
    note = _note("Done — I wrote `api/src/index.ts`.", worker)
    assert note is not None
    assert "`api/src/index.ts`" in note and "Sent to review instead of done" in note


def test_an_ambiguous_suffix_match_stays_missing():
    """Two real files both ending in ``api/src/index.ts`` — too ambiguous to
    pick one, so the name stays missing and the ticket still goes to review."""
    worker = _Worker({"sessions/501/packages/api/src/index.ts",
                      "sessions/501/services/api/src/index.ts"})
    note = _note("Done — I wrote `api/src/index.ts`.", worker)
    assert note is not None
    assert "`api/src/index.ts`" in note and "Sent to review instead of done" in note


def test_a_name_that_is_nowhere_at_all_stays_missing():
    worker = _Worker({"sessions/501/other/file.md"})
    note = _note("Done — I wrote `api/src/index.ts`.", worker)
    assert note is not None and "`api/src/index.ts`" in note


def test_the_suffix_fallback_only_runs_for_a_session_not_an_api_run():
    """An API run has no session folder to walk, so a name it gets wrong is
    just missing, the same as before #839."""
    worker = _Worker({"packages/api/src/index.ts"})
    note = _note("Done — I wrote `api/src/index.ts`.", worker, session=None)
    assert note is not None and "`api/src/index.ts`" in note


def test_the_walk_is_bounded_so_a_worker_that_cannot_answer_a_folder_is_not_a_verdict():
    worker = _Worker({"sessions/501/packages/api/src/index.ts"}, down_for={"sessions/501/packages"})
    note = _note("Done — I wrote `api/src/index.ts`.", worker)
    assert note is not None and "`api/src/index.ts`" in note


def test_a_timed_out_walk_keeps_what_it_found_and_a_genuinely_missing_name_stays_missing(monkeypatch):
    """The walk has its own clock (``WALK_TIMEOUT_SECONDS``), separate from the
    exact pass. ``docs/found.md`` resolves from a level the walk finishes
    before the cutoff (``extra`` → ``extra/docs``); ``extra/docs/deeper``
    never answers, so the walk is cut off there — without a regression where
    the whole check gives up and a genuinely missing name is let through."""
    monkeypatch.setattr(result_files, "WALK_TIMEOUT_SECONDS", 0.1)
    worker = _Worker({"sessions/501/extra/docs/found.md",
                      "sessions/501/extra/docs/deeper/buried.md"},
                     hang_for={"sessions/501/extra/docs/deeper"}, hang_seconds=2.0)
    note = _note("Done — I wrote `docs/found.md` and `nowhere/missing.md`.", worker)
    assert note is not None
    assert "`docs/found.md`" not in note
    assert "`nowhere/missing.md`" in note and "Sent to review instead of done" in note


def test_a_node_modules_copy_does_not_make_a_unique_suffix_ambiguous_and_is_not_walked():
    """``node_modules`` sits right next to the real subfolder at the top of the
    session's folder. A copy of the same file in there would, if walked, turn
    a unique suffix match into an ambiguous one — the walk must skip it
    outright, not just prefer the other match."""
    worker = _Worker({"sessions/501/packages/api/src/index.ts",
                      "sessions/501/node_modules/some-pkg/api/src/index.ts"})
    assert _note("Done — I wrote `api/src/index.ts`.", worker) is None
    assert not any("node_modules" in path for path in worker.listed)
