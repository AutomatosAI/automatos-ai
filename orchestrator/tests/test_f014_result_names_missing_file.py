"""F014 (night 1, #153) — a ticket may not close ``done`` pointing at nothing.

#153 closed ``done`` with "I wrote the onboarding pack and saved it here:
``deliverables/cafe_onboarding_pack.md``"; the file never existed anywhere.
The completion writer now looks up every file a result names that maps into
the workspace, and one that is not there sends the ticket to review with the
names on it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import api.board_tasks as bt
from services import result_files

RESULT_153 = """Done — I wrote the onboarding pack and saved it here:

- `deliverables/cafe_onboarding_pack.md`

It's short, friendly, and covers ordering terms and monthly invoicing."""
WS = "00000000-0000-0000-0000-0000000000c1"
HOST_ROOT = "/Users/me/Development/deliverables"


class _Worker:
    """The workspace worker's directory listing, from a set of existing paths."""

    def __init__(self, files=(), *, truncated=False, down=False):
        self.files = set(files)
        self.truncated = truncated
        self.down = down
        self.listed = []

    async def list_dir(self, path="."):
        self.listed.append(path)
        if self.down:
            raise ConnectionError("worker unreachable")
        names = {f.rpartition("/")[2] for f in self.files if (f.rpartition("/")[0] or ".") == path}
        return {"path": path, "entries": [{"name": n, "type": "file"} for n in sorted(names)],
                "truncated": self.truncated}


# ── what a result names ─────────────────────────────────────────────────────

def test_the_night_1_result_names_its_file():
    assert result_files.named_files(RESULT_153) == ["deliverables/cafe_onboarding_pack.md"]


def test_urls_commands_bare_names_and_home_paths_are_not_taken_for_files():
    text = ("See https://example.com/docs/a.md and run `python scripts/build.py --all`. "
            "I edited `notes.md` and ~/secret/x.md; the pack is at [pack](reports/pack.pdf) "
            f"and {HOST_ROOT}/sessions/283/verify-283.md.")
    assert result_files.named_files(text) == ["reports/pack.pdf", f"{HOST_ROOT}/sessions/283/verify-283.md"]


def test_where_each_name_is_looked_up(monkeypatch):
    monkeypatch.setattr("services.cli_host_service.configured_workspace_dir", lambda: HOST_ROOT)
    names = ["reports/pack.pdf", f"{HOST_ROOT}/sessions/283/verify-283.md", "/etc/x.md"]
    api_run = result_files.worker_paths(names, workspace_id=WS, runtime_ref=None, projects_dir=None)
    assert api_run == [("reports/pack.pdf", "reports/pack.pdf"),
                       (f"{HOST_ROOT}/sessions/283/verify-283.md", "sessions/283/verify-283.md")]
    session = {"host_id": "h1", "cwd": f"{HOST_ROOT}/sessions/283"}
    assert result_files.worker_paths(["out/summary.md"], workspace_id=WS, runtime_ref=session, projects_dir=None) == [
        ("out/summary.md", "sessions/283/out/summary.md")]
    in_a_repo = {"host_id": "h1", "cwd": "/Users/me/Development/shop"}
    assert result_files.worker_paths(["out/summary.md"], workspace_id=WS, runtime_ref=in_a_repo,
                                     projects_dir="/Users/me/Development") == [
        ("out/summary.md", "projects/shop/out/summary.md")]
    unknown_folder = {"host_id": "h1"}
    assert result_files.worker_paths(["out/summary.md"], workspace_id=WS, runtime_ref=unknown_folder,
                                     projects_dir=None) == []


# ── the note ────────────────────────────────────────────────────────────────

def _check(text, worker, runtime_ref=None, db=None):
    return asyncio.run(result_files.check_named_files(NS(id=153, runtime_ref=runtime_ref), text, WS,
                                                      db=db, client=worker))


def _note(text, worker, runtime_ref=None):
    check = _check(text, worker, runtime_ref)
    return None if check is None else check.note


class _KnowledgeBase:
    """documents(filename, original_filename) for the knowledge-base lookup."""

    def __init__(self, *names):
        self.names = names

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def all(self):
        return [(name, None) for name in self.names]


def test_a_named_file_that_is_there_passes_and_one_that_is_not_is_named():
    assert _note(RESULT_153, _Worker({"deliverables/cafe_onboarding_pack.md"})) is None
    note = _note(RESULT_153, _Worker())
    assert "`deliverables/cafe_onboarding_pack.md`" in note and "Sent to review instead of done" in note


def test_a_worker_that_cannot_answer_is_not_a_verdict():
    assert _note(RESULT_153, _Worker(down=True)) is None
    assert _note(RESULT_153, _Worker(truncated=True)) is None


def test_a_result_that_names_no_file_asks_the_worker_nothing():
    worker = _Worker()
    assert _note("All done — the answer is 42.", worker) is None and worker.listed == []


def test_a_name_saved_to_the_knowledge_base_is_found_there_not_sent_to_review():
    """Night 1's #153 did save its pack — as a knowledge-base document named
    ``deliverables/cafe_onboarding_pack.md`` (doc #473), not as a workspace file."""
    check = _check(RESULT_153, _Worker(), db=_KnowledgeBase("deliverables/cafe_onboarding_pack.md"))
    assert check.review is False
    assert check.note.startswith("Saved to the knowledge base, not as a file in the workspace: "
                                 "`deliverables/cafe_onboarding_pack.md`")
    by_basename = _check(RESULT_153, _Worker(), db=_KnowledgeBase("cafe_onboarding_pack.md"))
    assert by_basename.review is False


def test_one_listing_per_folder_and_the_note_counts_the_rest():
    text = " ".join(f"`reports/r{i}.md`" for i in range(5))
    worker = _Worker({"reports/r0.md"})
    note = _note(text, worker)
    assert worker.listed == ["reports"]
    assert "`reports/r1.md`, `reports/r2.md`, `reports/r3.md` and 1 more" in note and "them" in note


# ── the completion writer ───────────────────────────────────────────────────

class _Task:
    def __init__(self, runtime_ref=None):
        self.id, self.status, self.result, self.error_message = 153, "in_progress", None, None
        self.completed_at = self.lease_until = None
        self.runtime_ref = runtime_ref


class _Session:
    def __init__(self, task):
        self.task = task

    def query(self, *_a, **_k):
        return self

    def get(self, *_a, **_k):  # db.get(BoardTask, id, with_for_update=..., populate_existing=...)
        return self.task

    def commit(self):
        pass


@pytest.fixture
def writer(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)

    def run(task, text, worker):
        monkeypatch.setattr("core.workspace_client.WorkspaceClient", lambda ws: worker)
        return asyncio.run(bt.finalize_board_task_run(
            _Session(task), task_id=task.id, workspace_id=WS, agent_id=7,
            exec_result={"status": "success", "result": text}))
    return run


def test_night_1s_ticket_now_lands_in_review_with_the_missing_file_named(writer):
    task = _Task()
    assert writer(task, RESULT_153, _Worker()) == "review"
    assert task.result.startswith("Done — I wrote the onboarding pack")
    assert task.result.rstrip().endswith("Sent to review instead of done.")


def test_the_same_result_with_its_file_written_closes_done(writer):
    task = _Task()
    assert writer(task, RESULT_153, _Worker({"deliverables/cafe_onboarding_pack.md"})) == "done"
    assert task.result == RESULT_153


def test_a_result_saved_as_a_knowledge_base_document_closes_done_saying_where(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("core.workspace_client.WorkspaceClient", lambda ws: _Worker())

    class Session(_KnowledgeBase):
        def __init__(self, task):
            super().__init__("deliverables/cafe_onboarding_pack.md")
            self.task = task

        def get(self, *_a, **_k):
            return self.task

        def commit(self):
            pass

    task = _Task()
    status = asyncio.run(bt.finalize_board_task_run(
        Session(task), task_id=task.id, workspace_id=WS, agent_id=7,
        exec_result={"status": "success", "result": RESULT_153}))
    assert status == "done" and "Saved to the knowledge base" in task.result
