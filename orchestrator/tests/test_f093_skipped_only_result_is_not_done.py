"""F093 (night 3) — a ticket whose result is only skipped tool calls is not done.

#484 was filed done/ok on a whole result of "search_knowledge: Skipped: Tool
'search_knowledge' has reached its execution limit (5) for this turn" (29 model
calls, 369,414 tokens). Night 1's #639 report was the same shape eight times
over. Such a result now goes to review saying nothing was produced.
"""
from __future__ import annotations

import asyncio

import pytest

import api.board_tasks as bt
from services.result_substance import NOTHING_DONE_HEADER, TOOL_RESULTS_HEADER, nothing_done_note

WS = "00000000-0000-0000-0000-0000000000c1"
SKIP_484 = "**search_knowledge**: Skipped: Tool 'search_knowledge' has reached its execution limit (5) for this turn"
RESULT_484 = f"{TOOL_RESULTS_HEADER}\n\n{SKIP_484}"
RESULT_639 = "\n\n".join(
    ["**workspace_git**: Skipped: Tool 'workspace_git' has reached its execution limit (8) for this turn"] * 3)


def test_the_night_3_and_night_1_results_produced_nothing():
    note = nothing_done_note(RESULT_484)
    assert note.startswith("Nothing was produced")
    assert "search_knowledge: Tool 'search_knowledge' has reached its execution limit (5)" in note
    assert nothing_done_note(RESULT_639).count("workspace_git:") == 1          # one reason per distinct skip
    assert nothing_done_note(SKIP_484.replace("**", ""))                       # as the persona quoted it
    assert nothing_done_note(f"{NOTHING_DONE_HEADER}\n\n{SKIP_484}")          # the agent path's new header
    assert nothing_done_note("**slack_send**: Blocked by policy: ask the owner first")


def test_a_result_with_anything_real_in_it_is_left_alone():
    assert nothing_done_note("The brand voice guide says: warm, plain, local.") is None
    mixed = f"{TOOL_RESULTS_HEADER}\n\n**search_knowledge**: Found 3 passages about the brand voice.\n\n{SKIP_484}"
    assert nothing_done_note(mixed) is None
    assert nothing_done_note("I skipped the second search: the first one answered it.") is None
    assert nothing_done_note("") is None


# ── the completion writer ───────────────────────────────────────────────────

class _Task:
    def __init__(self):
        self.id, self.status, self.result, self.error_message = 484, "in_progress", None, None
        self.completed_at = self.lease_until = None
        self.runtime_ref = None


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
def finalize(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("services.result_files.check_named_files", _noop)

    def run(task, text):
        return asyncio.run(bt.finalize_board_task_run(
            _Session(task), task_id=task.id, workspace_id=WS, agent_id=7,
            exec_result={"status": "success", "result": text}))
    return run


def test_484_now_lands_in_review_saying_nothing_was_produced(finalize):
    task = _Task()
    assert finalize(task, RESULT_484) == "review"
    assert task.result.startswith(RESULT_484)                   # the skipped tool line stays in view
    assert "(search_knowledge: Tool 'search_knowledge' has reached its execution limit (5)" in task.result
    assert task.result.endswith("Sent to review instead of done.")


def test_a_real_result_still_closes_done(finalize):
    task = _Task()
    assert finalize(task, "The brand voice guide says: warm, plain, local.") == "done"
