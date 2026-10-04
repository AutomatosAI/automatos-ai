"""F297 (night 8): a run that wrote no answer says so plainly, and its card goes to review.

The agent path ended a run with no answer on "Based on the tool results:" over its
last round's raw tool output, and that was the card's answer: #0254's whole answer
was a raw Composio error and went to Review like that; #0234's was
``{"written": true, "path": "decaf_colombia_margin.csv", …}``. The result now says
that no answer was written and what each call did, names an app that is not
connected and what to do, and the card goes to review saying nothing was produced.
"""
from __future__ import annotations

import asyncio

import pytest

import api.board_tasks as bt
from core.services.ticket_reasons import NOTHING_DONE_NOTE_PREFIX
from modules.agents.factory.answer_check import said_plainly
from services.result_substance import NO_ANSWER_HEADER, TOOL_RESULTS_HEADER, nothing_done_note, plain_no_answer

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
GMAIL = ('{"success": false, "error": "\'GMAIL\' is assigned to agent 323 but is not connected for this '
         'workspace. Connect it first, then retry.", "error_type": "composio_not_connected", "data": null}')
DUMP_0254 = f"{TOOL_RESULTS_HEADER}\n\n**composio_execute**: {GMAIL}\n\n**composio_execute**: {GMAIL}"
DUMP_0234 = (f"{TOOL_RESULTS_HEADER}\n\n**workspace_write_file**: "
             '{"written": true, "path": "decaf_colombia_margin.csv", "size_bytes": 183, "success": true}')


def test_0254_says_gmail_is_not_connected_and_shows_no_raw_error():
    plain = plain_no_answer(DUMP_0254)
    assert plain.splitlines()[0] == NO_ANSWER_HEADER
    assert plain.count("Gmail is not connected in this workspace") == 1          # said once, not per call
    assert "connect it in Composio, or ask for the work without it" in plain
    assert "error_type" not in plain and "{" not in plain and "agent 323" not in plain


def test_0234_says_the_file_was_written_and_no_answer_was():
    plain = plain_no_answer(DUMP_0234)
    assert plain == f"{NO_ANSWER_HEADER}\n- workspace_write_file: done (decaf_colombia_margin.csv)."


def test_an_answer_is_left_alone():
    assert plain_no_answer("| Margin | £6.47 |\nSaved as decaf_colombia_margin.csv.") is None


def test_every_lane_gets_the_plain_result_from_the_agent_run():
    async def run(*_a, **_k):
        return {"status": "success", "result": DUMP_0254, "execution": {"actions": []}}

    async def failed(*_a, **_k):
        return {"status": "error", "error": "Agent 7 could not be activated"}

    out = asyncio.run(said_plainly(run)())
    assert out["result"] == plain_no_answer(DUMP_0254) and out["execution"] == {"actions": []}
    assert asyncio.run(said_plainly(failed)()) == {"status": "error", "error": "Agent 7 could not be activated"}


class _Task:
    def __init__(self):
        self.id, self.status, self.result, self.error_message = 1532, "in_progress", None, None
        self.completed_at = self.lease_until = None
        self.runtime_ref = None


class _Session:
    def __init__(self, task):
        self.task = task

    def query(self, *_a, **_k):
        return self

    def get(self, *_a, **_k):
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
            _Session(task), task_id=task.id, workspace_id=WS, agent_id=323,
            exec_result={"status": "success", "result": text}))
    return run


def test_the_card_goes_to_review_saying_nothing_was_produced(finalize):
    task = _Task()
    assert finalize(task, plain_no_answer(DUMP_0254)) == "review"
    assert "Gmail is not connected in this workspace" in task.result
    assert task.result.endswith(nothing_done_note(plain_no_answer(DUMP_0254)))
    assert nothing_done_note(plain_no_answer(DUMP_0254)).startswith(NOTHING_DONE_NOTE_PREFIX)
