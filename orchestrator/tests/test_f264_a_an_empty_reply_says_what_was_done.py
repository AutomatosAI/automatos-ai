"""F264 (night 8): an empty reply tells the owner what was done, never "try again".

About 15 replies ended "I apologize, but I encountered an issue generating a
response. Please try again.", most after the work had happened (#0231 approved,
#0205 sent back, #0220 made, missions #0282 and #0344 made). The answer came back
empty, most often the retry after F108's nudge; the forced answer was empty too.
Now a blank answer in Auto's turn is the account of the turn's calls, from their
results, or says plainly that nothing changed.
"""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_BOARD_TASK, LANE_CHAT, usage_scope
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.execution.turn_account import (
    ACCOUNT_HEADER, NOTHING_CHANGED, WHY_ALL_FAILED, WHY_NO_CALL, WHY_ONLY_READS, account_of,
)

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object",
                                                                                      "properties": {}}}}]
APOLOGY = "I apologize, but I encountered an issue"


def _ok(**raw):
    return {"success": True, "llm_context": json.dumps(raw), "raw_result": raw}


def _refused(error):
    return {"success": False, "llm_context": f"Tool platform_execute failed: {error}",
            "raw_result": {"success": False, "error": error}}


# ── the account ─────────────────────────────────────────────────────────────

def test_the_account_names_each_card_and_where_it_is_now():
    said = account_of([
        ("platform_update_task_status", {"task_id": 231, "status": "done"},
         _ok(task_id=1452, number="#0231", status="done")),
        ("platform_update_task_status", {"task_id": 205, "status": "assigned"},
         _ok(task_id=1401, number="#0205", status="assigned")),
        ("platform_create_task", {"title": "Draft reply to Tom Reyes"},
         _ok(task_id=1420, number="#0220", status="assigned", title="Draft reply to Tom Reyes")),
        ("platform_create_mission", {"goal": "Prepare December club box content"},
         _ok(mission_id="3f0c2b9e-1d4a-4c55-9a1e-6f1f0c8b2a10", number="#0282", state="awaiting_approval",
             message="Mission 3f0c2b9e-1d4a-4c55-9a1e-6f1f0c8b2a10 created with 4 task(s)")),
        ("platform_get_task", {"task_id": 231}, _ok(task={"id": 1452})),
    ])
    assert said.splitlines() == [
        ACCOUNT_HEADER,
        "- #0231: Done",
        "- #0205: with its agent",
        "- #0220 (Draft reply to Tom Reyes): with its agent",
        "- #0282: waiting for your approval",
    ]


def test_a_refusal_is_counted_never_quoted():
    """The guards' refusals name calls for the model; the owner never reads them."""
    said = account_of([
        ("platform_update_task_status", {"task_id": 422, "status": "done"},
         _refused("#0422: the owner asked to cancel, not approve. Use platform_update_task_status with "
                  "status 'cancelled'. Nothing was done.")),
        ("platform_schedule_playbook", {"playbook_id": 110, "enabled": False},
         _ok(success=True, message="Playbook 'Weekly Social Posts' timer switched off.")),
    ])
    assert said.splitlines() == [ACCOUNT_HEADER, "- Playbook 'Weekly Social Posts' timer switched off.",
                                 "- 1 other request didn't go through, so it changed nothing."]
    assert "platform_" not in said and "Nothing was done" not in said


@pytest.mark.parametrize("outcomes, why", [
    ([], WHY_NO_CALL),
    ([("platform_board_summary", {}, _ok(total=4))], WHY_ONLY_READS),
    ([("platform_cancel_mission", {"mission_id": "0365"}, _refused("No mission #0365. Nothing was done."))],
     WHY_ALL_FAILED),
])
def test_when_nothing_changed_the_reply_says_so(outcomes, why):
    assert account_of(outcomes) == NOTHING_CHANGED.format(why=why)


# ── the loop: a blank answer becomes the account ────────────────────────────

class _Model:
    def __init__(self, *texts):
        self.queue = [LLMResponse(content=text, tool_calls=None) for text in texts]

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


def _move(task_id, status):
    return {"id": f"call_{task_id}", "type": "function", "function": {"name": "platform_execute", "arguments": json.dumps(
        {"action": "platform_update_task_status", "params": {"task_id": task_id, "status": status}})}}


def _loop_turn(lane, first, *texts):
    async def tools(name, args, call_id, workspace_id):
        params = args["params"]
        return _ok(task_id=1000 + params["task_id"], number=f"#{params['task_id']:04d}", status=params["status"])

    executor = ToolLoopExecutor(llm_callback=_Model(*texts), tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "Please cancel #0422."}]
    with usage_scope(request_type=lane, execution_id=f"{lane}:1"):
        return asyncio.run(executor.run(initial_response=first, messages=messages, tools=TOOLS, workspace_id=WS))


def test_the_nudges_empty_retry_says_what_the_turns_call_did():
    """Night 8 05:43: #0422 moved to done, "moved to cancelled" claimed, the retry empty."""
    result = _loop_turn(LANE_CHAT, LLMResponse(content="", tool_calls=[_move(422, "done")]),
                        "Task #0422 has been moved to 'cancelled'.", "")
    assert APOLOGY not in result.response.content
    assert result.response.content.splitlines() == [ACCOUNT_HEADER, "- #0422: Done"]
    assert result.response.streamed is False                     # the chat shows it: it was never streamed


def test_an_empty_answer_after_a_claim_with_no_call_says_nothing_changed():
    result = _loop_turn(LANE_CHAT, LLMResponse(content="I've cancelled Mission #0336.", tool_calls=None), "")
    assert result.response.content == NOTHING_CHANGED.format(why=WHY_NO_CALL)


def test_an_agent_runs_empty_answer_is_left_to_its_run():
    result = _loop_turn(LANE_BOARD_TASK, LLMResponse(content="", tool_calls=[_move(231, "done")]), "")
    assert result.response.content == ""


# ── the chat turn ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the F186 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None,
              finish_reason="tool_calls" if calls else "stop")


class _ChatModel:
    def __init__(self, *texts):
        self.texts, self.sent = list(texts), []

    async def generate_response(self, messages, tools=None, on_delta=None):
        self.sent.append(copy.deepcopy(messages))
        text = self.texts.pop(0)
        if on_delta is not None and text:
            await on_delta("text", text)
        return _round(text)


class _Router:
    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        return _ok(task_id=1452, number="#0231", status="done")


def test_the_chat_turn_answers_with_the_account_not_the_apology():
    """Night 8 00:30:17: #0231 approved, the answer came back empty."""
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router()
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    runtime = NS(llm_manager=_ChatModel(""), agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."},
                {"role": "user", "content": "#0231 is a ticket on my board, in Review. Approve it with my note."}]

    async def run():
        final = None
        async for chunk in svc._stream_tool_loop(_round("", [_move(231, "done")]), messages, runtime, {}, TOOLS,
                                                 streamed_rounds=[], reasoning_log=[]):
            if isinstance(chunk, dict) and chunk.get("_final_response"):
                final = chunk["_final_response"]
        return final

    with usage_scope(request_type=LANE_CHAT, execution_id="chat:de1be3f0"):
        final = asyncio.run(run())
    assert final.content.splitlines() == [ACCOUNT_HEADER, "- #0231: Done"]
    assert not final.streamed
