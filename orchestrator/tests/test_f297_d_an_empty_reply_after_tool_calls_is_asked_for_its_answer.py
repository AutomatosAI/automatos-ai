"""F297 (night 8): an empty reply straight after a round of tool calls is asked once
for the answer.

#0234's Analyst wrote its CSV with workspace_write_file, then its next reply was
empty (0 tokens, 00:29:02), so the card's answer became the tool's output. The run
is now asked once, as the user's turn, for the work itself.
"""
from __future__ import annotations

import asyncio
import json

from core.llm.clients.base import LLMResponse
from modules.tools.execution.nudges import MISSING_ANSWER_MSG, as_a_check
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "workspace_write_file", "parameters": {}}}]
TABLE = "| Cost | £3.03 |\n| Margin | £6.47 (68.1%) |\n\nSaved as decaf_colombia_margin.csv."
WRITE = {"id": "c1", "type": "function",
         "function": {"name": "workspace_write_file",
                      "arguments": json.dumps({"path": "decaf_colombia_margin.csv", "content": "a,b"})}}


class _Model:
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    async def __call__(self, messages, tools):
        self.sent.append(list(messages))
        return self.replies.pop(0)


async def _write(name, args, call_id, workspace_id):
    return {"success": True, "llm_context": json.dumps({"written": True, "path": args["path"]})}


def _run(*replies):
    model = _Model(*replies)
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_write, max_iterations=5)
    messages = [{"role": "user", "content": "Decaf Colombia margin: the table on the card and a CSV"}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[WRITE]),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model


def test_0234_is_asked_once_and_the_card_gets_the_table():
    result, model = _run(LLMResponse(content="", tool_calls=None), LLMResponse(content=TABLE, tool_calls=None))
    assert result.response.content == TABLE
    asked = model.sent[1]
    assert asked[-2]["role"] == "tool" and asked[-1] == {"role": "user", "content": as_a_check(MISSING_ANSWER_MSG)}


def test_an_answer_after_the_tool_calls_is_not_asked_again():
    result, model = _run(LLMResponse(content=TABLE, tool_calls=None))
    assert result.response.content == TABLE and len(model.sent) == 1


def test_a_second_empty_reply_is_not_asked_again():
    result, model = _run(LLMResponse(content="", tool_calls=None), LLMResponse(content="", tool_calls=None))
    assert len(model.sent) == 2 and not (result.response.content or "").strip()
