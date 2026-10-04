"""F328 (night 9b) — a long card that reaches its round cap still ends on its answer.

#1994 (Ops, "Monday brief for 5 October") stopped twice and #1970 (BA, "Kirinyaga: Quay
plus Christmas") once at a step it announced. The backend log shows each run reached
"[tool-loop] iteration 10: 1 tool call(s)" and then "Hit max tool iterations (10)": the
11th reply asked for one more call, and the words before that call ("Let me check the
payment terms report:") were all the card got, so F306 sent it to review. Now the
agent run is asked once, with no tools, for its answer from what it already has, and
that answer is the card's. Chat turns keep their own handling of the cap.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_BOARD_TASK, LANE_CHAT, usage_scope
from modules.agents.factory.answer_check import said_plainly
from modules.tools.execution.tool_loop import ToolLoopExecutor
from services.result_substance import STOPPED_HEADER

CAP = 10  # a board card's execute_with_prompt default (max_tool_iterations=10)
STOPPED_1994_RUN_1 = "Let me check the payment terms report:"
STOPPED_1994_RUN_2 = "Now let me get supplier information:"
STOPPED_1970 = "Let me check Quay Bakehouse's recent orders through September:"
# #1994's third run, abridged: the brief the card should have carried.
BRIEF_1994 = ("## Monday Morning Brief - 5 October 2026\n\n**Club boxes going out Monday:** 63 Harvest Club "
              "boxes.\n\n**Reorder needed:**\n- **Kirinyaga AA** from Tidewater Importers (3 weeks lead time)")
TOOLS = [{"type": "function", "function": {"name": "platform_execute"}},
         {"type": "function", "function": {"name": "search_knowledge"}}]


def _call(i: int) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": "query", "n": i})}}


class _Model:
    """Asks for one new call a round (night 9b: a new tool each round), then the
    reply that announces a step past the cap, then whatever it is scripted to say."""

    def __init__(self, stopped_at: str, *after: LLMResponse):
        rounds = [LLMResponse(content="", tool_calls=[_call(i)]) for i in range(1, CAP)]
        self.queue = [*rounds, LLMResponse(content=stopped_at, tool_calls=[_call(99)]), *after]
        self.calls = []

    async def __call__(self, messages, tools):
        self.calls.append(([dict(m) for m in messages], tools))
        return self.queue.pop(0)


class _Tools:
    def __init__(self):
        self.ran = []

    async def __call__(self, name, args, call_id, workspace_id):
        self.ran.append(args.get("n"))
        return {"success": True, "llm_context": json.dumps({"rows": [{"n": args.get("n")}]})}


def _run(model, lane=LANE_BOARD_TASK):
    tools = _Tools()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=CAP)
    messages = [{"role": "user", "content": "Write my Monday brief for 5 October."}]

    async def go():
        with usage_scope(request_type=lane, execution_id="board_task:1994"):
            return await executor.run(initial_response=LLMResponse(content="", tool_calls=[_call(0)]),
                                      messages=messages, tools=TOOLS, workspace_id="ws")

    return asyncio.run(go()), tools


@pytest.mark.parametrize("stopped_at", [STOPPED_1994_RUN_1, STOPPED_1994_RUN_2, STOPPED_1970])
def test_a_card_that_reaches_its_cap_gets_its_answer_not_the_step_it_announced(stopped_at):
    model = _Model(stopped_at, LLMResponse(content=BRIEF_1994, tool_calls=None))

    result, tools = _run(model)

    assert result.response.content == BRIEF_1994                       # night 9b: the step was the answer
    assert result.max_iterations_reached                                 # the cap still held
    assert tools.ran == list(range(CAP))                                 # and no call ran past it
    messages, offered = model.calls[-1]
    assert offered is None                                               # asked with tools off
    assert messages[-2] == {"role": "assistant", "content": stopped_at}  # what it announced stays
    assert messages[-1]["role"] == "user" and stopped_at in messages[-1]["content"]


def test_one_more_call_at_most_and_a_reply_that_still_asks_for_a_tool_is_sent_to_review():
    model = _Model(STOPPED_1994_RUN_1, LLMResponse(content="Let me look again:", tool_calls=[_call(100)]))

    result, tools = _run(model)

    assert len(model.calls) == CAP + 1                                   # ten rounds, then one ask
    assert tools.ran == list(range(CAP))
    assert result.response.content == STOPPED_1994_RUN_1                # the run ends where it stopped

    async def agent_run(*_a, **_k):
        return {"status": "success", "result": result.response.content}

    assert asyncio.run(said_plainly(agent_run)())["result"].startswith(STOPPED_HEADER)   # F306: review


def test_a_chat_turn_handles_its_own_cap():
    model = _Model(STOPPED_1994_RUN_1, LLMResponse(content="not asked", tool_calls=None))

    result, _tools = _run(model, lane=LANE_CHAT)

    assert len(model.calls) == CAP                                       # no extra call from the loop
    assert result.max_iterations_reached and result.response.tool_calls  # chat synthesizes it itself
