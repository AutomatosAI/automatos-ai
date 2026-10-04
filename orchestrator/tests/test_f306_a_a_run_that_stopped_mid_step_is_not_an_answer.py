"""F306 (night 9) — a run that stopped at a step it announced is not the card's answer.

F297 caught a run that ended on raw tool output, not one that stopped mid-thought.
Night 9's cards kept a half-sentence as their answer: "Let me try a more specific
query:" (#1879, twice), "Let me try asking the orchestrator for guidance on this
technical issue:" (#1856), "Now let me get the total kilograms per account to see
which cafés order the most:" (#1881), and "I'll attempt a query to list all tables
in" (#1888, cut off). The loop now nudges such a reply once (make the call, or write
the answer), and a result that still ends on one is said plainly: review for a card,
a failed attempt for a mission step.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.llm.clients.base import LLMResponse
from modules.agents.factory.answer_check import said_plainly
from modules.tools.execution.nudges import announced_step
from modules.tools.execution.tool_loop import ToolLoopExecutor
from services.result_substance import STOPPED_HEADER, STOPPED_NOTE, as_step_failure, nothing_done_note, stopped_mid_step

TOOLS = [{"type": "function", "function": {"name": "smart_query_database"}}]
STOPPED_1879 = "Let me try a more specific query:"
STOPPED_1881 = ("I can see from the search results that the product containing Brazil Cerrado is the Harbour "
                "Blend. Let me now query the shop system to get the total kilograms per café:")
CUT_OFF_1888 = "The query failed on wo.order_date.\n\nI'll attempt a query to list all tables in"


@pytest.mark.parametrize("text", [STOPPED_1879, STOPPED_1881, CUT_OFF_1888,
                                  "Let me try asking the orchestrator for guidance on this technical issue:"])
def test_a_reply_that_stops_at_the_step_it_announces_is_seen(text):
    assert announced_step(text) == text.splitlines()[-1].strip()[:160]


@pytest.mark.parametrize("text", [
    "Corner Bakehouse 30 kg, The Lantern Room 24 kg.",
    "Here is the table.\n\n| Café | kg |\n|---|---|\n| Kiln | 214 |",
    "Let me know if you want the October figures too.",
    "I'll send the price list on Thursday.",
])
def test_an_answer_is_never_taken_for_a_stopped_step(text):
    assert announced_step(text) is None


class _Model:
    def __init__(self, *responses):
        self.queue, self.calls = list(responses), []

    async def __call__(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.queue.pop(0)


class _Tools:
    def __init__(self):
        self.executed = []

    async def __call__(self, name, args, call_id, workspace_id):
        self.executed.append(name)
        return {"success": True, "rows": [{"account": "Kiln Bakehouse", "kg": "214.0"}]}


def _query():
    return {"id": "call_q", "type": "function",
            "function": {"name": "smart_query_database", "arguments": json.dumps({"question": "kg per café"})}}


def test_the_loop_nudges_a_stopped_step_once_and_the_retry_makes_the_call():
    model = _Model(LLMResponse(content="", tool_calls=[_query()]),
                   LLMResponse(content="Kiln Bakehouse 214 kg.", tool_calls=None))
    tools = _Tools()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "Which cafés took the most coffee in September?"}]

    result = asyncio.run(executor.run(initial_response=LLMResponse(content=STOPPED_1879, tool_calls=None),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))

    assert tools.executed == ["smart_query_database"]                   # night 9: the step never ran
    assert result.response.content == "Kiln Bakehouse 214 kg."
    nudge = model.calls[0][-1]
    assert nudge["role"] == "user" and STOPPED_1879 in nudge["content"]


def test_a_result_that_still_stops_mid_step_says_so_and_keeps_what_it_wrote():
    plain = stopped_mid_step(STOPPED_1881)

    assert plain.splitlines()[0] == STOPPED_HEADER
    assert STOPPED_1881 in plain                                          # nothing the agent wrote is lost
    assert nothing_done_note(plain) == STOPPED_NOTE                       # the card goes to review
    assert as_step_failure({"status": "success", "result": plain})["status"] == "error"   # a step retries
    assert stopped_mid_step("Kiln Bakehouse 214 kg.") is None


def test_every_agent_run_gets_the_plain_result():
    async def _run(*args, **kwargs):
        return {"status": "success", "result": STOPPED_1879}

    out = asyncio.run(said_plainly(_run)())

    assert out["result"].startswith(STOPPED_HEADER) and out["status"] == "success"
