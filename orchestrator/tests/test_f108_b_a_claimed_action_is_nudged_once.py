"""F108 (night 3), the check — a reply that says an action was done with no action
behind it never reaches the owner as a report of that action.

F099's mechanism for a source that did not run, applied to claimed actions. The
check runs on the first reply and on the loop's final reply (night 3's "I've
approved the mission. It's now running" came after other tools had run).

P256-FIX-N15-1 (F422, night 15): it used to nudge the reply and return the retry in its
place; the retry answered the nudge and dropped what the reads had fetched. Now the reply
stands, the sentences that claim are replaced by one not-done line, and the model is not
called again.
"""
from __future__ import annotations

import asyncio
import json

from core.llm.clients.base import LLMResponse
from modules.tools.execution.nudges import is_nudge
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]
NOT_APPROVED = "Just to be clear: I haven't approved anything in this reply. Ask me again if you want it done."


class _Model:
    def __init__(self, *responses):
        self.queue, self.calls = list(responses), []

    async def __call__(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.queue.pop(0)


class _Tools:
    def __init__(self, refuse=()):
        self.executed, self.refuse = [], set(refuse)

    async def __call__(self, name, args, call_id, workspace_id):
        self.executed.append(args["action"])
        return {"success": args["action"] not in self.refuse}


def _call(action):
    return {"id": f"call_{action}", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


def _reply(text):
    return LLMResponse(content=text, tool_calls=None)


def _run(model, tools, first):
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5, promises=True)
    messages = [{"role": "user", "content": "approve the Harbourline mission"}]
    result = asyncio.run(executor.run(initial_response=first, messages=messages, tools=TOOLS, workspace_id="ws"))
    nudges = [m for m in messages if is_nudge(m)]
    return result, nudges


def test_the_final_reply_claims_an_approval_and_the_line_takes_its_place():
    model = _Model(_reply("I've approved the mission. It's now running."))
    tools = _Tools()
    result, nudges = _run(model, tools, LLMResponse(content="", tool_calls=[_call("platform_get_mission")]))
    assert tools.executed == ["platform_get_mission"]
    assert result.response.content == NOT_APPROVED
    assert nudges == [] and len(model.calls) == 1                       # no retry


def test_what_the_reply_says_besides_the_claim_stays():
    model = _Model(_reply("The mission is waiting in Review. I've approved the mission."))
    result, nudges = _run(model, _Tools(), LLMResponse(content="", tool_calls=[_call("platform_get_mission")]))
    assert result.response.content == f"The mission is waiting in Review. {NOT_APPROVED}"
    assert nudges == [] and len(model.calls) == 1


def test_a_refused_approval_does_not_back_the_claim():
    model = _Model(_reply("I've approved the mission."))
    tools = _Tools(refuse={"platform_approve_mission"})
    result, nudges = _run(model, tools, LLMResponse(content="", tool_calls=[_call("platform_approve_mission")]))
    assert nudges == [] and result.response.content == (
        "Just to be clear: I haven't approved anything in this reply; I tried to approve the mission and it didn't "
        "go through. Ask me again if you want it done.")


def test_an_approval_that_ran_is_left_as_it_is():
    model = _Model(_reply("I've approved the mission. It's now running."))
    result, nudges = _run(model, _Tools(), LLMResponse(content="", tool_calls=[_call("platform_approve_mission")]))
    assert nudges == [] and len(model.calls) == 1
    assert result.response.content == "I've approved the mission. It's now running."


def test_a_first_reply_that_claims_is_checked_too():
    model = _Model()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_Tools(), max_iterations=5, promises=True)
    messages = [{"role": "user", "content": "the Taster plan is £14 now"}]
    result = asyncio.run(executor.run(initial_response=_reply("I've noted that the Taster plan is now £14."),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    assert model.calls == [] and not any(is_nudge(m) for m in messages)
    assert result.response.content == (
        "Just to be clear: I haven't noted anything in this reply. Ask me again if you want it done.")


def test_without_tools_offered_nothing_is_checked():
    model = _Model()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_Tools(), max_iterations=5)
    result = asyncio.run(executor.run(initial_response=_reply("I've approved the mission."),
                                      messages=[{"role": "user", "content": "hi"}], tools=None, workspace_id="ws"))
    assert model.calls == [] and result.response.content == "I've approved the mission."
