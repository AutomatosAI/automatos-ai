"""F108 (night 3), the nudge — a reply that says an action was done with no action
behind it is retried once: make the call, or say plainly it has not happened.

F099's mechanism for a source that did not run, applied to claimed actions. The
check runs on the first reply and on the loop's final reply (night 3's "I've
approved the mission. It's now running" came after other tools had run); a
retry that makes the call runs it in the same loop. One nudge per run.
"""
from __future__ import annotations

import asyncio
import json

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]
NUDGE = "says something was approved, but no tool call in this turn did that"


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
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "approve the Harbourline mission"}]
    result = asyncio.run(executor.run(initial_response=first, messages=messages, tools=TOOLS, workspace_id="ws"))
    nudges = [m for m in messages if m["role"] == "system" and NUDGE in m["content"]]
    return result, nudges


def test_the_final_reply_claims_an_approval_and_the_retry_makes_the_call():
    model = _Model(_reply("I've approved the mission. It's now running."),
                   LLMResponse(content="", tool_calls=[_call("platform_approve_mission")]),
                   _reply("Approved — the mission is running."))
    tools = _Tools()
    result, nudges = _run(model, tools, LLMResponse(content="", tool_calls=[_call("platform_get_mission")]))
    assert tools.executed == ["platform_get_mission", "platform_approve_mission"]
    assert result.response.content == "Approved — the mission is running."
    assert len(nudges) == 1


def test_a_retry_that_says_plainly_it_has_not_happened_is_the_reply():
    model = _Model(_reply("I've approved the mission. It's now running."),
                   _reply("I have not approved it — the approval is refused for agents; it needs you."))
    tools = _Tools()
    result, nudges = _run(model, tools, LLMResponse(content="", tool_calls=[_call("platform_get_mission")]))
    assert tools.executed == ["platform_get_mission"]
    assert result.response.content.startswith("I have not approved it")
    assert len(nudges) == 1


def test_a_refused_approval_does_not_back_the_claim():
    model = _Model(_reply("I've approved the mission."), _reply("The approval was refused — it needs you."))
    tools = _Tools(refuse={"platform_approve_mission"})
    result, nudges = _run(model, tools, LLMResponse(content="", tool_calls=[_call("platform_approve_mission")]))
    assert len(nudges) == 1 and result.response.content.startswith("The approval was refused")


def test_an_approval_that_ran_is_not_nudged():
    model = _Model(_reply("I've approved the mission. It's now running."))
    result, nudges = _run(model, _Tools(), LLMResponse(content="", tool_calls=[_call("platform_approve_mission")]))
    assert nudges == [] and len(model.calls) == 1
    assert result.response.content == "I've approved the mission. It's now running."


def test_one_nudge_per_run():
    model = _Model(_reply("I've approved the mission."), _reply("I've approved it, as I said."))
    result, nudges = _run(model, _Tools(), LLMResponse(content="", tool_calls=[_call("platform_get_mission")]))
    assert len(nudges) == 1 and len(model.calls) == 2
    assert result.response.content == "I've approved it, as I said."    # the chat notice says so


def test_a_first_reply_that_claims_is_nudged_too():
    model = _Model(LLMResponse(content="", tool_calls=[_call("platform_store_memory")]), _reply("Saved."))
    tools = _Tools()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "the Taster plan is £14 now"}]
    result = asyncio.run(executor.run(initial_response=_reply("I've noted that the Taster plan is now £14."),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    assert tools.executed == ["platform_store_memory"] and result.response.content == "Saved."
    assert any(m["role"] == "system" and "says something was noted" in m["content"] for m in messages)


def test_without_tools_offered_nothing_is_nudged():
    model = _Model()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_Tools(), max_iterations=5)
    result = asyncio.run(executor.run(initial_response=_reply("I've approved the mission."),
                                      messages=[{"role": "user", "content": "hi"}], tools=None, workspace_id="ws"))
    assert model.calls == [] and result.response.content == "I've approved the mission."
