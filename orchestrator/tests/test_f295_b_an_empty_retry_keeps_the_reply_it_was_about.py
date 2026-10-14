"""F295 (night 8): a nudge that gets an empty reply leaves the reply it was about.

#0245's Support Agent answered in 108 tokens, the loop nudged it for a claim
("changed"), and the empty retry replaced the answer: the run failed "Empty
response from LLM" twice, on a card the next agent answered first time. An empty
retry no longer wins over the reply it was asked about; one with words or a tool
call still does.

P256-FIX-N15-1: a claim is no longer nudged at all (its sentence is replaced by the
not-done line, with no retry); the nudges that remain (a narrated reply, an announced
step) keep this rule.
"""
from __future__ import annotations

import asyncio

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {}}}]
CLAIMING_0245 = ("To: nadia@brook.example\n\nHi Nadia,\n\nThanks for letting us know. I've changed the "
                 "address on your club box, so the November box goes to the new one.\n\nGerard")
NARRATED = ("Let me create OPS. Now let me assign Gmail to it. Good — both created.")
CHANGED = "I've changed the address on your club box, so the November box goes to the new one."
NOT_CHANGED = "Just to be clear: I haven't changed anything in this reply. Ask me again if you want it done."
STEPPING_0245 = ("To: nadia@brook.example\n\nHi Nadia,\n\nThanks for letting us know.\n\n"
                 "Let me look up the club box on your account:")


class _Model:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.replies.pop(0)


def _run(first, *replies):
    model = _Model(*replies)
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=None, max_iterations=5)
    messages = [{"role": "user", "content": "Reply to Nadia Brook - change of address for her club box"}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content=first, tool_calls=None),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model


def test_a_claim_is_never_retried_and_the_answer_stays():
    """#0245's answer stands, its claim's sentence replaced by the line: the model is not called again."""
    result, model = _run(CLAIMING_0245, LLMResponse(content="  ", tool_calls=None))
    assert model.calls == 0
    assert result.response.content == CLAIMING_0245.replace(CHANGED, NOT_CHANGED)
    assert result.response.content.endswith("\n\nGerard")


def test_an_empty_retry_of_an_announced_step_leaves_the_answer():
    """F306's nudge (a reply that ends announcing a step it never took) still retries once."""
    result, model = _run(STEPPING_0245, LLMResponse(content="  ", tool_calls=None))
    assert model.calls == 1
    assert result.response.content == STEPPING_0245


def test_an_empty_retry_of_a_narrated_reply_leaves_the_reply():
    result, _ = _run(NARRATED, LLMResponse(content="", tool_calls=None))
    assert result.response.content == NARRATED


def test_a_retry_with_words_still_replaces_the_reply():
    plain = "I have not changed the address yet: the club tool needs you to do it."
    result, model = _run(STEPPING_0245, LLMResponse(content=plain, tool_calls=None))
    assert model.calls == 1 and result.response.content == plain
