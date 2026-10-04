"""F295 (night 8): a nudge is the user's turn, so the model answers it.

#0207's Inventory Watchdog (claude-sonnet-4 over OpenRouter) answered the card in
435 tokens; the loop took the answer for narrated actions and nudged it with a
system message after the reply. OpenRouter folds system messages into Anthropic's
system prompt, so the conversation the model saw ended on its own reply, and the
retry came back with 3 tokens (llm_usage rows 77947 → 77952). It happened on all
16 nudged retries of the night's failed runs.

``_OpenRouterToAnthropic`` does what that route does to the conversation: system
messages leave it, and a conversation that ends on the model's own reply gets
nothing more.
"""
from __future__ import annotations

import asyncio

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "platform_create_agent", "parameters": {}}}]
# A real answer that reads like narration to the loop (two "let me" cues).
ANSWER_0207 = ("Let me work it out. Now let me check the orders: 38 kg on the shelf, 41 kg ordered for "
               "Tuesday, so yes, roast an extra 3 kg of Harbour Blend on Tuesday.")
RETRY = "Yes: roast an extra 3 kg of Harbour Blend on Tuesday (41 kg ordered, 38 kg on the shelf)."


class _OpenRouterToAnthropic:
    def __init__(self) -> None:
        self.sent = []

    async def __call__(self, messages, tools):
        turns = [m for m in messages if m["role"] != "system"]
        self.sent.append(turns)
        if turns[-1]["role"] == "assistant":
            return LLMResponse(content="", tool_calls=None)       # the 3-token reply
        return LLMResponse(content=RETRY, tool_calls=None)


def _run(model):
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=None, max_iterations=5)
    messages = [{"role": "system", "content": "You are the Inventory Watchdog."},
                {"role": "user", "content": "Do we need an extra Harbour Blend roast on Tuesday?"}]
    return asyncio.run(executor.run(initial_response=LLMResponse(content=ANSWER_0207, tool_calls=None),
                                    messages=messages, tools=TOOLS, workspace_id="ws"))


def test_the_nudged_retry_is_sent_a_conversation_that_ends_on_the_users_turn():
    model = _OpenRouterToAnthropic()
    _run(model)
    assert len(model.sent) == 1
    assert model.sent[0][-1]["role"] == "user"
    assert model.sent[0][-2] == {"role": "assistant", "content": ANSWER_0207}


def test_the_card_gets_the_retrys_answer():
    assert _run(_OpenRouterToAnthropic()).response.content == RETRY
