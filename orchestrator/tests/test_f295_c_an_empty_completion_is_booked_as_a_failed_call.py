"""F295 (night 8): an empty completion is booked as a failed call in llm_usage.

The night's 27 empty claude-sonnet-4 completions (2 or 3 output tokens, no text,
no tool call) were booked ``success``, so the cost report, which counts rows whose
status is not success, said 9 failed calls of 17,529 while eight runs had failed on
them. The row keeps its tokens; its status says the call failed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
EMPTY = NS(content="\n", tool_calls=None, finish_reason="stop", model="anthropic/claude-sonnet-4",
           usage={"prompt_tokens": 13530, "completion_tokens": 3, "total_tokens": 13533})
CALL = NS(content="", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
          finish_reason="tool_calls", usage=None)
ANSWER = NS(content="Yes, roast 3 kg more.", tool_calls=None, finish_reason="stop", usage=None)


class _Provider:
    def __init__(self, response):
        self.response = response

    async def generate_response(self, messages, tools=None):
        return self.response


def _booked(response):
    from core.llm.clients.base import LLMConfig, LLMProvider
    from core.llm.manager import LLMManager

    mgr = LLMManager(config=LLMConfig(provider=LLMProvider.OPENROUTER, model="anthropic/claude-sonnet-4",
                                      max_tokens=8000, api_key="k"), workspace_id=WS, agent_id=328)
    booked = []
    mgr.provider = _Provider(response)
    mgr._track_usage = lambda resp, start, status="success": booked.append((resp, status))
    returned = asyncio.run(mgr.generate_response([{"role": "user", "content": "go"}]))
    return returned, booked


def test_an_empty_completion_is_a_failed_call_and_is_still_returned():
    returned, booked = _booked(EMPTY)
    assert returned is EMPTY
    assert booked == [(EMPTY, "error")]


def test_a_tool_call_or_an_answer_is_a_success():
    assert _booked(CALL)[1][0][1] == "success"
    assert _booked(ANSWER)[1][0][1] == "success"

