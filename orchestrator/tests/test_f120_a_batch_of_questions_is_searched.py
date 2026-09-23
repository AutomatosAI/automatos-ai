"""F120 (run 4) — eight questions in one message are eight searches.

The owner asked eight questions in one message; the model issued eight
search_knowledge calls in one response. The tracker counted them as retries
against a per-turn limit of 5, so the last three were skipped, and the chat
told the model "Do NOT call search_knowledge again" after the second — Q6–Q8
came back "I cannot find this in the provided documents" (7/8 asked one at a
time, 6/8 batched, same documents). Now a response's calls are one batch: they
all run up to a ceiling of 12, only an identical query repeats within it, a
similar query is a repeat only of an earlier response's, the nudge counts
responses, and a skipped search names the question it left.
"""
from __future__ import annotations

import asyncio
import inspect
import json

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.tool_loop import ToolLoopExecutor

# The persona's by-name batch: one price per coffee — similar in shape, eight questions.
QUESTIONS = [f"What is the wholesale price per kilo of {name}?" for name in (
    "Guji Hambela Alaka", "Guji Shakiso Natural", "Sidamo Bensa", "Yirgacheffe Kochere",
    "Huila Pink Bourbon", "Cerrado Mineiro", "Kayanza Gatukuza", "Nyeri Gatomboya")]


def _batch(tracker, queries):
    tracker.begin_round()
    skipped = {}
    for q in queries:
        skip, reason = tracker.should_skip_execution("search_knowledge", {"query": q, "limit": 5})
        if skip:
            skipped[q] = reason
        else:
            tracker.record_execution("search_knowledge", {"query": q, "limit": 5})
    return skipped


def test_eight_questions_in_one_response_are_all_searched():
    assert _batch(ToolExecutionTracker(), QUESTIONS) == {}


def test_the_same_query_twice_is_still_a_repeat_and_says_so():
    skipped = _batch(ToolExecutionTracker(), [QUESTIONS[0], QUESTIONS[0]])
    assert list(skipped) == [QUESTIONS[0]]
    assert skipped[QUESTIONS[0]].startswith(f'Not searched again: "{QUESTIONS[0]}"')


def test_a_rephrasing_in_a_later_response_is_a_repeat():
    tracker = ToolExecutionTracker()
    _batch(tracker, [QUESTIONS[0]])
    rephrased = "what is the wholesale price per kilo of Guji Hambela Alaka"
    skipped = _batch(tracker, [rephrased])
    assert rephrased in skipped and QUESTIONS[0] in skipped[rephrased]


def test_a_thirteenth_search_is_capped_and_named_not_searched():
    tracker = ToolExecutionTracker()
    distinct = QUESTIONS + [f"Who roasts batch {n} this week?" for n in range(1, 6)]
    skipped = _batch(tracker, distinct)
    assert list(skipped) == [distinct[12]]
    reason = skipped[distinct[12]]
    assert reason.startswith(f'Not searched: "{distinct[12]}"') and "not in the documents" in reason


# ── through the tool loop ───────────────────────────────────────────────────

class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


def _search(i, query):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": "search_knowledge", "arguments": json.dumps({"query": query, "limit": 5})}}


def test_the_loop_runs_all_eight_searches_of_one_response():
    searched = []

    async def tools(name, args, call_id, workspace_id):
        searched.append(args["query"])
        return {"success": True, "results": [{"content": f"answer to {args['query']}"}]}

    executor = ToolLoopExecutor(llm_callback=_Model(LLMResponse(content="All eight answered.", tool_calls=None)),
                                tool_callback=tools, max_iterations=5)
    first = LLMResponse(content="", tool_calls=[_search(i, q) for i, q in enumerate(QUESTIONS)])
    asyncio.run(executor.run(initial_response=first, messages=[{"role": "user", "content": "eight questions"}],
                             tools=[{"type": "function", "function": {"name": "search_knowledge"}}],
                             workspace_id="ws"))
    assert searched == QUESTIONS


# ── the chat's "do not call it again" nudge ─────────────────────────────────

def _nudge(rounds, *, attempts, nudged=False):
    from consumers.chatbot.service import StreamingChatService

    svc = StreamingChatService.__new__(StreamingChatService)
    messages = []
    fired = svc._inject_loop_prevention(messages, "search_knowledge", {"search_knowledge": attempts}, 0, {},
                                        None, set(), rounds=rounds, nudged_this_round=nudged)
    return fired, messages


def test_the_nudge_counts_responses_not_calls():
    assert _nudge(1, attempts=8) == (False, [])                    # one batch of eight: no nudge
    fired, messages = _nudge(2, attempts=9)                         # called again in a second response
    assert fired and "Do NOT call `search_knowledge` again" in messages[0]["content"]
    assert _nudge(2, attempts=10, nudged=True) == (False, [])       # once per response, not per call


def test_the_chat_loop_tracks_the_response_each_call_came_in():
    from consumers.chatbot import service

    loop = inspect.getsource(service.StreamingChatService._stream_tool_loop)
    assert "tool_rounds.setdefault(name, set()).add(_round_index)" in loop
    assert "rounds=len(tool_rounds.get(name) or ())" in loop
    assert "_round_index += 1" in loop
