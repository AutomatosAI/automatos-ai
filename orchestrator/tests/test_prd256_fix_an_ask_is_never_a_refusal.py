"""PRD-256 P256-FIX-RVW-4: an owner's-click ask reaches no reader of the loop as a refusal.

FX-004 fixed the receipt, the router's text and the tool-end line. The chat's tool callback
hands the loop the router's envelope (``{success: False, llm_context, raw_result: {
requires_confirmation: True, …}}``), and the loop's other readers looked only at its top
level: the tracker recorded the ask as failed with "it reported a failure", so a claim after
it was nudged with "A write in this turn was refused: platform_update_agent … tell the owner
plainly that it was refused", an identical retry was told REFUSED_AGAIN ("this exact call
already failed … tell the owner it could not be done"), and F205's owner-words re-prompt and
F030's identical-failure guard counted it a failure.

Now an ask is waiting: neither failed, refused nor succeeded. A claim after it is nudged with
the card that waits, a repeat is skipped with the card's words, and F030 never counts it. A
genuine refusal, through the same envelope, is still refused, now with the tool's own reason.
"""
from __future__ import annotations

import asyncio
import json

from core.llm.clients.base import LLMResponse
from modules.tools.execution.nudges import is_nudge, refused_writes
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.test_prd256_fix_waiting_receipt import UPDATE_AGENT, _agent_ask, _chats, _routed

ACT = "change an agent 'Scout' (agent #12)"
CARD = f"Card raised: {ACT}. Nothing changes until the owner clicks"
PARAMS = {"agent_id": 12, "name": "Scout", "model": "claude-sonnet-5-5"}
SAID_DONE = "I've changed Scout's model to Sonnet, so it's ready for tomorrow."
SAID_WAITING = "The approval card is up: click Approve and Scout moves to Sonnet."
NO_AGENT = "No agent #99 in this workspace."
TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]


class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


def _update_call(call_id="call_update", params=None):
    args = {"action": UPDATE_AGENT, "params": PARAMS if params is None else params}
    return {"id": call_id, "type": "function", "function": {"name": "platform_execute", "arguments": json.dumps(args)}}


def _ask_envelope():
    """What the chat's tool callback hands the loop for an owner-only ask (service.py _tool_callback)."""
    return _chats(_routed(_agent_ask()))


def _refusal_envelope():
    """The same envelope for a genuine refusal: the reason sits in ``raw_result``."""
    return _chats(_routed({"success": False, "error": NO_AGENT}))


def _run(answer_with, *replies):
    """The loop, from a first response that makes the update call, over the chat-shaped result.

    The envelope is built before the loop runs: ``_routed`` runs the router with
    ``asyncio.run``, which cannot be called inside the executor's running loop."""
    ran = []
    envelope = answer_with()

    async def tools(name, args, call_id, workspace_id):
        ran.append(args)
        return {**envelope}

    executor = ToolLoopExecutor(llm_callback=_Model(*replies), tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "move Scout to Sonnet"}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[_update_call()]),
                                      messages=messages, tools=TOOLS, workspace_id="ws"))
    return executor, messages, ran, result


def _nudges(messages):
    return [m["content"] for m in messages if is_nudge(m)]


def _tool_results(messages):
    return [m["content"] for m in messages if m.get("role") == "tool"]


# ── the tracker: an ask is waiting ──────────────────────────────────────────────

def test_the_tracker_records_an_ask_inside_the_chats_envelope_as_waiting():
    tracker = ToolExecutionTracker()
    args = {"action": UPDATE_AGENT, "params": PARAMS}
    tracker.record_execution("platform_execute", args)
    tracker.record_outcome("platform_execute", args, _ask_envelope())

    assert (tracker.failed, tracker.succeeded, tracker.refused) == (set(), set(), {})
    assert len(tracker.outcomes) == 1                                   # its receipt still reads it (waiting)
    assert refused_writes(tracker.outcomes) == []
    assert list(tracker.waiting.values())[0].startswith(CARD)


# ── through the loop: the nudge, the repeat ─────────────────────────────────────

def test_a_claim_after_an_ask_is_nudged_with_its_card_never_a_refusal():
    executor, messages, ran, result = _run(_ask_envelope,
                                           LLMResponse(content=SAID_DONE, tool_calls=None),
                                           LLMResponse(content=SAID_WAITING, tool_calls=None))
    nudges = _nudges(messages)

    assert len(ran) == 1 and len(nudges) == 1
    assert f"Card raised: {ACT}." in nudges[0]
    assert "refused" not in nudges[0].lower()
    assert "it reported a failure" not in nudges[0]
    assert "A write in this turn was refused" not in nudges[0]
    assert not executor.tracker.failed                                  # F205's re-prompt reads this
    assert result.response.content == SAID_WAITING


def test_the_same_ask_repeated_is_skipped_with_the_cards_words():
    executor, messages, ran, _result = _run(_ask_envelope,
                                            LLMResponse(content="", tool_calls=[_update_call("call_again")]),
                                            LLMResponse(content=SAID_WAITING, tool_calls=None),
                                            LLMResponse(content=SAID_WAITING, tool_calls=None))   # spare
    repeat = _tool_results(messages)[-1]

    assert len(ran) == 1                                                # the repeat raised no second card
    assert repeat.startswith("Skipped: this exact call already waits for the owner")
    assert CARD in repeat
    assert "already failed" not in repeat and "could not be done" not in repeat
    assert "did its work" not in repeat                                 # nor is it said to have run
    assert not executor.tracker.failed


# ── a genuine refusal is still a refusal ────────────────────────────────────────

def test_a_genuine_refusal_still_gets_refused_again_and_the_refused_write_note():
    executor, messages, ran, _result = _run(_refusal_envelope,
                                            LLMResponse(content="", tool_calls=[_update_call("call_again")]),
                                            LLMResponse(content=SAID_DONE, tool_calls=None),
                                            LLMResponse(content="That change was refused: no agent #99.",
                                                        tool_calls=None))
    repeat = _tool_results(messages)[1]
    nudges = _nudges(messages)

    assert len(ran) == 1
    assert repeat.startswith("Skipped: Skipped: this exact call already failed in this reply")
    assert NO_AGENT in repeat                                           # the tool's own reason, from raw_result
    assert len(nudges) == 1 and "A write in this turn was refused" in nudges[0]
    assert f'{UPDATE_AGENT} (the tool said: "{NO_AGENT}")' in nudges[0]
    assert "Card raised" not in nudges[0]
    assert executor.tracker.failed == {UPDATE_AGENT}


# ── F030: an ask is never an identical failure ──────────────────────────────────

def test_f030_never_counts_an_ask_but_still_counts_a_refusal():
    from consumers.chatbot.same_failure import same_failure_key

    assert same_failure_key(UPDATE_AGENT, _ask_envelope()) is None
    assert same_failure_key(UPDATE_AGENT, _refusal_envelope()) == f"{UPDATE_AGENT}:{NO_AGENT}"
    assert same_failure_key(UPDATE_AGENT, {"success": True}) is None


def test_the_chat_reads_f030_from_the_one_module():
    from consumers.chatbot import service, same_failure

    assert service._same_failure_key is same_failure.same_failure_key
    assert service.MAX_IDENTICAL_TOOL_FAILURES == same_failure.MAX_IDENTICAL_TOOL_FAILURES
