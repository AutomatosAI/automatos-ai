"""F186 (night 6) — a chat turn's reply is its answer. What the model said before
its tool calls is narration.

41 of night 6's 159 saved replies were two replies in one. #1110 at 03:05: the
first round told the owner the ticket was "waiting for the Shopify Support
Agent", the tools showed it done, and the second round answered "This task was
completed". Both halves were joined into the saved reply and into memory, and
the next turn read the pre-tool guess as if it had been checked. Within the
turn the loop sent the model's tool-call message with content None. Round 2
never saw what it had already said, and it greeted the owner again.

Now the round's text rides its tool-call message: on the OpenRouter/Gemini route
as sent, and on the Anthropic route as a text block beside the tool_use blocks.
Each streamed round is marked as narration or, when F108 nudged it, retracted.
The reply saved, remembered and read by the next turn is the answer, with the
narration in its own part. A nudged claim is not saved beside its retry.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
from types import MethodType, SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
NARRATION = ("I see ticket #1110 on your board, waiting for the Shopify Support Agent and for your review. "
             "I will check the current activity for you.")
ANSWER = "You got it, Gerard! Ticket #1110 was completed at 03:04."
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object",
                                                                                      "properties": {}}}}]


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the concierge tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _call(action, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None,
              finish_reason="tool_calls" if calls else "stop")


class _Model:
    """A scripted model that streams each round's text and keeps what it was sent."""

    def __init__(self, *rounds):
        self.rounds, self.sent = list(rounds), []

    async def generate_response(self, messages, tools=None, on_delta=None):
        self.sent.append(copy.deepcopy(messages))
        text, calls = self.rounds.pop(0)
        if on_delta is not None and text:
            await on_delta("text", text)
        return _round(text, calls)


class _Router:
    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        return {"success": True, "llm_context": "Ticket #1110: done at 03:04:20.", "raw_result": {"status": "done"}}


def _turn(model, first, *, rounds=None):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router()
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    runtime = NS(llm_manager=model, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."},
                {"role": "user", "content": "Is ticket 1110 still waiting on anyone?"}]
    extra = {} if rounds is None else {"streamed_rounds": rounds, "reasoning_log": []}

    async def run():
        frames, final = [], None
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, **extra):
            if isinstance(chunk, dict) and chunk.get("_final_response"):
                final = chunk["_final_response"]
            else:
                frames.append(chunk)
        return frames, final
    frames, final = asyncio.run(run())
    return frames, final


# ── within the turn ─────────────────────────────────────────────────────────

def test_round_2_sees_what_round_1_said_before_its_tool_call():
    model = _Model((ANSWER, None))
    _turn(model, _round(NARRATION, [_call("platform_get_activity_feed")]))

    (sent,) = model.sent
    tool_message = next(m for m in sent if m["role"] == "assistant" and m.get("tool_calls"))
    assert tool_message["content"] == NARRATION


# ── what is saved ───────────────────────────────────────────────────────────

def test_the_saved_reply_is_the_answer_and_the_narration_is_its_own_part():
    from consumers.chatbot.narration import reply_parts, split_reply

    first = _round(NARRATION, [_call("platform_get_activity_feed")])
    rounds = [first]
    frames, final = _turn(_Model((ANSWER, None)), first, rounds=rounds)

    narration, answer = split_reply(rounds, final, final.content)
    assert (narration, answer) == ([NARRATION], ANSWER)
    assert reply_parts("", "\n\n".join(narration), answer) == [
        {"type": "narration", "narration": NARRATION}, {"type": "text", "text": ANSWER}]


def test_a_nudged_claim_is_not_saved_beside_its_retry():
    """02:42:01: F108 nudged a claim, and the claim and its retry were both saved."""
    from consumers.chatbot.narration import split_reply

    claim, retry = "I've approved the mission. It's now running.", "I have not approved it: that needs you."
    first = _round("Let me look at the mission first.", [_call("platform_get_mission")])
    rounds = [first]
    frames, final = _turn(_Model((claim, None), (retry, None)), first, rounds=rounds)

    assert final.content == retry
    assert split_reply(rounds, final, final.content) == (["Let me look at the mission first."], retry)
    marks = [json.loads(f[2:]) for f in frames if f.startswith('d:{"type": "narration"')]
    assert {"type": "narration", "data": {"text": claim, "retracted": True}} in marks


def test_an_empty_answer_falls_back_to_the_narration():
    from consumers.chatbot.narration import split_reply

    first = _round(NARRATION, [_call("platform_get_activity_feed")])
    assert split_reply([first], None, "") == ([], NARRATION)


def test_the_chat_saves_the_split_and_remembers_only_the_answer():
    from consumers.chatbot import service

    turn = inspect.getsource(service.StreamingChatService)
    assert "narration, full_response = split_reply(streamed_rounds, final_round, final_text)" in turn
    assert "assistant_parts = reply_parts(joined_reasoning, narration_text, full_response)" in turn
    assert '"\\n\\n".join(t for t in streamed_text if t)' not in turn


def test_the_next_turn_and_the_previews_read_only_the_answer():
    from api.chat import _parts_text
    from modules.context.sections.conversation import _parts_to_text

    parts = [{"type": "narration", "narration": NARRATION}, {"type": "text", "text": ANSWER}]
    assert _parts_to_text(parts) == ANSWER and _parts_text(parts) == ANSWER


# ── the converters ──────────────────────────────────────────────────────────

HISTORY = [
    {"role": "system", "content": "You are Auto."},
    {"role": "user", "content": "Is ticket 1110 still waiting on anyone?"},
    {"role": "assistant", "content": NARRATION, "tool_calls": [_call("platform_get_activity_feed", "toolu_1")]},
    {"role": "tool", "tool_call_id": "toolu_1", "content": "Ticket #1110: done at 03:04:20."},
    {"role": "system", "content": "Based on the tool results above, answer the owner."},
]


def test_anthropic_gets_the_narration_beside_its_tool_use_and_the_result_after():
    from core.llm.clients.anthropic_client import AnthropicProvider

    system, sent = AnthropicProvider._convert_messages_to_anthropic_format(
        AnthropicProvider.__new__(AnthropicProvider), HISTORY)

    assert system == "You are Auto."                                     # a later system line never replaces it
    assert sent[1] == {"role": "assistant", "content": [
        {"type": "text", "text": NARRATION},
        {"type": "tool_use", "id": "toolu_1", "name": "platform_execute",
         "input": {"action": "platform_get_activity_feed", "params": {}}}]}
    assert sent[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Ticket #1110: done at 03:04:20."}]}
    assert sent[3] == {"role": "user", "content": "Based on the tool results above, answer the owner."}


def test_openrouter_sends_the_narration_with_its_tool_calls_as_they_are():
    from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider

    stub = NS(spec=NS(prompt_cache_control=True, reports_cost=True, web_search_tool=None),
              config=NS(model="google/gemini-2.5-flash", temperature=0.2, max_tokens=1024, top_p=None,
                        frequency_penalty=None, presence_penalty=None, stop=None), _extra_body=None)
    for name in ("_base_kwargs", "_sanitize_tools", "_web_search_server_tool", "_request_kwargs"):
        attr = inspect.getattr_static(OpenAICompatibleProvider, name)
        if isinstance(attr, staticmethod):
            setattr(stub, name, attr.__func__)
        elif isinstance(attr, classmethod):
            setattr(stub, name, MethodType(attr.__func__, OpenAICompatibleProvider))
        else:
            setattr(stub, name, MethodType(attr, stub))

    sent = stub._request_kwargs(HISTORY, TOOLS)["messages"]
    assert sent[2] == HISTORY[2]
