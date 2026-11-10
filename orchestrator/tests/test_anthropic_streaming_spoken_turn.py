"""Anthropic streaming (voice latency B-3): a whole spoken turn whose model calls stream.

With ``ANTHROPIC_STREAMING=quick-ask`` a quick ask's Anthropic calls hand their text over as it
is generated. The turn then does with those rounds what it does with any streamed round: the
words before a tool call are narration, a reply whose claim no call backs is taken off and the
corrected copy sent in its place. Before this an Anthropic turn showed only its final text.

These cases drive the real tool loop (``StreamingChatService._stream_tool_loop``) the way
``tests/test_prd256_fix_one_answer_on_screen.py`` does, but the model is a real
``AnthropicProvider`` on the real SDK client over recorded server-sent events, inside a real
``LLMManager``, on a turn marked ``quick-ask``. The turn's frames then pass through
``spoken_text_frames``, as ``POST /api/chat`` with ``spokenMode`` passes them, and are applied
with the client's rule (``withoutNarration``, ported below).

* A spoken tool turn is narration, the narration frame, the tool frames, then the answer, in
  that order, with no text sent twice.
* A spoken reply with an unbacked claim is the draft, one retraction, then the corrected copy;
  taking the retraction's text out of what was sent leaves only the corrected copy.

The texts, tools and tool results are the ones ``tests/test_f186_a_the_reply_is_the_answer.py``
and ``tests/test_prd256_fix_one_answer_on_screen.py`` run through the same loop.

No database, no model and no socket: the SDK reads recorded bytes from a mock transport.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from config import config
from core.llm.request_scope import QUICK_ASK, set_spoken_turn
from tests.helpers_anthropic_stream import manager_on, recorded_round, sdk_provider_in_rounds

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
TOOL = "platform_execute"
TOOLS = [{"type": "function", "function": {"name": TOOL, "parameters": {
    "type": "object", "properties": {"action": {"type": "string"}, "params": {"type": "object"}}}}}]
CALL_ID = "toolu_1"
UNIT_SEPARATOR = " "

# F186's turn: the model says what it sees, reads the activity, then answers.
ASKED_ABOUT_A_TICKET = "Is ticket 1110 still waiting on anyone?"
NARRATION_PIECES = ("I see ticket #1110 on your board, waiting for the Shopify Support Agent ",
                    "and for your review. I will check the current activity for you.")
NARRATION = "".join(NARRATION_PIECES)
ANSWER_PIECES = ("You got it, Gerard! ", "Ticket #1110 was completed at 03:04.")
ANSWER = "".join(ANSWER_PIECES)
THE_TICKET_IS_DONE = {"success": True, "llm_context": "Ticket #1110: done at 03:04:20.",
                      "raw_result": {"status": "done"}}

# FX-017's turn: the model reads the mission, then says it approved it. No call did.
ASKED_TO_APPROVE = "Approve mission 41 for me."
LOOKING = "Let me look at the mission first."
CLAIM_PIECES = ("I've approved the mission. ", "It's now running.")
CLAIM = "".join(CLAIM_PIECES)
CORRECTED = "Just to be clear: I haven't approved anything in this reply. Ask me again if you want it done."
THE_MISSION_WAITS = {"success": True, "llm_context": "Mission #41: waiting for your approval.",
                     "raw_result": {"status": "awaiting_approval"}}

ORDERED_KINDS = ("text", "narration", "tool-start", "tool-end")


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the loop tests pin them."""
    monkeypatch.setattr(type(config), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(config), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(config), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _reads(action):
    """The round's one tool call: a read through the generic platform tool."""
    return (CALL_ID, TOOL, {"action": action, "params": {}})


class _Router:
    def __init__(self, result):
        self.result = result

    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        return dict(self.result)


def _service(result):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router(result)
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


async def _first_call(manager, handler, messages):
    """The turn's first model call: (its reply, the text frames it streamed)."""
    live = []

    async def on_delta(kind, text):
        if kind == "text":
            live.append(handler.format_aisdk_text(text))

    return await manager.generate_response(messages, TOOLS, on_delta=on_delta), live


async def _unstreamed_tail(handler, final):
    """``service.py``: an answer the screen has not seen goes out through the typewriter."""
    if not getattr(final, "streamed", False):
        async for chunk in handler.stream_text_aisdk(final.content):
            yield chunk


async def _turn_frames(*, svc, manager, messages, finals):
    """The turn's frames as the chat sends them: the first call's, then the loop's, then the
    unstreamed answer's. The loop's answer is kept in ``finals``."""
    from consumers.chatbot.narration import called_tools

    handler = svc.streaming_handler
    runtime = NS(llm_manager=manager, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    first, live = await _first_call(manager, handler, messages)
    for frame in live:
        yield frame
    if first.content and called_tools(first):  # F186: it spoke before its tool calls
        yield handler.format_aisdk_narration(first.content)
    rounds = [first] if first.streamed and first.content else []
    async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS,
                                             streamed_rounds=rounds, reasoning_log=[]):
        if isinstance(chunk, dict) and chunk.get("_final_response"):
            finals.append(chunk["_final_response"])
        else:
            yield chunk
    for final in finals[-1:]:
        async for chunk in _unstreamed_tail(handler, final):
            yield chunk


def _spoken_turn(monkeypatch, provider, *, owner, result):
    """One quick ask as the chat runs it, on its own Screen (``keeps_one_answer_on_screen``, called
    with keywords only, as the harness this follows runs it), with every frame passed through
    ``spoken_text_frames``. Returns (the frames the phone is sent, the loop's answer)."""
    from consumers.chatbot.on_screen import keeps_one_answer_on_screen
    from modules.voice.spoken_stream import spoken_text_frames

    monkeypatch.setattr(config, "ANTHROPIC_STREAMING", "quick-ask")
    svc, finals = _service(result), []
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": owner}]
    turn = keeps_one_answer_on_screen(_turn_frames)(
        svc=svc, manager=manager_on(provider, monkeypatch), messages=messages, finals=finals)

    async def run():
        set_spoken_turn(QUICK_ASK)
        return [frame async for frame in spoken_text_frames(turn, svc.streaming_handler.format_aisdk_text)]

    sent = asyncio.run(run())
    assert len(finals) == 1, "the loop answers once"
    return sent, finals[0]


# ── reading the frames, and the client's rule, ported ───────────────────────

def _event(frame):
    return json.loads(frame[2:]) if frame.startswith("d:") else {}


def _kind(frame):
    return "text" if frame.startswith("0:") else _event(frame).get("type")


def _order(frames):
    """The kinds of the turn's text, narration and tool frames in the order sent, a run of one
    kind said once."""
    kinds = [kind for kind in map(_kind, frames) if kind in ORDERED_KINDS]
    return [kind for at, kind in enumerate(kinds) if at == 0 or kinds[at - 1] != kind]


def _texts(frames):
    return [json.loads(frame[2:]) for frame in frames if frame.startswith("0:")]


def _narrations(frames, *, retracted):
    events = [_event(frame) for frame in frames]
    return [event["data"]["text"] for event in events
            if event.get("type") == "narration" and bool(event["data"].get("retracted")) is retracted]


def _without_narration(content, text):
    """frontend/lib/chat/narration.ts ``withoutNarration``: the last exact occurrence goes."""
    if not text:
        return content
    at = content.rfind(text)
    return content if at < 0 else content[:at] + content[at + len(text):]


def _heard_after(frames):
    """What the client holds once every frame is applied: a text frame is added, a narration
    frame takes its text out (the reducer Go and the web quick ask share)."""
    content = ""
    for frame in frames:
        if frame.startswith("0:"):
            content += json.loads(frame[2:])
        elif _event(frame).get("type") == "narration":
            content = _without_narration(content, _event(frame)["data"]["text"])
    return content


def _every_round_streamed(provider, rounds):
    """The turn asked the model exactly ``rounds`` times, each time for a stream."""
    assert provider.unasked == []
    assert [body.get("stream") for body in provider.requests] == [True] * rounds


# ── the turn ────────────────────────────────────────────────────────────────

def test_a_spoken_tool_turn_is_narration_then_tools_then_the_answer(monkeypatch):
    provider = sdk_provider_in_rounds(
        recorded_round(*NARRATION_PIECES, tool=_reads("platform_get_activity_feed")),
        recorded_round(*ANSWER_PIECES))

    sent, final = _spoken_turn(monkeypatch, provider, owner=ASKED_ABOUT_A_TICKET, result=THE_TICKET_IS_DONE)

    _every_round_streamed(provider, 2)
    assert _order(sent) == ["text", "narration", "tool-start", "tool-end", "text"]
    texts = _texts(sent)
    assert "".join(texts) == NARRATION + UNIT_SEPARATOR + ANSWER
    assert len(set(texts)) == len(texts)                              # no text frame twice
    assert _narrations(sent, retracted=False) == [NARRATION]          # the text the phone was sent
    assert _narrations(sent, retracted=True) == []
    assert (final.content, final.streamed) == (ANSWER, True)          # it streamed: no typewriter copy
    assert _heard_after(sent).strip() == ANSWER


def test_a_spoken_reply_with_an_unbacked_claim_is_retracted_and_corrected(monkeypatch):
    provider = sdk_provider_in_rounds(
        recorded_round(LOOKING, tool=_reads("platform_get_mission")),
        recorded_round(*CLAIM_PIECES))

    sent, final = _spoken_turn(monkeypatch, provider, owner=ASKED_TO_APPROVE, result=THE_MISSION_WAITS)

    _every_round_streamed(provider, 2)                                # the claim is taken out, not asked again
    assert _order(sent) == ["text", "narration", "tool-start", "tool-end", "text", "narration", "text"]
    texts = _texts(sent)
    assert "".join(texts) == UNIT_SEPARATOR.join((LOOKING, CLAIM, CORRECTED))
    assert len(set(texts)) == len(texts)
    assert _narrations(sent, retracted=False) == [LOOKING]
    assert _narrations(sent, retracted=True) == [CLAIM]               # one retraction: the draft as sent
    assert final.content == CORRECTED and not final.streamed          # the copy never streamed
    assert _heard_after(sent).strip() == CORRECTED                    # only the corrected copy is left
