"""PRD-256 FX-017 (night 12, G1/G2): one answer on the screen, never two and never none.

A108, A209, A622: a nudged reply's retraction carried the response's content, but the
screen held what streamed (its ``<think>`` block, the owner-words rewrite made live), so
the frontend found nothing to take out and both replies stayed. Eval M-terms-a turn 2:
the F205 re-prompt came back blank, the original reply stood and was saved, but its
retraction had already gone out and the screen was empty.

Now a frame carries the exact text its round streamed, and a retraction waits for a
replacement with something in it. These tests drive the real tool loop with a scripted
provider streaming through ``in_owner_words``, and apply the frames with the frontend's
rule (frontend/lib/chat/narration.ts ``withoutNarration``, hooks.ts), ported below.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {
    "type": "object", "properties": {"action": {"type": "string"}, "params": {"type": "object"}}}}}]
LOOKING = "Let me look at the mission first."
CLAIM = "I've approved the mission. It's now running."
ANSWER = "I have not approved it: that needs you."
NAMED = "I missed a required parameter for the create_blog_post tool. What title would you like?"
THINKING_CLAIM = ["<think>The owner wants it ", "approved.</think>", "I've approved the mission. ",
                  "It's now running (`platform_", "execute`)."]
STREAMED_CLAIM = "<think>The owner wants it approved.</think>I've approved the mission. It's now running (execute)."


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the loop tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _call(action):
    return {"id": "call_1", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


class _Provider:
    """A scripted streaming provider behind LLMManager's streamed path: each round's text
    deltas go out live through ``in_owner_words``; its response is what the stream
    assembler builds (``<think>`` lifted out of the content)."""

    def __init__(self, *rounds):
        self.rounds, self.tools = list(rounds), []

    async def stream(self, messages, tools, on_delta):
        from core.llm.clients.base import LLMResponse
        from core.llm.reasoning import split_think_tags

        assert self.rounds, "the model was called more often than scripted"
        deltas, calls = self.rounds.pop(0)
        for delta in deltas:
            await on_delta("text", delta)
        content, reasoning = split_think_tags("".join(deltas))
        return LLMResponse(content=content, tool_calls=calls, reasoning=reasoning, streamed=bool(deltas),
                           finish_reason="tool_calls" if calls else "stop")

    async def generate_response(self, messages, tools=None, on_delta=None):
        from core.llm.owner_words_stream import in_owner_words

        self.tools.append(tools)
        return await in_owner_words(self.stream, messages, tools, on_delta)


class _Router:
    def __init__(self, success):
        self.success = success

    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        if self.success:
            return {"success": True, "llm_context": "Mission #41: waiting for your approval.",
                    "raw_result": {"status": "awaiting_approval"}}
        return {"success": False, "llm_context": "Missing required parameter: title",
                "raw_result": {"success": False, "error": "Missing required parameter: title"}}


def _service(success):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router(success)
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


def _turn(provider, *, success=True, chat_lane=False, owner="Approve mission 41 for me."):
    """One turn as the chat runs it: the first call streams, then the loop (frames, final)."""
    from consumers.chatbot.narration import called_tools
    from consumers.chatbot.on_screen import keeps_one_answer_on_screen
    from core.llm.usage_context import LANE_CHAT, usage_scope

    svc = _service(success)
    handler = svc.streaming_handler
    runtime = NS(llm_manager=provider, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": owner}]

    @keeps_one_answer_on_screen
    async def turn():
        live = []

        async def on_delta(kind, text):
            if kind == "text":
                live.append(handler.format_aisdk_text(text))
        first = await provider.generate_response(messages, TOOLS, on_delta=on_delta)
        for frame in live:
            yield frame
        if first.content and called_tools(first):  # F186: it spoke before its tool calls
            yield handler.format_aisdk_narration(first.content)
        rounds = [first] if first.streamed and first.content else []
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS,
                                                 streamed_rounds=rounds, reasoning_log=[]):
            yield chunk

    async def run():
        frames, final = [], None
        scope = usage_scope(request_type=LANE_CHAT, execution_id="chat:fx017") if chat_lane \
            else contextlib.nullcontext()
        with scope:
            async for chunk in turn():
                if isinstance(chunk, dict) and chunk.get("_final_response"):
                    final = chunk["_final_response"]
                else:
                    frames.append(chunk)
        return frames, final
    return asyncio.run(run())


# ── the frontend's rule, ported ─────────────────────────────────────────────

def _without_narration(content, text):
    """frontend/lib/chat/narration.ts ``withoutNarration``: the last exact occurrence goes."""
    if not text:
        return content
    at = content.rfind(text)
    return content if at < 0 else content[:at] + content[at + len(text):]


def _screen_after(frames):
    """frontend/lib/chat/hooks.ts: a text frame is added, a narration frame takes its text out."""
    content = ""
    for frame in frames:
        if frame.startswith("0:"):
            content += json.loads(frame[2:])
        elif frame.startswith("d:"):
            event = json.loads(frame[2:])
            if event.get("type") == "narration":
                content = _without_narration(content, event["data"]["text"])
    return content


def _retractions(frames):
    events = [json.loads(f[2:]) for f in frames if f.startswith("d:")]
    return [e["data"]["text"] for e in events if e.get("type") == "narration" and e["data"].get("retracted")]


# ── the retraction names what was streamed ──────────────────────────────────

def test_a_thinking_rewritten_draft_is_retracted_as_it_streamed():
    """The draft streamed its <think> block and its name said in plain words live; its
    content lost the block. The retraction is the streamed text, so the screen keeps one answer."""
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), (THINKING_CLAIM, None), ([ANSWER], None))
    frames, final = _turn(provider, chat_lane=True)

    assert _retractions(frames) == [STREAMED_CLAIM]
    assert STREAMED_CLAIM != "I've approved the mission. It's now running (execute)."  # the draft's content
    assert final.content == ANSWER
    assert _screen_after(frames) == ANSWER


def test_a_nudged_draft_then_a_real_answer_is_one_retraction_and_one_answer():
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([CLAIM], None), ([ANSWER], None))
    frames, final = _turn(provider)

    assert _retractions(frames) == [CLAIM]
    assert final.content == ANSWER and _screen_after(frames) == ANSWER


# ── a blank retry never replaces an answer ──────────────────────────────────

def test_a_blank_owner_words_retry_sends_no_retraction_and_the_original_is_saved():
    """Eval M-terms-a turn 2: the F205 re-prompt came back blank; the original stands."""
    provider = _Provider(([], [_call("platform_create_blog_post")]), ([NAMED], None), ([], None))
    frames, final = _turn(provider, success=False, owner="Write a blog post about how we pick the club coffee.")

    assert provider.tools[-1] is None                       # the F205 re-prompt ran, with no tools
    assert _retractions(frames) == []
    assert final.content == NAMED
    assert _screen_after(frames) == NAMED                    # the saved answer is the screen


def test_a_blank_retry_to_a_nudge_keeps_the_draft_on_screen():
    """``kept_if_blank``: the nudge got an empty reply, the reply it was about stands."""
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([CLAIM], None), ([], None))
    frames, final = _turn(provider)

    assert _retractions(frames) == []
    assert final.content == CLAIM and _screen_after(frames) == CLAIM


def test_a_held_retraction_goes_out_when_the_answer_replaced_the_draft():
    """Auto's own turn turns a blank reply into the account of its calls (F264): that account
    replaces the draft, so the draft is retracted once, before the answer the turn streams."""
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([CLAIM], None), ([], None))
    frames, final = _turn(provider, chat_lane=True)

    assert _retractions(frames) == [CLAIM]
    assert final.content != CLAIM and not getattr(final, "streamed", False)  # the turn streams it next
    assert _screen_after(frames) == ""


# ── P256-FIX-RVW-24: a draft is retracted once, whatever its retry did ───────

DONE = "Done."
MOVED = "I've moved #0231 to Done."


def test_a_draft_whose_retry_only_calls_tools_is_retracted_once_and_the_answer_stays():
    """The retry made the call with no text, so it never became the latest streamed round:
    the answer after it (the draft's own words) must not retract the draft a second time,
    or the frontend's rule takes the answer off the screen."""
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([CLAIM], None),
                         ([], [_call("platform_update_task_status")]), ([CLAIM], None))
    frames, final = _turn(provider, chat_lane=True)

    assert _retractions(frames) == [CLAIM]
    assert final.content == CLAIM
    assert _screen_after(frames) == CLAIM


def test_an_answer_that_contains_the_draft_stays_whole_after_a_tool_only_retry():
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([DONE], None),
                         ([], [_call("platform_update_task_status")]), ([MOVED], None))
    frames, final = _turn(provider, chat_lane=True, owner="Move ticket 231 to Done.")

    assert _retractions(frames) == [DONE]
    assert final.content == MOVED
    assert _screen_after(frames) == MOVED


# ── the screen on its own ───────────────────────────────────────────────────

def _response(content, calls=None):
    return NS(content=content, tool_calls=calls)


def test_the_screen_holds_a_retraction_until_the_answer_is_known():
    from consumers.chatbot.on_screen import Screen

    screen = Screen()
    original = _response(CLAIM)
    screen.ended(original, "<think>x</think>" + CLAIM)
    screen.ended(_response(""), "")
    assert screen.retraction(CLAIM) is None                                  # held: its retry is blank
    assert screen.settled(original) == []                                    # the original stands

    screen.ended(_response(""), "")
    assert screen.retraction(CLAIM) is None
    assert screen.settled(_response("Nothing changed.")) == ["<think>x</think>" + CLAIM]


def test_a_retraction_sent_later_is_never_sent_twice():
    from consumers.chatbot.on_screen import Screen

    screen = Screen()
    screen.ended(_response(CLAIM), CLAIM)
    screen.ended(_response(""), "")
    assert screen.retraction(CLAIM) is None
    screen.ended(_response(ANSWER), ANSWER)
    assert screen.retraction(CLAIM) == CLAIM
    assert screen.settled(_response(ANSWER)) == []


def test_without_a_screen_the_frame_carries_the_text_it_is_given():
    from consumers.chatbot.streaming import get_streaming_handler

    frame = get_streaming_handler().format_aisdk_narration(CLAIM, retracted=True)
    assert json.loads(frame[2:]) == {"type": "narration", "data": {"text": CLAIM, "retracted": True}}


def test_the_turn_and_its_calls_are_wired_to_the_screen():
    from consumers.chatbot.service import StreamingChatService
    from core.llm.owner_words_stream import in_owner_words

    assert StreamingChatService._stream_response_with_agent_scoped.__code__.co_qualname == \
        "keeps_one_answer_on_screen.<locals>.wrapped"
    assert StreamingChatService._stream_tool_loop.__code__.co_qualname == "the_loop_writes_receipts.<locals>.wrapped"
    assert StreamingChatService._stream_tool_loop.__wrapped__.__code__.co_qualname == \
        "settles_held_retractions.<locals>.wrapped"
    assert in_owner_words.__code__.co_qualname == "watched.<locals>.wrapped"


def test_the_screen_retracts_a_round_once_after_a_tool_only_retry():
    from consumers.chatbot.on_screen import Screen

    screen = Screen()
    screen.ended(_response(CLAIM), CLAIM)
    screen.ended(_response("", [_call("platform_update_task_status")]), "")
    assert screen.retraction(CLAIM) == CLAIM                                 # the retry replaced it
    screen.ended(_response(CLAIM), CLAIM)
    assert screen.retraction(CLAIM) is None                                  # the answer: never again
    assert screen.settled(_response(CLAIM)) == []


def test_a_held_round_settled_is_never_retracted_again():
    from consumers.chatbot.on_screen import Screen

    screen = Screen()
    screen.ended(_response(CLAIM), CLAIM)
    screen.ended(_response(""), "")
    assert screen.retraction(CLAIM) is None
    assert screen.settled(_response(ANSWER)) == [CLAIM]
    screen.ended(_response(ANSWER), ANSWER)
    assert screen.retraction(CLAIM) is None


# ── P256-FIX-RVW-43: a held round is settled by where the answer came from ──

UPDATED = "I've updated the agent and switched its model."
APOLOGY_R = ("My apologies. platform_update_agent was refused: it needs a value I don't have. "
             "Which model should the agent use?")
WITHOUT_APOLOGY_R = APOLOGY_R.removeprefix("My apologies. ")


def test_a_blank_owner_words_retry_after_an_apology_stripped_answer_leaves_it_on_screen():
    """The write is refused, the claim is nudged, the retry opens with an apology and names
    the action; the answer saved is the retry without its apology (a copy), and the F205
    re-prompt comes back blank. The retry's text stays on the screen: never none."""
    provider = _Provider(([LOOKING], [_call("platform_update_agent")]), ([UPDATED], None),
                         ([APOLOGY_R], None), ([], None))
    frames, final = _turn(provider, success=False, owner="Switch OPS to a cheaper model.")

    assert provider.tools[-1] is None                       # the F205 re-prompt ran, with no tools
    assert final.content == WITHOUT_APOLOGY_R and final.streamed
    assert _retractions(frames) == [UPDATED]                 # the nudged claim, and only it
    assert _screen_after(frames) == APOLOGY_R


def _gated(*rounds):
    """The provider behind LLMManager's re-prompt wrapper (core/llm/turn_order.py): a reply
    to a nudge streams through the apology gate, which drops the apology from the stream."""
    from core.llm.turn_order import reprompts_in_the_users_turn

    class _Gated(_Provider):
        generate_response = reprompts_in_the_users_turn(_Provider.generate_response)
    return _Gated(*rounds)


def test_a_gated_apology_never_streamed_and_the_blank_retry_leaves_the_answer_on_screen():
    """Through LLMManager the apology never reaches the screen: the screen's round is the
    copy the gate saved, holding what the gate let through, so the blank F205 re-prompt
    holds it and the answer, that copy, stands."""
    provider = _gated(([LOOKING], [_call("platform_update_agent")]), ([UPDATED], None),
                      ([APOLOGY_R], None), ([], None))
    frames, final = _turn(provider, success=False, owner="Switch OPS to a cheaper model.")

    assert provider.tools[-1] is None
    assert final.content == WITHOUT_APOLOGY_R
    assert _retractions(frames) == [UPDATED]
    assert _screen_after(frames) == WITHOUT_APOLOGY_R


def test_the_gate_reports_what_it_let_through_to_the_screen():
    from consumers.chatbot.on_screen import Screen
    from core.llm.reprompt_reply import gated

    shown = []

    async def on_delta(kind, text):
        shown.append(text)

    async def run():
        gate = gated(on_delta)
        for delta in ("My apologies. platform_update_agent ", "was refused."):
            await gate("text", delta)
        await gate.close()
        return gate.passed
    assert asyncio.run(run()) == "".join(shown) == "platform_update_agent was refused."

    screen, full = Screen(), _llm(APOLOGY_R)
    screen.ended(full, APOLOGY_R)
    copy = _llm(WITHOUT_APOLOGY_R)
    screen.restated(full, copy, WITHOUT_APOLOGY_R)
    assert screen.narration(WITHOUT_APOLOGY_R) == WITHOUT_APOLOGY_R
    screen.ended(_llm(""), "")
    assert screen.retraction(WITHOUT_APOLOGY_R) is None                      # held, as the copy
    assert screen.settled(copy) == []


def _llm(content, streamed=True):
    from core.llm.clients.base import LLMResponse

    return LLMResponse(content=content, streamed=streamed, finish_reason="stop")


def test_a_held_round_stands_for_its_own_response_or_a_streamed_copy_of_it():
    from consumers.chatbot.on_screen import Screen
    from modules.tools.execution.nudges import without_the_apology

    for answer_of in (lambda r: r, without_the_apology):
        screen, retry = Screen(), _llm(APOLOGY_R.replace(". ", ".\n\n", 1))
        screen.ended(retry, retry.content)
        screen.ended(_llm(""), "")
        assert screen.retraction(retry.content) is None                      # held: the re-prompt is blank
        assert screen.settled(answer_of(retry)) == []                        # it is the answer: it stands


def test_a_held_round_is_retracted_when_the_answer_only_reads_like_it():
    """Equal text is not where an answer came from: an account the turn writes (not
    streamed), or a streamed answer that is not the round's own last words, replaces it."""
    from consumers.chatbot.on_screen import Screen

    for answer in (_llm(CLAIM, streamed=False), _llm("now running."), _llm("   ")):
        screen = Screen()
        screen.ended(_llm(CLAIM), CLAIM)
        screen.ended(_llm(""), "")
        assert screen.retraction(CLAIM) is None
        assert screen.settled(answer) == [CLAIM]
