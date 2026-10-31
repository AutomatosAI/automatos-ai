"""PRD-268 US-C12c (§6.3 'In the thread', §6.6 'Storage', K12, K13): the turn's parts.

The reply a chat turn saves carries its reply chips (``reply_chips@1``) and the follow-ups it
scheduled (``follow_up@1``) after its text part. They go out live on one ``tool-data`` frame under
``turn_parts``, beside the receipts frames, before the finish. When the chips end the run on a
round's own text (US-C12b), that text is on the screen once, sent once and saved once: no narration
part or frame repeats it.

The turn here is driven through the receipts decorators the chat uses (``writes_its_receipts``
outside the turn's own Screen), over the real tool loop (``StreamingChatService._stream_tool_loop``)
and a scripted provider streaming through ``in_owner_words``. The first reply and the tail are
service.py's own steps, pinned below (service.py is untouched by this story). The frames are applied
with the frontend's rule (frontend/lib/chat/hooks.ts, narration.ts ``withoutNarration``), ported.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {
    "type": "object", "properties": {"action": {"type": "string"}, "params": {"type": "object"}}}}}]
QUESTION = "Shall I book the Lisbon trip for Friday?"
CHIPS = ["Book it", "Not yet"]
LOOKING = "Let me look at the mission first."
AT = "2026-10-30T15:10:00+00:00"
FOLLOW_UP_ID = 41
ANSWERS = {
    "platform_offer_replies": {"success": True, "replies": CHIPS},
    "platform_schedule_follow_up": {"success": True, "follow_up_id": FOLLOW_UP_ID, "at": AT,
                                    "message": "Follow-up scheduled."},
    "platform_get_brief": {"success": True, "sections": []},
    "platform_get_mission": {"success": True, "status": "running"},
}
REFUSED_OFFER = {"success": False, "error": "Offer 2 or 3 replies.", "error_code": "reply_count"}


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the loop tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _call(action, params=None, n=1):
    return {"id": f"call_{n}", "type": "function", "function": {
        "name": "platform_execute", "arguments": json.dumps({"action": action, "params": params or {}})}}


OFFER = _call("platform_offer_replies", {"replies": CHIPS}, n=2)
SCHEDULE = _call("platform_schedule_follow_up", {"reason": "Check whether the Lisbon card moved", "in_minutes": 15})


class _Provider:
    """A scripted streaming provider: each round's text streams live through ``in_owner_words``
    (so the turn's Screen records it), and every call is counted."""

    def __init__(self, *rounds):
        self.rounds, self.calls = list(rounds), 0

    async def stream(self, messages, tools, on_delta):
        from core.llm.clients.base import LLMResponse

        assert self.rounds, "the model was called more often than scripted"
        deltas, calls = self.rounds.pop(0)
        for delta in deltas:
            await on_delta("text", delta)
        text = "".join(deltas)
        return LLMResponse(content=text, tool_calls=calls, streamed=bool(text), model="google/gemini-2.5-flash",
                           finish_reason="tool_calls" if calls else "stop")

    async def generate_response(self, messages, tools=None, on_delta=None):
        from core.llm.owner_words_stream import in_owner_words

        async def nowhere(kind, text):
            return None
        self.calls += 1
        return await in_owner_words(self.stream, messages, tools, on_delta or nowhere)


class _Router:
    def __init__(self, **answers):
        self.answers = {**ANSWERS, **answers}

    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        raw = self.answers[(tool_args or {}).get("action")]
        return {"success": raw.get("success") is True, "llm_context": json.dumps(raw), "raw_result": raw}


def _service(router, widget):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, widget
    svc.widget_scopes, svc.widget_team, svc.widget_agent_lock = (), None, None
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), router
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


async def _loop_then_tail(svc, provider, rounds, first, messages):
    """The loop, then service.py's tail: the answer, a tail only when it did not stream, the finish."""
    from consumers.chatbot.narration import called_tools

    runtime = NS(llm_manager=provider, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    final = first
    if called_tools(first):
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, streamed_rounds=rounds,
                                                 reasoning_log=[], prefetched=[]):
            if isinstance(chunk, dict) and chunk.get("_final_response"):
                final = chunk["_final_response"]
            else:
                yield chunk
    if not getattr(final, "streamed", False):
        async for chunk in svc.streaming_handler.stream_text_aisdk(final.content):
            yield chunk
    yield {"_answered": final}


async def _run_turn(provider, router=None, *, widget=False):
    """One chat turn as service.py runs it, through the receipts decorators. Returns its frames and
    the parts it saved."""
    from consumers.chatbot.narration import called_tools, reply_parts, split_reply
    from consumers.chatbot.on_screen import keeps_one_answer_on_screen
    from consumers.chatbot.receipts import writes_its_receipts

    svc = _service(router or _Router(), widget)
    handler, saved = svc.streaming_handler, []
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": "Book Lisbon."}]

    @keeps_one_answer_on_screen
    async def scoped(chat):
        live = []

        async def on_delta(kind, text):
            live.append(handler.format_aisdk_text(text))
        first = await provider.generate_response(messages, TOOLS, on_delta=on_delta)
        for frame in live:
            yield frame
        rounds = [first] if first.streamed and first.content else []
        if rounds and called_tools(first):          # service.py: it spoke before its tool calls
            yield handler.format_aisdk_narration(first.content)
        final = first
        async for chunk in _loop_then_tail(svc, provider, rounds, first, messages):
            if isinstance(chunk, dict):
                final = chunk["_answered"]
            else:
                yield chunk
        narration, answer = split_reply(rounds, final, final.content)
        yield handler.format_aisdk_finish()
        saved.append(reply_parts("", "\n\n".join(narration), answer))

    @writes_its_receipts
    async def turn(chat, **kwargs):
        async for chunk in scoped(chat):
            yield chunk

    frames = [chunk async for chunk in turn(svc, spoken_mode=False)]
    return frames, saved[0]


def _turn(provider, router=None, *, widget=False):
    return asyncio.run(_run_turn(provider, router, widget=widget))


# ── the frontend's rule, ported ─────────────────────────────────────────────

def _screen_after(frames):
    """hooks.ts: a text frame is added; a narration frame takes the last exact occurrence of its
    text out of the answer and, unless retracted, adds it to the activity trail."""
    content, trail = "", []
    for frame in frames:
        if frame.startswith("0:"):
            content += json.loads(frame[2:])
        elif frame.startswith("d:"):
            event = json.loads(frame[2:])
            if event.get("type") == "narration":
                said = event["data"]["text"]
                at = content.rfind(said)
                content = content if at < 0 else content[:at] + content[at + len(said):]
                trail = trail if event["data"].get("retracted") else [*trail, said]
    return content, trail


def _live_parts(frames):
    events = [json.loads(f[2:]) for f in frames if f.startswith('d:{"type":"tool-data"')]
    return [e["data"]["turn_parts"] for e in events if "turn_parts" in e["data"]]


def _text_sent(frames):
    return "".join(json.loads(f[2:]) for f in frames if f.startswith("0:"))


def _types(parts):
    return [part["type"] for part in parts]


def _at(frames, needle):
    return next(i for i, frame in enumerate(frames) if needle in frame)


def _narrated(frames):
    events = [json.loads(f[2:]) for f in frames if f.startswith('d:{"type": "narration"')]
    return [e["data"]["text"] for e in events]


# ── a chips turn ────────────────────────────────────────────────────────────

def test_a_chips_turn_saves_receipts_text_and_chips_and_sends_one_frame_before_the_finish():
    provider = _Provider(([QUESTION], [OFFER]))
    frames, parts = _turn(provider)

    assert _types(parts) == ["receipts", "text", "reply_chips@1"]
    assert parts[1:] == [{"type": "text", "text": QUESTION}, {"type": "reply_chips@1", "replies": CHIPS}]
    assert _live_parts(frames) == [[{"type": "reply_chips@1", "replies": CHIPS}]]   # one frame
    assert _at(frames, '"turn_parts"') > _at(frames, '"type": "receipts"')         # beside the receipts
    assert _at(frames, '"turn_parts"') < _at(frames, '"type":"finish"')
    assert provider.calls == 1                                                      # no call after the chips


def test_the_round_the_chips_end_on_is_on_the_screen_sent_and_saved_once():
    frames, parts = _turn(_Provider(([QUESTION], [OFFER])))

    assert _screen_after(frames) == (QUESTION, [])         # the answer, and no trail line repeating it
    assert _text_sent(frames) == QUESTION                  # the wire carries it once
    assert "narration" not in _types(parts)
    assert sum(json.dumps(part).count(QUESTION) for part in parts) == 1


def test_chips_in_a_later_round_end_on_that_round_and_the_earlier_one_stays_narration():
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), ([QUESTION], [OFFER]))
    frames, parts = _turn(provider)

    assert _screen_after(frames) == (QUESTION, [LOOKING])
    assert _text_sent(frames) == LOOKING + QUESTION
    assert _types(parts) == ["receipts", "narration", "text", "reply_chips@1"]
    assert parts[1]["narration"] == LOOKING and parts[2]["text"] == QUESTION
    assert provider.calls == 2


def test_a_refused_offer_leaves_its_round_narration_as_before():
    provider = _Provider(([QUESTION], [OFFER]), (["Which day suits you?"], None))
    frames, parts = _turn(provider, _Router(platform_offer_replies=REFUSED_OFFER))

    content, trail = _screen_after(frames)
    assert content.endswith("Which day suits you?") and QUESTION not in content and trail == [QUESTION]
    assert _at(frames, '"type": "narration"') < _at(frames, '0:"Which day')      # it leaves before the next text
    assert "reply_chips@1" not in _types(parts) and _live_parts(frames) == []
    assert [part["narration"] for part in parts if part["type"] == "narration"] == [QUESTION]
    assert parts[-1]["type"] == "text" and parts[-1]["text"].endswith("Which day suits you?")


def test_a_turn_that_called_the_brief_saves_no_chips():
    brief = "Here is your brief."
    provider = _Provider(([brief], [_call("platform_get_brief"), OFFER]), (["That is everything on it."], None))
    frames, parts = _turn(provider)

    assert _types(parts) == ["receipts", "narration", "text"]
    assert _live_parts(frames) == [] and provider.calls == 2
    assert _screen_after(frames) == ("That is everything on it.", [brief])
    assert _narrated(frames) == [brief] and _at(frames, '"type": "narration"') < _at(frames, '0:"That is')


# ── a follow-up ─────────────────────────────────────────────────────────────

def test_a_scheduling_turn_saves_its_follow_up_after_the_text():
    provider = _Provider(("", [SCHEDULE]), (["You'll hear from me about the Lisbon card."], None))
    frames, parts = _turn(provider)

    line = {"type": "follow_up@1", "follow_up_id": FOLLOW_UP_ID, "at": AT}
    assert _types(parts) == ["receipts", "text", "follow_up@1"] and parts[-1] == line
    assert _live_parts(frames) == [[line]]
    assert "Check whether" not in json.dumps(parts)           # the reason never rides the part


def test_chips_and_a_follow_up_in_one_round_ride_together():
    frames, parts = _turn(_Provider(([QUESTION], [SCHEDULE, OFFER])))

    assert _types(parts) == ["receipts", "text", "reply_chips@1", "follow_up@1"]
    assert _screen_after(frames) == (QUESTION, [])


def test_a_refused_schedule_saves_no_follow_up():
    refused = {"success": False, "error": "Too many follow-ups.", "error_code": "follow_up_cap"}
    provider = _Provider(("", [SCHEDULE]), (["I could not set that up."], None))
    frames, parts = _turn(provider, _Router(platform_schedule_follow_up=refused))

    assert "follow_up@1" not in _types(parts) and _live_parts(frames) == []


# ── turns with neither, a spoken turn, a widget visitor ─────────────────────

def test_a_plain_turn_saves_exactly_todays_parts_and_sends_no_frame():
    frames, parts = _turn(_Provider((["Hello Gerard."], None)))

    assert parts == [{"type": "receipts", "receipts": []}, {"type": "text", "text": "Hello Gerard."}]
    assert _live_parts(frames) == []


def test_a_tool_turn_without_chips_or_a_follow_up_is_as_before():
    provider = _Provider(([LOOKING], [_call("platform_get_mission")]), (["It is running."], None))
    frames, parts = _turn(provider)

    assert _types(parts) == ["receipts", "narration", "text"] and _live_parts(frames) == []
    assert _screen_after(frames) == ("It is running.", [LOOKING])


def test_a_spoken_turn_saves_its_chips_and_never_speaks_them():
    from consumers.chatbot.streaming import get_streaming_handler
    from modules.voice.spoken_stream import spoken_text_frames

    frames, parts = _turn(_Provider(([QUESTION], [OFFER])))

    async def source():
        for frame in frames:
            yield frame

    async def heard():
        return [f async for f in spoken_text_frames(source(), get_streaming_handler().format_aisdk_text)]
    spoken = _text_sent(asyncio.run(heard()))
    assert spoken and all(chip not in spoken for chip in CHIPS)
    assert parts[1] == {"type": "text", "text": QUESTION} and parts[2]["replies"] == CHIPS


def test_the_next_turn_starts_with_no_parts():
    async def two_turns():
        await _run_turn(_Provider(([QUESTION], [SCHEDULE, OFFER])))
        return await _run_turn(_Provider((["Hello again."], None)))
    frames, parts = asyncio.run(two_turns())

    assert parts == [{"type": "receipts", "receipts": []}, {"type": "text", "text": "Hello again."}]
    assert _live_parts(frames) == []


def test_a_widget_visitors_turn_sends_no_frame():
    frames, parts = _turn(_Provider(([QUESTION], [OFFER])), widget=True)

    assert _live_parts(frames) == [] and not any('"receipts"' in frame for frame in frames)   # F155
    assert _types(parts) == ["receipts", "text", "reply_chips@1"]


# ── the parts, read back ────────────────────────────────────────────────────

def test_history_returns_the_parts_unchanged(monkeypatch):
    import api.chat as chat_api

    _frames, parts = _turn(_Provider(([QUESTION], [SCHEDULE, OFFER])))
    message = NS(id="7c1f", role="assistant", parts=parts, attachments=None, source=None,
                 created_at=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc))

    class _Chats:
        def __init__(self, db):
            pass

        def get_chat(self, chat_id, workspace_id=None):
            return NS(user_id=7)

        def get_messages_by_chat_id(self, chat_id):
            return [message]
    monkeypatch.setattr(chat_api, "ChatService", _Chats)
    monkeypatch.setattr(chat_api, "get_user_id", lambda db, ctx: 7)

    (got,) = chat_api.get_chat_messages("chat-1", ctx=NS(workspace_id=WS), db=None)
    assert got["parts"] == parts and got["source"] is None


def test_the_parts_are_never_read_as_the_reply():
    from api.chat import _parts_text
    from modules.context.sections.conversation import _parts_to_text

    parts = [{"type": "text", "text": QUESTION}, {"type": "reply_chips@1", "replies": CHIPS},
             {"type": "follow_up@1", "follow_up_id": FOLLOW_UP_ID, "at": AT}]
    assert _parts_to_text(parts) == QUESTION and _parts_text(parts) == QUESTION


# ── the builders ────────────────────────────────────────────────────────────

def _tracker(*outcomes):
    return NS(outcomes=list(outcomes))


def test_turn_parts_reads_chips_then_each_follow_up_that_succeeded():
    from consumers.chatbot.turn_parts import turn_parts

    scheduled = {"success": True, "follow_up_id": 9, "at": AT}
    tracker = _tracker(("platform_schedule_follow_up", {}, {"success": True, "raw_result": scheduled}),
                       (" platform_schedule_follow_up", {}, {**scheduled, "follow_up_id": 10}),
                       ("platform_schedule_follow_up", {}, {"success": False, "error": "cap"}),
                       ("platform_list_follow_ups", {}, {"success": True, "follow_ups": []}))
    parts = turn_parts(NS(content=QUESTION, reply_chips=CHIPS), tracker)
    assert parts == ({"type": "reply_chips@1", "replies": CHIPS},
                     {"type": "follow_up@1", "follow_up_id": 9, "at": AT},
                     {"type": "follow_up@1", "follow_up_id": 10, "at": AT})
    assert turn_parts(NS(content="Hi.", reply_chips=None), None) == ()


def test_a_follow_up_cancelled_in_the_same_turn_or_named_twice_is_one_line_or_none():
    from consumers.chatbot.turn_parts import turn_parts

    def scheduled(row):
        return ("platform_schedule_follow_up", {}, {"success": True, "follow_up_id": row, "at": AT})
    cancelled = ("platform_cancel_follow_up", {}, {"success": True, "raw_result": {
        "success": True, "follow_up_id": 9, "status": "cancelled"}})
    refused = ("platform_cancel_follow_up", {}, {"success": False, "raw_result": {"success": False, "follow_up_id": 10}})
    parts = turn_parts(NS(content="Done.", reply_chips=None),
                       _tracker(scheduled(9), scheduled(10), scheduled(10), cancelled, refused))
    assert parts == ({"type": "follow_up@1", "follow_up_id": 10, "at": AT},)


def test_a_follow_ups_time_is_saved_in_utc():
    from consumers.chatbot.turn_parts import follow_up_part

    part = follow_up_part("platform_schedule_follow_up", {"success": True, "follow_up_id": 3,
                                                          "at": "2026-10-30T17:10:00+02:00"})
    assert part["at"] == AT


@pytest.mark.parametrize("follow_up_id, at", [(True, AT), ("41", AT), (0, AT), (41, "2026-10-30T15:10:00"),
                                              (41, "at ten past three"), (41, None)])
def test_a_follow_up_answer_without_a_row_id_and_a_time_gives_no_part(follow_up_id, at):
    from consumers.chatbot.turn_parts import follow_up_part

    assert follow_up_part("platform_schedule_follow_up", {"success": True, "follow_up_id": follow_up_id, "at": at}) is None


@pytest.mark.parametrize("chips", [["one", "two", "three", "four"], ["Yes", "yes"], ["a link: https://x.io", "No"],
                                   "Yes, No"])
def test_chips_that_fail_the_tools_check_are_not_saved(chips):
    from consumers.chatbot.turn_parts import chips_part

    assert chips_part(NS(content=QUESTION, reply_chips=chips), []) is None


@pytest.mark.parametrize("brief", ["platform_get_brief", " platform_get_brief", "platform_get_brief\n"])
def test_a_brief_turn_has_no_chips_part(brief):
    from consumers.chatbot.turn_parts import chips_part

    assert chips_part(NS(content=QUESTION, reply_chips=CHIPS), [(brief, {}, {"success": True})]) is None


def test_a_round_is_the_answer_only_when_the_chips_ended_the_run_on_its_text():
    from consumers.chatbot.narration import split_reply

    offered = NS(content=QUESTION, tool_calls=[OFFER])
    assert split_reply([offered], NS(content=QUESTION, reply_chips=CHIPS), QUESTION) == ([], QUESTION)
    assert split_reply([offered], NS(content=QUESTION, reply_chips=None), QUESTION) == ([QUESTION], QUESTION)
    assert split_reply([offered], NS(content="Other.", reply_chips=CHIPS), "Other.") == ([QUESTION], "Other.")
    spaced = QUESTION.replace(" ", "  ") + "\n"
    assert split_reply([offered], NS(content=spaced, reply_chips=CHIPS), spaced) == ([], spaced)  # spacing aside


def test_which_rounds_offer_replies():
    from consumers.chatbot.turn_parts import offers_replies

    direct = {"id": "c", "type": "function", "function": {"name": "platform_offer_replies", "arguments": "{}"}}
    as_dict = {"id": "c", "type": "function", "function": {"name": "platform_execute",
                                                           "arguments": {"action": " platform_offer_replies"}}}
    assert offers_replies(NS(tool_calls=[direct])) and offers_replies(NS(tool_calls=[as_dict]))
    assert offers_replies(NS(tool_calls=[_call("platform_get_mission"), OFFER]))
    assert not offers_replies(NS(tool_calls=[_call("platform_get_mission")])) and not offers_replies(NS(tool_calls=None))


def test_the_screen_holds_a_chips_rounds_narration_until_the_answer_is_known():
    from consumers.chatbot.on_screen import Screen

    offered = NS(content=QUESTION, tool_calls=[OFFER])
    screen = Screen()
    screen.ended(offered, QUESTION)
    assert screen.narration(QUESTION) == QUESTION                                  # its frame is made
    assert screen.offered_released() == []                                         # held only once it passes
    assert screen.hold(QUESTION) and not screen.hold(QUESTION) and not screen.hold(LOOKING)
    sent, its_round = screen.offered_settled(NS(content=QUESTION, reply_chips=CHIPS))
    assert sent == [] and its_round.response is offered                            # it is the answer
    screen.ended(offered, QUESTION)
    screen.narration(QUESTION)
    assert screen.hold(QUESTION)
    assert screen.offered_settled(NS(content="Other.", reply_chips=None)) == ([QUESTION], None)


def test_a_held_round_that_was_retracted_is_never_sent_again():
    """N15-1: a round answered by a corrected copy is retracted once; its held narration is not sent too."""
    from consumers.chatbot.on_screen import Screen

    offered, screen = NS(content=QUESTION, tool_calls=[OFFER]), Screen()
    screen.ended(offered, QUESTION)
    screen.narration(QUESTION)
    screen.hold(QUESTION)
    copy = NS(content="Shall I book it?", reply_chips=None, streamed=False)
    screen.corrected(offered, copy)
    assert screen.settled(copy) == [QUESTION]                                   # the one retraction
    assert screen.offered_settled(copy) == ([], None)


def test_a_loop_that_ends_with_no_answer_holds_nothing_back():
    from consumers.chatbot.on_screen import Screen, settles_held_retractions
    from consumers.chatbot.streaming import get_streaming_handler
    from core.llm.screen_watch import WATCHER

    chat, screen = NS(streaming_handler=get_streaming_handler()), Screen()
    screen.ended(NS(content=QUESTION, tool_calls=[OFFER]), QUESTION)

    @settles_held_retractions
    async def loop(chat):
        yield chat.streaming_handler.format_aisdk_narration(QUESTION)              # held as it passes
        yield chat.streaming_handler.format_aisdk_tool_data({"progress": 1})

    async def run():
        WATCHER.set(screen)
        return [frame async for frame in loop(chat)]
    frames = asyncio.run(run())
    assert len(frames) == 2 and _narrated(frames[1:]) == [QUESTION]


def test_an_answer_that_cannot_be_marked_streamed_leaves_its_round_narration():
    from consumers.chatbot.on_screen import Screen, _answer_known
    from consumers.chatbot.streaming import get_streaming_handler
    from consumers.chatbot.turn_parts import as_streamed

    class Frozen:
        __slots__ = ("content", "reply_chips")

        def __init__(self):
            self.content, self.reply_chips = QUESTION, CHIPS
    frozen, screen = Frozen(), Screen()
    screen.ended(NS(content=QUESTION, tool_calls=[OFFER]), QUESTION)
    screen.narration(QUESTION)
    screen.hold(QUESTION)
    frames, chunk = _answer_known(get_streaming_handler(), screen, {"_final_response": frozen}, frozen)
    assert as_streamed(frozen) is frozen and chunk["_final_response"] is frozen
    assert [json.loads(f[2:])["data"]["text"] for f in frames] == [QUESTION]


def test_as_streamed_marks_a_copy():
    from core.llm.clients.base import LLMResponse
    from consumers.chatbot.turn_parts import as_streamed

    for response in (LLMResponse(content=QUESTION, reply_chips=CHIPS), NS(content=QUESTION, reply_chips=CHIPS)):
        marked = as_streamed(response)
        assert marked.streamed is True and marked.reply_chips == CHIPS and marked is not response
        assert not getattr(response, "streamed", False)


# ── the seams ───────────────────────────────────────────────────────────────

def test_the_names_are_the_tools():
    from consumers.chatbot import turn_parts
    from modules.tools.discovery import follow_up_context

    assert turn_parts.SCHEDULE_FOLLOW_UP == follow_up_context.SCHEDULE_FOLLOW_UP
    assert turn_parts.CANCEL_FOLLOW_UP == follow_up_context.CANCEL_FOLLOW_UP


def test_service_py_takes_the_steps_this_turn_drives():
    """The harness above takes the chat's own steps; service.py is untouched by this story."""
    from consumers.chatbot import service

    source = inspect.getsource(service)
    for step in ('final_streamed = bool(getattr(final_response, "streamed", False))',
                 'unstreamed_tail = "" if final_streamed else final_text',
                 "narration, full_response = split_reply(streamed_rounds, final_round, final_text)",
                 "assistant_parts = reply_parts(joined_reasoning, narration_text, full_response)",
                 "yield self.streaming_handler.format_aisdk_narration(response.content)",
                 "await sse_queue.put(self.streaming_handler.format_aisdk_narration(resp.content))"):
        assert step in source, step
    assert service.StreamingChatService._stream_tool_loop.__code__.co_qualname == "the_loop_writes_receipts.<locals>.wrapped"
    assert service.StreamingChatService.stream_response_with_agent.__code__.co_qualname == (
        "writes_its_receipts.<locals>.wrapped")
