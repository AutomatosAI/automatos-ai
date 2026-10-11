"""PRD-256 FX-005 (night 12, C3 and G3): what the stream says is true.

C3: the tool loop set the tool-end event's ``success`` to True whenever the call did not raise,
and the chat's tool callback never raises, so every refusal streamed ``success: true`` (25 of 139
eval calls: a green tick on a refusal). Now the flag is the result's own: a refused result, or an
ask waiting for the owner's click (FX-004), streams ``success: false``, and its line says why.

G3: two producers wrote "Just to be clear…": the turn's ``no_tool_call`` notice (from
``unexecuted_claims_notice``'s claim family) and the receipts' line above the answer; nothing
deduped them. Now the receipts rule is the only producer: the notice keeps the unrun source
(F099) and the prose narration (#746), and the narration defers to the receipts when they already
say nothing was done. Both reply paths are tested: the first reply that ran no tool, and the
tool loop's reply.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.receipts import (
    ABOVE, DONE, NOTHING_DONE_LINE, READ, WRITE, honesty_lines, says_nothing_was_done,
)
from consumers.chatbot.tool_summary import tool_result_summary
from modules.tools.execution.card_raised import ACT, TOOL_END, WAITING, emit_flagged
from modules.tools.execution.tool_loop import ToolLoopExecutor

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
MODEL = "google/gemini-2.5-flash"
CLEAR = "Just to be clear"
REFUSED = {"success": False, "error": "Mission m-1 is not waiting for approval."}
ASK = {"success": False, "requires_confirmation": True, "grant_id": 42, ACT: "change an agent 'Scout' (agent #12)",
       "message": "Waiting for the owner's click on the card."}
DONE_RESULT = {"success": True, "message": "Moved #0422 to Done"}
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object",
                                                                                      "properties": {}}}}]
APPROVE = {"action": "platform_approve_mission", "params": {"mission_id": "m-1"}}
SAID_APPROVED = "I've approved the mission. It's now running."
# Night 3's claims (F108): each one, with no write behind it, is the receipts' line now.
NIGHT_3 = [
    ("I've approved the mission. It's now running.", "platform_approve_mission"),
    ("I've noted that the Taster plan is now £14.", "platform_store_memory"),
    ("I've put your newsletter on the board.", "platform_create_task"),
    ("Done — I've created a new agent called REPORT GENERATOR.", "platform_create_agent"),
    ("I've emailed Declan the invoice.", "GMAIL_SEND_EMAIL"),
    ("I've cancelled the Friday schedule.", "platform_cancel_scheduled_task"),
    ("I've renamed task 12 for you.", "platform_update_task"),
]


# ── C3: the tool-end flag is the result's own ───────────────────────────────

class _Rounds:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def _tool_end(result):
    """The loop's tool-end event for one call that returned ``result``."""
    from core.llm.clients.base import LLMResponse

    events = []

    async def tool(name, args, call_id, workspace_id):
        return result

    async def on_event(event):
        events.append(event)

    call = {"id": "call_1", "type": "function", "function": {"name": "platform_execute", "arguments": "{}"}}
    executor = ToolLoopExecutor(llm_callback=_Rounds(LLMResponse(content="Here is what happened.", tool_calls=None)),
                                tool_callback=tool, max_iterations=3)
    asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[call]),
                             messages=[{"role": "user", "content": "do it"}], tools=[], workspace_id="ws",
                             on_event=on_event))
    (end,) = [e for e in events if e.get("type") == "tool-end"]
    return end


def test_a_refused_result_streams_success_false_with_its_reason():
    end = _tool_end(REFUSED)
    assert end["success"] is False
    assert WAITING not in end
    assert tool_result_summary(end["result"]) == REFUSED["error"]


def test_an_ask_streams_success_false_with_the_waiting_line():
    """P256-FIX-RVW-22: beside ``success: false`` the ask carries ``waiting: true``, so the
    activity trail draws an hourglass and the card's words, never a red "<tool> failed"."""
    end = _tool_end(ASK)
    assert end["success"] is False
    assert end[WAITING] is True
    assert tool_result_summary(end["result"]) == TOOL_END.format(act=ASK[ACT])


def test_a_done_result_streams_success_true():
    end = _tool_end(DONE_RESULT)
    assert end["success"] is True
    assert WAITING not in end
    assert tool_result_summary(end["result"]) == DONE_RESULT["message"]


def _frame_of(result, *, call_id="call_1", format_id="call_1"):
    """The tool-end frame the chat formats for the loop's event, as its callback does."""
    from consumers.chatbot.streaming import get_streaming_handler

    handler, frames = get_streaming_handler(), []

    async def on_event(event):
        frames.append(handler.format_aisdk_tool_end(
            tool_call_id=format_id, tool_name=event["tool_name"], success=bool(event.get("success")),
            summary=tool_result_summary(event.get("result")),
        ))

    event = {"type": "tool-end", "tool_call_id": call_id, "tool_name": "platform_execute", "success": True,
             "duration_ms": 3, "result": result}
    asyncio.run(emit_flagged(on_event, event))
    (frame,) = frames
    return json.loads(frame[2:])["data"]


def test_the_frame_of_an_ask_says_waiting_and_no_other_frame_does():
    ask = _frame_of(ASK)
    assert ask["success"] is False and ask[WAITING] is True
    assert ask["summary"] == TOOL_END.format(act=ASK[ACT])
    assert WAITING not in _frame_of(REFUSED)
    assert WAITING not in _frame_of(DONE_RESULT)
    assert WAITING not in _frame_of(ASK, format_id="call_2")            # another call's frame: unchanged


def test_a_frame_formatted_outside_the_loop_never_waits():
    from consumers.chatbot.streaming import get_streaming_handler

    _frame_of(ASK)                                                       # the loop's wait is over
    plain = json.loads(get_streaming_handler().format_aisdk_tool_end("call_1", "t", False)[2:])["data"]
    assert WAITING not in plain
    said = json.loads(get_streaming_handler().format_aisdk_tool_end("c", "t", False, waiting=True)[2:])["data"]
    assert said[WAITING] is True


@pytest.mark.parametrize("result, success", [
    ({"success": False, "llm_context": "…", "raw_result": REFUSED}, False),
    ({"success": False, "llm_context": "…", "raw_result": ASK}, False),
    ({"success": True, "llm_context": "…", "raw_result": ASK}, False),        # an ask inside is never a tick
    ({"success": True, "llm_context": "…", "raw_result": DONE_RESULT}, True),
    ({"results": []}, True),                                                 # no flag: it came back
    ({"success": None}, False),                                              # bool(result.get("success", True))
    ("plain text", True),                                                    # not a dict: it came back
], ids=["refused", "ask", "ask-inside", "done", "no-flag", "none", "text"])
def test_the_chats_envelope_says_the_same(result, success):
    """The chat's tool callback hands the loop its envelope, the executor's answer as ``raw_result``."""
    assert _tool_end(result)["success"] is success


# ── G3: one producer of the not-done line ───────────────────────────────────

def _notice(reply, *, ran=frozenset(), any_tool_ran=False):
    from consumers.chatbot.service import unexecuted_claims_notice

    return unexecuted_claims_notice(reply, TOOLS, set(ran), any_tool_ran=any_tool_ran)


def _receipt(action, status, kind=WRITE):
    return {"action": action, "kind": kind, "status": status, "subject": "", "effect": "", "reason": None}


@pytest.mark.parametrize("reply, backing", NIGHT_3, ids=[r[:24] for r, _ in NIGHT_3])
def test_a_claim_with_nothing_behind_it_is_the_receipts_line_only(reply, backing):
    """F108's sentences (moved from test_f108's chat-notice test): the receipts say it; the
    notice never says it again, whatever other tools ran."""
    assert _notice(reply) is None
    assert _notice(reply, ran={"platform_get_mission"}, any_tool_ran=True) is None
    assert honesty_lines([], reply) == [NOTHING_DONE_LINE]
    assert honesty_lines([_receipt("platform_get_mission", DONE, READ)], reply) == [NOTHING_DONE_LINE]
    assert honesty_lines([_receipt(backing, DONE)], reply) == []          # a write went through: no line


def test_a_first_reply_claiming_work_with_no_tool_gets_exactly_one_line():
    narrated_claim = "Let me set that up now. I've created the OPS agent ✅"
    assert says_nothing_was_done(narrated_claim)
    assert _notice(narrated_claim) is None                                # the receipts say it
    assert honesty_lines([], narrated_claim) == [NOTHING_DONE_LINE]


def test_a_narration_the_receipts_do_not_read_keeps_its_one_notice():
    """#746 kept: "both created" is no report the receipts rule reads, so the notice is the one line."""
    from consumers.chatbot.service import NARRATED_ACTIONS_NOTICE

    narrated = "Now let me create OPS and TRACKER. Good — both created."
    assert not says_nothing_was_done(narrated)
    assert _notice(narrated) == NARRATED_ACTIONS_NOTICE
    assert honesty_lines([], narrated) == []


def test_the_notice_has_no_claim_family_and_takes_no_done_set():
    from consumers.chatbot import service

    assert "done" not in inspect.signature(service.unexecuted_claims_notice).parameters
    source = inspect.getsource(service.unexecuted_claims_notice)
    assert "claimed_action_not_done" not in source and "not_done(" not in source
    assert "says_nothing_was_done(" in source


# ── both reply paths, streamed: one "Just to be clear" ─────────────────────

@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the US-002 tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None, model=MODEL,
              finish_reason="tool_calls" if calls else "stop")


class _Model:
    """The model's rounds in order; the last one again for any retry the loop asks for (F108's nudge)."""

    def __init__(self, *rounds):
        self.rounds = list(rounds)

    async def generate_response(self, messages, tools=None, on_delta=None):
        text, calls = self.rounds.pop(0) if len(self.rounds) > 1 else self.rounds[0]
        if on_delta is not None and text:
            await on_delta("text", text)
        return _round(text, calls)


class _Router:
    """ToolRouter.execute_and_format's envelope, the executor's answer as ``raw_result``."""

    def __init__(self, result):
        self.result = result

    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        success = bool(self.result.get("success"))
        said = "" if success else f"Tool {tool_name} failed: {self.result.get('error')}"
        return {"success": success, "frontend_data": {}, "llm_context": said or json.dumps(self.result),
                "raw_result": self.result, "fatal_error": False, "error_type": None}


def _service(result):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.widget_scopes, svc.widget_team, svc.widget_agent_lock = (), None, None
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router(result)
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


def _turn(monkeypatch, *, answer, result=None, with_loop=True):
    """One chat turn: the real tool loop (the model calls ``APPROVE``, then answers), or a first
    reply that ran nothing and asks the notice as the turn's first-reply branch does; then the
    answer's additions and the finish."""
    from consumers.chatbot.service import StreamingChatService, unexecuted_claims_notice

    monkeypatch.setattr(StreamingChatService, "_hidden_action_scope", lambda self, agent_id: nullcontext())
    svc = _service(result or {"success": True})

    async def loop():
        runtime = NS(llm_manager=_Model((answer, None)), agent_id=1, workspace_id=WS, metadata=NS(name="Auto"))
        messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": "Approve it."}]
        call = {"id": "call_1", "type": "function",
                "function": {"name": "platform_execute", "arguments": json.dumps(APPROVE)}}
        async for chunk in svc._stream_tool_loop(_round("", [call]), messages, runtime, {}, TOOLS, prefetched=[]):
            yield chunk

    async def scoped(*args, **kwargs):
        if with_loop:
            async for chunk in loop():
                if isinstance(chunk, str):
                    yield chunk
        else:
            notice = unexecuted_claims_notice(answer, TOOLS, set(), any_tool_ran=False)
            if notice:
                yield svc.streaming_handler.format_aisdk_limit_reached(limit="no_tool_call", value=0, message=notice)
        StreamingChatService._answer_additions(None, _round(answer))
        yield svc.streaming_handler.format_aisdk_finish()

    svc._stream_response_with_agent_scoped = scoped

    async def run():
        return [c async for c in svc.stream_response_with_agent(
            chat_id="chat-1", messages=[{"role": "user", "content": "Approve it."}], agent_id=1, user_id=1)]
    return asyncio.run(run())


def _data(chunks, kind):
    return [json.loads(c[2:])["data"] for c in chunks if isinstance(c, str) and c.startswith(f'd:{{"type": "{kind}"')]


def _lines_said(chunks):
    """Every line the owner is shown about what was or was not done: the notices, then the lines above."""
    notices = [d["message"] for d in _data(chunks, "limit_reached")]
    (frame,) = _data(chunks, "receipts")
    return notices + list(frame.get(ABOVE, []))


def test_the_tool_loop_reply_carries_one_just_to_be_clear(monkeypatch):
    chunks = _turn(monkeypatch, answer=SAID_APPROVED, result=REFUSED)

    said = _lines_said(chunks)
    assert sum(line.startswith(CLEAR) for line in said) == 1
    assert said[-1] == NOTHING_DONE_LINE
    (end,) = _data(chunks, "tool-end")
    assert end["success"] is False and end["summary"]      # no green tick on the refusal
    assert WAITING not in end                              # and no hourglass: it was refused


def test_the_no_tool_first_reply_carries_one_just_to_be_clear(monkeypatch):
    chunks = _turn(monkeypatch, answer="Let me set that up now. I've created the OPS agent ✅", with_loop=False)

    assert _lines_said(chunks) == [NOTHING_DONE_LINE]
    assert not _data(chunks, "limit_reached")


def test_a_done_write_streams_true_and_no_line(monkeypatch):
    chunks = _turn(monkeypatch, answer=SAID_APPROVED, result={"success": True, "mission_id": "m-1"})

    assert _lines_said(chunks) == []
    (end,) = _data(chunks, "tool-end")
    assert end["success"] is True
    assert WAITING not in end


def test_an_ask_in_the_chats_loop_streams_its_frame_as_waiting(monkeypatch):
    """P256-FIX-RVW-22: through the chat's own callback (inside ``_stream_tool_loop``, untouched)."""
    chunks = _turn(monkeypatch, answer="Card raised: change Scout. Nothing changes until you click.", result=ASK)

    (end,) = _data(chunks, "tool-end")
    assert end["success"] is False and end[WAITING] is True
    assert end["summary"] == TOOL_END.format(act=ASK[ACT])
