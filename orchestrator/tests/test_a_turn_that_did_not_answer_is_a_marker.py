"""Every turn that did not answer reaches the next turn as the marker (review of the 23 Oct fix).

The 23 Oct fix read a failed turn's ``turn_error`` note as ``FAILED_TURN_MARKER``. The review
found the turns it missed, each leaving the request before it looking open, so the model
answered that instead of the newest message:

* a tool loop that failed after its first round handed its sentence back as the answer, and
  the turn saved it as Auto's reply (``a_failed_loop_fails_the_turn``);
* a turn queued behind another read the conversation before that turn ended
  (``api.chat._produce_turn`` reads it once it holds the chat's queue);
* a turn stopped, or failed outside the streaming service, saved nothing; the session-agent
  lane saved nothing (``turn_notes``);
* the empty-completion and blank-after-tools sentences were saved as replies;
* a note saved after the turn's own reply read as "not answered";
* a note the aborted transaction refused was lost with a debug line;
* the thread checkpoint's distiller read the notes as Auto's replies.

The tool loop is the real one; the rest is DB-free, over a chat that keeps its rows in order.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from api.chat_models import ChatMessageRequest, MessagePart
from consumers.chatbot.empty_completion import EMPTY_COMPLETION_FALLBACK, NOTHING_SAID
from consumers.chatbot.model_history import FAILED_TURN_MARKER, history_for_model, is_turn_reply, rows_for_the_turn
from consumers.chatbot.prompt_analyzer import PromptAnalyzer
from consumers.chatbot.service import StreamingChatService, nothing_said_fallback
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.turn_errors import (
    CODE_PROVIDER_FAILED, CODE_SESSION_AGENT_TERMINAL, CODE_TURN_FAILED, CODE_TURN_STOPPED, TurnError, TurnFailed,
    describe_turn_error,
)
from consumers.chatbot.turn_notes import note_an_unanswered_turn, save_turn_note

WS = "6a0f3c1e-2b4d-4e8f-9a1b-0000000f7e02"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-0000000c7a02"
ASKED = "Can you research documents for agentfactory"
NEWEST = "Hey"
UPSTREAM = "upstream connect error"
MODEL = "google/gemini-2.5-flash"
PROVIDER_DOWN = TurnError(CODE_PROVIDER_FAILED, "Auto could not finish this reply: the AI provider had a problem "
                                                "on its side. Nothing needs changing on your side; ask again in a minute.")
TERMINAL = "NEWSROOM runs as a Claude Code session: talk to it in its terminal on the Runtime Canvas."
CARD_MOVE = {"action": "platform_update_task_status", "params": {"task_id": "#0422", "status": "done"}}
CARD_MOVED = {"success": True, "llm_context": "Ticket #0422 moved to done.",
              "raw_result": {"success": True, "task_id": 422, "number": "#0422", "status": "done"}}
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object",
                                                                                      "properties": {}}}}]


class _Chat:
    """A chat's saved rows, in order, with the ChatService calls the turn makes."""

    def __init__(self) -> None:
        self.rows = []

    def save_message(self, *, chat_id, role, parts, workspace_id, source=None, **_):
        row = NS(id=uuid4(), role=role, parts=parts, source=source)
        self.rows.append(row)
        return row

    def get_messages_by_chat_id(self, chat_id, limit=None):
        return list(self.rows)


def _ask(chat, text):
    return chat.save_message(chat_id=CHAT_ID, role="user", parts=[{"type": "text", "text": text}], workspace_id=WS)


def _reply(chat, text, source=None):
    return chat.save_message(chat_id=CHAT_ID, role="assistant", parts=[{"type": "text", "text": text}],
                             workspace_id=WS, source=source)


def _fail(chat, err=PROVIDER_DOWN):
    """The note the streaming service saves when its turn fails."""
    StreamingChatService._persist_turn_error(NS(chat_service=chat, workspace_id=WS), CHAT_ID, err)


def _model_reads(history):
    return PromptAnalyzer().convert_to_llm_messages(history, system_prompt="")


def _said(role, content):
    return {"role": role, "content": content}


async def _drain(stream):
    return [chunk async for chunk in stream]


# ── a tool loop that fails after its first round ────────────────────────────

@pytest.fixture
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the receipts tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None, model=MODEL,
              finish_reason="tool_calls" if calls else "stop")


class _ProviderDown:
    """Every model call after the first round fails at the provider."""

    async def generate_response(self, messages, tools=None, on_delta=None):
        raise RuntimeError(UPSTREAM)


class _Router:
    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        return CARD_MOVED


def _service():
    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.widget_scopes, svc.widget_team, svc.widget_agent_lock = (), None, None
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router()
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    return svc


def _failing_loop():
    """The real tool loop: round 1 moves #0422 to done, the model call after it fails."""
    svc, chunks = _service(), []
    runtime = NS(llm_manager=_ProviderDown(), agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": ASKED}]
    first = _round("", [{"id": "call_1", "type": "function",
                         "function": {"name": "platform_execute", "arguments": json.dumps(CARD_MOVE)}}])

    async def run():
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, prefetched=[]):
            chunks.append(chunk)

    with pytest.raises(TurnFailed) as raised:
        asyncio.run(run())
    return raised.value, chunks


@pytest.mark.usefixtures("_chat_budgets")
def test_a_later_round_failure_fails_the_turn_after_its_receipts_went_out():
    failed, chunks = _failing_loop()

    assert failed.error.code == CODE_TURN_FAILED
    assert failed.error.message == f"The agent could not finish this reply: {UPSTREAM}"
    receipts = [c for c in chunks if isinstance(c, str) and c.startswith('d:{"type": "receipts"')]
    (frame,) = [json.loads(c[2:])["data"] for c in receipts]
    assert frame["receipts"][0]["subject"] == "#0422"          # what ran before the failure is still shown
    finals = [c for c in chunks if isinstance(c, dict) and c.get("_final_response")]
    assert finals and chunks[-1] is finals[-1]                  # nothing after it: the turn's except takes over


@pytest.mark.usefixtures("_chat_budgets")
def test_the_turn_saves_the_loops_failure_as_its_note_and_the_next_turn_reads_the_marker():
    failed, _chunks = _failing_loop()
    err = describe_turn_error(failed, agent_name="Auto")        # the turn's own except, as for any failure

    assert err is failed.error                                  # described once, never wrapped again
    chat = _Chat()
    _ask(chat, ASKED)
    _fail(chat, err)
    _ask(chat, NEWEST)
    assert chat.rows[1].source == {"origin": "turn_error", "label": "Error", "code": CODE_TURN_FAILED}
    assert _model_reads(history_for_model(chat.rows)) == [
        _said("user", ASKED), _said("assistant", FAILED_TURN_MARKER), _said("user", NEWEST)]


def test_a_turn_errors_sentence_is_plain_text_on_the_wire():
    frame = get_streaming_handler().format_aisdk_error(PROVIDER_DOWN.message, code=PROVIDER_DOWN.code)
    assert json.loads(frame[2:]) == {"message": str(PROVIDER_DOWN.message), "code": CODE_PROVIDER_FAILED}


# ── a turn queued behind another reads how it ended ─────────────────────────

def _message(text):
    return ChatMessageRequest(parts=[MessagePart(type="text", text=text)])


def _turn(route, own, *, session_agent=False):
    lane = NS(session_agent=session_agent, suggest_mission=False, agent_id=11, skip_composio=False, assessment=None)
    return route._Turn(workspace_id=WS, chat_id=CHAT_ID, user_id=7, message=_message(own.parts[0]["text"]),
                       message_id=own.id, lane=lane, message_text=own.parts[0]["text"], mission_mode=False,
                       plan_mode=False, is_super_admin=False, page_context={}, spoken_mode=False)


def test_a_turn_queued_behind_a_failing_one_reads_its_note(monkeypatch):
    """The research turn is still running when "Hi" is sent; it fails while "Hi" waits for the queue."""
    from api import chat as route

    chat, asked = _Chat(), []
    _ask(chat, ASKED)
    hi = _ask(chat, "Hi")

    class _Streams:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = get_streaming_handler()

        async def stream_response_with_agent(self, **kwargs):
            asked.append(kwargs)
            yield get_streaming_handler().format_aisdk_finish()

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            _fail(chat)          # the turn ahead ends (its note) before this one holds the chat
            yield

    monkeypatch.setattr(route, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(route, "ChatService", lambda db: chat)
    monkeypatch.setattr(route, "StreamingChatService", _Streams)
    monkeypatch.setattr(route, "get_session_queue", lambda: _Queue())

    asyncio.run(_drain(route._produce_turn(_turn(route, hi))))

    (kwargs,) = asked
    assert _model_reads(kwargs["messages"]) == [
        _said("user", ASKED), _said("assistant", FAILED_TURN_MARKER), _said("user", "Hi")]


def test_the_turns_own_message_is_newest_and_later_messages_wait_for_their_turns():
    chat = _Chat()
    first = _ask(chat, ASKED)
    own = _ask(chat, "Hi")
    later = _ask(chat, "Hello?")
    answered = _reply(chat, "Here is the agentfactory research.")

    assert rows_for_the_turn(chat.rows, own.id) == [first, answered, own]
    assert later not in rows_for_the_turn(chat.rows, own.id)
    assert rows_for_the_turn(chat.rows, None) == chat.rows      # no message to read from: as saved


# ── a turn that saved nothing ───────────────────────────────────────────────

@pytest.fixture
def notes_db(monkeypatch):
    """``note_an_unanswered_turn``'s own session, over the in-memory chat."""
    import consumers.chatbot.service as service_module
    import core.database.database as database

    chat = _Chat()
    monkeypatch.setattr(database, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(service_module, "ChatService", lambda db: chat)
    return chat


def test_a_stopped_turn_leaves_its_note_and_the_next_turn_reads_the_marker(notes_db):
    asked = _ask(notes_db, ASKED)

    assert asyncio.run(note_an_unanswered_turn(WS, CHAT_ID, asked.id, cancelled=True)) is True

    assert notes_db.rows[-1].source == {"origin": "turn_error", "label": "Error", "code": CODE_TURN_STOPPED}
    _ask(notes_db, "Hi")
    assert _model_reads(history_for_model(notes_db.rows))[1] == _said("assistant", FAILED_TURN_MARKER)


def test_a_turn_that_failed_outside_the_service_says_it_did_not_finish(notes_db):
    asked = _ask(notes_db, ASKED)

    asyncio.run(note_an_unanswered_turn(WS, CHAT_ID, asked.id, cancelled=False))

    assert notes_db.rows[-1].source["code"] == CODE_TURN_FAILED


def test_stop_after_the_reply_was_saved_leaves_no_note(notes_db):
    asked = _ask(notes_db, ASKED)
    _reply(notes_db, "Here is the agentfactory research.")

    assert asyncio.run(note_an_unanswered_turn(WS, CHAT_ID, asked.id, cancelled=True)) is False
    assert len(notes_db.rows) == 2


def test_the_chat_route_notes_only_a_turn_that_did_not_complete(monkeypatch):
    from api import chat as route

    noted = []

    async def _note(workspace_id, chat_id, message_id, *, cancelled):
        noted.append((message_id, cancelled))
        return True

    monkeypatch.setattr(route, "note_an_unanswered_turn", _note)
    on_complete = route._after_the_turn(WS, CHAT_ID, 7, "message-1")

    asyncio.run(on_complete(completed=False, cancelled=True, client_gone=False))
    asyncio.run(on_complete(completed=True, cancelled=False, client_gone=False))

    assert noted == [("message-1", True)]


def test_the_session_agent_lane_keeps_its_answer_as_the_turns_note(monkeypatch):
    from api import chat as route
    import services.cli_ticket_lane as cli_lane

    chat = _Chat()
    own = _ask(chat, "Ship the release notes")
    monkeypatch.setattr(cli_lane, "session_agent_terminal_message", lambda db, agent_id: TERMINAL)
    monkeypatch.setattr(route, "ChatService", lambda db: chat)
    service = NS(streaming_handler=get_streaming_handler())

    (frame,) = asyncio.run(_drain(route._turn_chunks(service, NS(), _turn(route, own, session_agent=True), [])))

    assert json.loads(frame[2:]) == {"message": TERMINAL, "code": CODE_SESSION_AGENT_TERMINAL}
    assert chat.rows[-1].parts == [{"type": "text", "text": f"⚠️ {TERMINAL}"}]
    assert chat.rows[-1].source["code"] == CODE_SESSION_AGENT_TERMINAL


# ── what counts as no answer ────────────────────────────────────────────────

@pytest.mark.parametrize("said", [EMPTY_COMPLETION_FALLBACK, NOTHING_SAID])
def test_a_reply_saved_in_place_of_an_answer_is_the_marker(said):
    chat = _Chat()
    _ask(chat, ASKED)
    _reply(chat, said)
    _ask(chat, NEWEST)

    assert _model_reads(history_for_model(chat.rows))[1] == _said("assistant", FAILED_TURN_MARKER)


def test_a_card_waiting_for_approval_is_autos_answer():
    waiting = nothing_said_fallback({"tool_approval": {"message": "Send the invoice to Raj?"}})
    chat = _Chat()
    _ask(chat, "Email the invoice to Raj")
    _reply(chat, waiting)

    assert history_for_model(chat.rows)[1]["parts"][0]["text"] == waiting


def test_a_note_after_the_turns_own_reply_is_left_out():
    """The reply was saved, then a step after the save failed and the turn left a note too."""
    chat = _Chat()
    _ask(chat, "Draft and save the October post")
    _reply(chat, "Saved the October post as a draft.")
    _fail(chat)
    _ask(chat, NEWEST)

    assert _model_reads(history_for_model(chat.rows)) == [
        _said("user", "Draft and save the October post"), _said("assistant", "Saved the October post as a draft."),
        _said("user", NEWEST)]


def test_a_queued_turns_note_after_another_turns_reply_is_its_marker():
    chat = _Chat()
    _ask(chat, ASKED)
    _ask(chat, "Hi")
    _reply(chat, "Here is the agentfactory research.")
    _fail(chat)                                            # the "Hi" turn's
    _ask(chat, NEWEST)

    assert [m["content"] for m in _model_reads(history_for_model(chat.rows)) if m["role"] == "assistant"] == [
        "Here is the agentfactory research.", FAILED_TURN_MARKER]


def test_a_background_post_answers_no_request():
    chat = _Chat()
    _ask(chat, ASKED)
    _reply(chat, "Your 9 o'clock report is ready.", source={"origin": "watcher", "label": "Auto · background"})
    _fail(chat)
    _ask(chat, NEWEST)

    assert _model_reads(history_for_model(chat.rows))[2] == _said("assistant", FAILED_TURN_MARKER)
    assert not is_turn_reply(chat.rows[1])
    assert is_turn_reply(NS(role="assistant", source={"origin": "voice", "label": "Auto · voice"}))


def test_the_marker_never_says_nothing_was_done():
    """A turn can fail after its tools ran (the forced synthesis, a later round): F185's rule."""
    assert "may already be done" in FAILED_TURN_MARKER and "check before doing it again" in FAILED_TURN_MARKER
    assert "not answered" not in FAILED_TURN_MARKER


# ── a note the failure's transaction refused ────────────────────────────────

class _AbortedChat(_Chat):
    """A session the turn's failure left aborted: its first ``failures`` saves are refused."""

    def __init__(self, failures):
        super().__init__()
        self.failures, self.rollbacks = failures, 0
        self.db = NS(rollback=self._rolled_back)

    def _rolled_back(self):
        self.rollbacks += 1

    def save_message(self, **kwargs):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("current transaction is aborted, commands ignored until end of transaction block")
        return super().save_message(**kwargs)


def test_a_note_the_aborted_transaction_refused_is_saved_after_a_rollback():
    chat = _AbortedChat(failures=1)

    assert save_turn_note(chat, CHAT_ID, WS, PROVIDER_DOWN) is True
    assert chat.rollbacks == 1 and chat.rows[-1].source["origin"] == "turn_error"


def test_a_note_that_never_saves_is_logged_with_its_trace(caplog):
    chat = _AbortedChat(failures=2)

    with caplog.at_level(logging.WARNING, logger="consumers.chatbot.turn_notes"):
        assert save_turn_note(chat, CHAT_ID, WS, PROVIDER_DOWN) is False

    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors and errors[-1].exc_info is not None and chat.rows == []


# ── the thread checkpoint's distiller ───────────────────────────────────────

def test_the_checkpoint_reads_a_failed_turn_as_the_marker():
    from modules.memory.thread_checkpoint import _load_messages

    chat = _Chat()
    _ask(chat, ASKED)
    _fail(chat)
    _ask(chat, NEWEST)
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = \
        list(reversed(chat.rows))                          # newest first, as the query orders them

    loaded = _load_messages(db, CHAT_ID)

    assert loaded[1] == {"role": "assistant", "parts": [{"type": "text", "text": FAILED_TURN_MARKER}]}
    assert loaded[-1]["parts"][0]["text"] == NEWEST
