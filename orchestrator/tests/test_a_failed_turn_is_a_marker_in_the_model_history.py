"""A failed turn is a marker in the model's history, so the next message is answered.

Production, 23 Oct: the person asked "Can you research documents for agentfactory" (the
turn failed at the AI provider), said "Hi" three times (each failed), then "Hey" once the
provider worked, and Auto answered the agentfactory research. Each failure is saved as an
assistant note, "⚠️ <sentence>" with source origin ``turn_error``
(``StreamingChatService._persist_turn_error``), and the next turn's history handed that
note to the model as Auto's own reply.

The notes here are saved by the real ``_persist_turn_error``, and the history is built by
the real entry points: the chat route (``api.chat.turn_history``, then
``PromptAnalyzer.convert_to_llm_messages``), the public widget and the voice call. No
database, network or model.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from api import chat as chat_route
from api.chat_models import ChatMessageRequest, MessagePart
from consumers.chatbot.model_history import FAILED_TURN_MARKER
from consumers.chatbot.prompt_analyzer import PromptAnalyzer
from consumers.chatbot.service import StreamingChatService
from consumers.chatbot.turn_errors import CODE_PROVIDER_FAILED, TurnError

WS = "6a0f3c1e-2b4d-4e8f-9a1b-0000000f7e01"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-0000000c7a01"
ASKED = "Can you research documents for agentfactory"
NEWEST = "Hey"
WARNING_MARK = "⚠️"
SAVED_ORIGIN = "turn_error"  # what the production rows carry, and what the client reads
PROVIDER_DOWN = TurnError(
    CODE_PROVIDER_FAILED,
    "Auto could not finish this reply: the AI provider had a problem on its side. "
    "Nothing needs changing on your side; ask again in a minute.",
)


class _SavedChat:
    """A chat's saved messages, in order: the two ChatService calls a turn makes."""

    def __init__(self) -> None:
        self.rows = []

    def save_message(self, *, chat_id, role, parts, workspace_id, source=None, **_):
        row = NS(role=role, parts=parts, source=source)
        self.rows.append(row)
        return row

    def get_messages_by_chat_id(self, chat_id, limit=None):
        return list(self.rows)


def _ask(chat: _SavedChat, text: str) -> None:
    chat.save_message(chat_id=CHAT_ID, role="user", parts=[{"type": "text", "text": text}], workspace_id=WS)


def _fail(chat: _SavedChat) -> None:
    """The turn failed at the AI provider: the note the streaming service saves for it."""
    StreamingChatService._persist_turn_error(NS(chat_service=chat, workspace_id=WS), CHAT_ID, PROVIDER_DOWN)


def _the_23_oct_chat() -> _SavedChat:
    chat = _SavedChat()
    _ask(chat, ASKED)
    _fail(chat)
    for _ in range(3):
        _ask(chat, "Hi")
        _fail(chat)
    _ask(chat, NEWEST)
    return chat


def _chat_turn_history(chat: _SavedChat):
    """The history the chat route hands the turn (the detached turn reads the same)."""
    newest = ChatMessageRequest(parts=[MessagePart(type="text", text=NEWEST)])
    return chat_route.turn_history(chat, CHAT_ID, newest, {})


def _model_reads(history):
    """The messages the model is given for that history."""
    return PromptAnalyzer().convert_to_llm_messages(history, system_prompt="")


def _no_warning_reaches_the_model(model) -> bool:
    return all(WARNING_MARK not in message["content"] for message in model)


def test_a_failed_turns_note_reaches_the_model_as_the_marker_not_as_autos_reply():
    chat = _SavedChat()
    _ask(chat, ASKED)
    _fail(chat)
    _ask(chat, NEWEST)
    assert chat.rows[1].source["origin"] == SAVED_ORIGIN  # the note under test is the real one

    model = _model_reads(_chat_turn_history(chat))

    assert _no_warning_reaches_the_model(model)
    assert PROVIDER_DOWN.message not in "\n".join(message["content"] for message in model)
    assert model == [
        {"role": "user", "content": ASKED},
        {"role": "assistant", "content": FAILED_TURN_MARKER},
        {"role": "user", "content": NEWEST},
    ]


def test_every_failed_turn_in_a_row_gets_its_own_marker_and_the_newest_message_is_last():
    model = _model_reads(_chat_turn_history(_the_23_oct_chat()))

    assert _no_warning_reaches_the_model(model)
    assert [m["content"] for m in model if m["role"] == "assistant"] == [FAILED_TURN_MARKER] * 4
    assert [m["content"] for m in model if m["role"] == "user"] == [ASKED, "Hi", "Hi", "Hi", NEWEST]
    assert model[-1] == {"role": "user", "content": NEWEST}


def test_the_saved_note_and_what_the_ui_shows_stay_as_they_are():
    chat = _the_23_oct_chat()

    _chat_turn_history(chat)

    note = chat.rows[1]
    assert note.parts == [{"type": "text", "text": f"{WARNING_MARK} {PROVIDER_DOWN.message}"}]
    assert note.source == {"origin": SAVED_ORIGIN, "label": "Error", "code": CODE_PROVIDER_FAILED}


def test_a_conversation_without_a_failed_turn_is_unchanged():
    chat = _SavedChat()
    _ask(chat, "Hi")
    chat.save_message(chat_id=CHAT_ID, role="assistant", workspace_id=WS,
                      parts=[{"type": "text", "text": "Hello! What can I do for you?"}])
    _ask(chat, "What's on the board?")
    # A message a background producer posted (PRD-205) carries a source too: it stays Auto's words.
    chat.save_message(chat_id=CHAT_ID, role="assistant", workspace_id=WS,
                      parts=[{"type": "text", "text": "Your 9 o'clock report is ready."}],
                      source={"origin": "watcher", "label": "Auto · background"})
    _ask(chat, NEWEST)

    history = _chat_turn_history(chat)

    assert history == [{"role": row.role, "parts": row.parts} for row in chat.rows]
    assert FAILED_TURN_MARKER not in "\n".join(m["content"] for m in _model_reads(history))


def test_the_widget_turn_reads_the_marker():
    from api.widgets import chat as widget_chat

    chat = _SavedChat()
    _ask(chat, ASKED)
    _fail(chat)
    turn = widget_chat._WidgetTurn(message=NEWEST, system_preamble=None, vertical="generic", is_proactive=False)

    model = _model_reads(widget_chat._history_for_turn(chat, CHAT_ID, turn, WS, "[widget test]"))

    assert _no_warning_reaches_the_model(model)
    assert model[1] == {"role": "assistant", "content": FAILED_TURN_MARKER}
    assert model[-1] == {"role": "user", "content": NEWEST}


def test_the_voice_turn_reads_the_marker(monkeypatch):
    import api.voice_retell as voice
    import consumers.chatbot as chatbot_pkg
    import modules.voice.call_binding as call_binding

    chat = _the_23_oct_chat()
    asked = {}

    class _Streams:
        def __init__(self, db, workspace_id=None):
            pass

        def stream_response_with_agent(self, **kwargs):
            asked.update(kwargs)

    monkeypatch.setattr(chatbot_pkg, "ChatService", lambda db: chat)
    monkeypatch.setattr(chatbot_pkg, "StreamingChatService", _Streams)
    monkeypatch.setattr(call_binding, "upsert_voice_user_message", lambda db, **kwargs: None)
    monkeypatch.setattr(voice, "named_call_agent", lambda db, workspace_id, agent_id: 1)
    monkeypatch.setattr(voice.config, "VOICE_LIVE_TURN_HISTORY_MESSAGES", len(chat.rows))
    binding = NS(chat_id=CHAT_ID, workspace_id=WS, user_id=7, is_super_admin=False)

    voice._start_spoken_turn(MagicMock(), NS(user_text=NEWEST, agent_id=None), binding)

    model = _model_reads(asked["messages"])
    assert _no_warning_reaches_the_model(model)
    assert [m["content"] for m in model if m["role"] == "assistant"] == [FAILED_TURN_MARKER] * 4
    assert model[-1] == {"role": "user", "content": NEWEST}
