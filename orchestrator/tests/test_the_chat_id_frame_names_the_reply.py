"""A chat turn's ``chat-id`` frame names the id its reply is saved under (``consumers/chatbot/reply_id``).

The id is chosen before the turn's first frame, said on the ``chat-id`` frame as ``messageId`` beside
``chatId``, and the turn's reply (its assistant save with a trace) is saved under it, once. Any other
message the turn saves, such as the note a failed turn leaves, takes its own id, so a failed turn's
id names no message. Every turn has its own id. The service's own stream handler and chat service
come back when the turn ends, however it ends. ``ChatService.save_message`` saves under a given id
(real Postgres, CI); everything else is DB-free.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from consumers.chatbot.reply_id import says_its_replys_id
from consumers.chatbot.streaming import StreamingHandler

CHAT = "0c0ffee0-0000-4000-8000-00000000c4a7"
WS = str(uuid.uuid4())


def _data(frame: str) -> dict:
    assert frame.startswith("d:") and frame.endswith("\n")
    return json.loads(frame[2:])


class _Chats:
    """A chat service that records each save and answers it as saved."""

    def __init__(self):
        self.saved = []

    def save_message(self, **kwargs):
        self.saved.append(kwargs)
        return SimpleNamespace(id=kwargs.get("message_id") or uuid.uuid4())


def _service():
    return SimpleNamespace(streaming_handler=StreamingHandler(), chat_service=_Chats())


def _reply(chat):
    return chat.chat_service.save_message(chat_id=CHAT, role="assistant", parts=[{"type": "text", "text": "Hi"}],
                                          workspace_id=WS, context_trace={"mode": "lean"})


def _note(chat):
    return chat.chat_service.save_message(chat_id=CHAT, role="assistant", parts=[{"type": "text", "text": "Failed"}],
                                          workspace_id=WS, source={"origin": "turn_error"})


def _run(turn, chat):
    async def go():
        return [frame async for frame in says_its_replys_id(turn)(chat)]
    return asyncio.run(go())


def test_the_frame_names_the_message_when_given_and_is_unchanged_without():
    handler = StreamingHandler()
    assert _data(handler.format_aisdk_chat_id(CHAT, message_id="m-1")) == {
        "type": "chat-id", "chatId": CHAT, "messageId": "m-1", "ai_generated": True}
    assert _data(handler.format_aisdk_chat_id(CHAT)) == {"type": "chat-id", "chatId": CHAT, "ai_generated": True}


def test_the_turns_reply_is_saved_under_the_id_its_chat_id_frame_said():
    chat = _service()

    async def turn(chat):
        yield chat.streaming_handler.format_aisdk_chat_id(CHAT)
        _reply(chat)

    (frame,) = _run(turn, chat)
    said = _data(frame)["messageId"]
    (saved,) = chat.chat_service.saved
    assert str(saved["message_id"]) == said
    assert uuid.UUID(said).version == 4


def test_a_failed_turns_note_takes_its_own_id_so_the_said_id_names_nothing():
    chat = _service()

    async def turn(chat):
        yield chat.streaming_handler.format_aisdk_chat_id(CHAT)
        _note(chat)

    (frame,) = _run(turn, chat)
    (note,) = chat.chat_service.saved
    assert "message_id" not in note
    assert _data(frame)["messageId"]


def test_only_the_first_reply_save_takes_the_id():
    chat = _service()

    async def turn(chat):
        yield chat.streaming_handler.format_aisdk_chat_id(CHAT)
        _reply(chat)
        _reply(chat)

    (frame,) = _run(turn, chat)
    first, second = chat.chat_service.saved
    assert str(first["message_id"]) == _data(frame)["messageId"] and "message_id" not in second


def test_each_turn_has_its_own_id():
    async def turn(chat):
        yield chat.streaming_handler.format_aisdk_chat_id(CHAT)

    ids = {_data(_run(turn, _service())[0])["messageId"] for _ in range(3)}
    assert len(ids) == 3


def test_the_services_own_handler_and_chat_service_come_back_even_when_the_turn_fails():
    chat = _service()
    handler, chats = chat.streaming_handler, chat.chat_service

    async def turn(chat):
        assert chat.streaming_handler is not handler and chat.chat_service is not chats
        yield chat.streaming_handler.format_aisdk_chat_id(CHAT)
        raise RuntimeError("the model went away")

    with pytest.raises(RuntimeError):
        _run(turn, chat)
    assert chat.streaming_handler is handler and chat.chat_service is chats


def test_the_chat_turn_is_wrapped_outermost():
    from consumers.chatbot.service import StreamingChatService

    async def _stub(chat, *args, **kwargs):
        yield None

    turn = StreamingChatService._stream_response_with_agent_scoped
    assert turn.__code__ is says_its_replys_id(_stub).__code__


def test_save_message_saves_under_the_id_it_is_given(db_session, seed_workspace):
    from consumers.chatbot.service import ChatService

    ws = seed_workspace()
    tag = uuid.uuid4().hex[:10]
    user = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                              {"e": f"reply-{tag}@reply.example.test", "u": f"reply-{tag}"}).scalar()
    chat_id = db_session.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility) VALUES "
                                      "(gen_random_uuid(), :u, CAST(:w AS uuid), 'walk', 'private') RETURNING id"),
                                 {"u": user, "w": ws}).scalar()
    chosen = uuid.uuid4()
    chats = ChatService(db_session)

    given = chats.save_message(chat_id=str(chat_id), role="assistant", parts=[], workspace_id=ws, message_id=chosen)
    fresh = chats.save_message(chat_id=str(chat_id), role="assistant", parts=[], workspace_id=ws)

    assert given.id == chosen and fresh.id != chosen
    stored = db_session.execute(text("SELECT count(*) FROM messages WHERE id = :id"), {"id": chosen}).scalar()
    assert stored == 1
