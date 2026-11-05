"""F105 (4 Nov): a streaming chat turn hands its request's connection back before it streams.

``POST /api/chat`` saves the person's message and reads the history and the lane on the
request's ``get_db`` session, then streams a turn that runs on a session of its own
(``_produce_turn``). FastAPI closes the request's session only when the response ends,
so for as long as the reply streamed its connection sat idle in transaction beside the
turn's: every turn in flight held two of the pool's 30. A self-hosted install that ran
out of them froze for 30 s at a time (the pool's wait) on the routes that came next.

The route ends the request session's transaction before it returns the stream (as
``speak`` in api/voice_speech.py does), which hands the connection back; ``get_db`` still
closes the session after the response. The turn's first frame comes after it, and the
frames are the turn's, unchanged.

DB-free: no database, model or socket.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

from api import chat
from api.chat_models import ChatRequest
from consumers.chatbot.streaming import StreamingHandler

WS = "6a0f3c1e-2b4d-4e8f-9a1b-f10500c00001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-f10500c0c4a7"
HANDLER = StreamingHandler()
FRAMES = (HANDLER.format_aisdk_chat_id(CHAT_ID), HANDLER.format_aisdk_text("Done."), HANDLER.format_aisdk_finish())


class _RequestSession:
    """The request's ``get_db`` session: records when its transaction ends."""

    def __init__(self, events):
        self._events = events

    def rollback(self):
        self._events.append("request transaction ended")


def _patch_the_route(monkeypatch, events):
    class _ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kwargs):
            return NS(id=42)

    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = StreamingHandler()

        async def stream_response_with_agent(self, **kwargs):
            events.append("turn produced")
            for frame in FRAMES:
                yield frame

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _auto_answers(*args, **kwargs):
        return NS(agent_id=11, assessment=None, session_agent=False, suggest_mission=False, skip_composio=False)

    async def _inline(*, chat_id, produce, on_complete=None):
        async for chunk in produce():
            yield chunk

    monkeypatch.setattr(chat, "ChatService", _ChatService)
    monkeypatch.setattr(chat, "StreamingChatService", _StreamingService)
    monkeypatch.setattr(chat, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Queue())
    monkeypatch.setattr(chat, "get_user_id", lambda db, ctx=None: 7)
    monkeypatch.setattr(chat, "_open_chat", lambda *args, **kwargs: CHAT_ID)
    monkeypatch.setattr(chat, "turn_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat, "_turn_lane", _auto_answers)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)


def test_the_request_session_hands_its_connection_back_before_the_turn_streams(monkeypatch):
    events = []
    _patch_the_route(monkeypatch, events)
    ctx = NS(workspace_id=WS, user=NS(id="owner@f105.test", system_role="user"))
    body = {"message": {"role": "user", "parts": [{"type": "text", "text": "What needs me?"}]}}

    async def run():
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=_RequestSession(events))
        events.append("route returned")
        return [chunk async for chunk in response.body_iterator]

    frames = asyncio.run(run())

    assert events == ["request transaction ended", "route returned", "turn produced"]
    assert tuple(frames) == FRAMES
