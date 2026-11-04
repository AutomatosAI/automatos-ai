"""F105 (4 Nov): a widget turn hands its request's connection back and streams on a session of its own.

``POST /api/widgets/chat`` saved the visitor's message and read the history, the agent
and the AI notice on the request's ``get_db`` session, synchronously on the event loop,
then handed that same session to the streaming service. FastAPI closes it only when the
response ends, so the request's pooled connection was the turn's for the whole stream.

Now the turn's database work runs in worker threads and ends with the request's
transaction (its connection goes back to the pool) before the route returns; the stream
builds its streaming service on a session of its own and closes it when the stream ends.
The frames are unchanged: the AI notice first, the reply, then ``done``.

DB-free: no database, model or socket.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace as NS
from uuid import uuid4

import api.widgets.chat as chat_mod
import consumers.chatbot as chatbot
from api.widgets.auth import WidgetAuthContext
from modules.policy.ai_disclosure import build_disclosure

REPLY = "Welcome to Harbour Coffee"


class _Events:
    """What happened, in order, and on which thread."""

    def __init__(self):
        self.seen = []

    def add(self, what):
        self.seen.append((what, threading.get_ident()))

    def names(self):
        return [what for what, _ in self.seen]

    def thread_of(self, what):
        return next(thread for name, thread in self.seen if name == what)


class _RequestSession:
    """The request's ``get_db`` session: records when its transaction ends."""

    def __init__(self, events):
        self._events = events

    def commit(self):
        self._events.add("request transaction ended")


class _TurnSession:
    """The turn's own session (``SessionLocal``)."""

    def __init__(self, events):
        self._events = events
        events.add("turn session opened")

    def close(self):
        self._events.add("turn session closed")


def _patch_the_turn(monkeypatch, events):
    class _Chats:
        def __init__(self, db):
            self.db = db

        def create_chat(self, **kwargs):
            return NS(id=uuid4())

        def save_message(self, **kwargs):
            events.add("message saved")

        def get_messages_by_chat_id(self, chat_id):
            return [NS(role="user", parts=[{"type": "text", "text": "hi"}])]

    class _Streams:
        def __init__(self, db, **kwargs):
            events.streamed_on = db

        async def stream_response_with_agent(self, **kwargs):
            events.add("turn produced")
            yield f'0:"{REPLY}"\n'

    def _user(db):
        events.add("user looked up")
        return 1

    monkeypatch.setattr(chatbot, "ChatService", _Chats)
    monkeypatch.setattr(chatbot, "StreamingChatService", _Streams)
    monkeypatch.setattr(chat_mod, "SessionLocal", lambda: _TurnSession(events))
    monkeypatch.setattr(chat_mod, "_get_widget_user_id", _user)
    monkeypatch.setattr(chat_mod, "_resolve_workspace_vertical", lambda db, workspace_id: "generic")
    monkeypatch.setattr(chat_mod, "_effective_agent_id", lambda *args: 7)
    monkeypatch.setattr(chat_mod, "_retrieval_team", lambda *args: None)
    monkeypatch.setattr(chat_mod, "resolve_ai_disclosure", lambda db, workspace_id: build_disclosure({}, None))


def _one_turn(monkeypatch):
    events = _Events()
    _patch_the_turn(monkeypatch, events)
    auth = WidgetAuthContext(workspace_id=uuid4(), api_key_id=uuid4(), permissions=["chat"])
    request_db = _RequestSession(events)

    async def turn():
        response = await chat_mod.widget_chat(
            body=chat_mod.WidgetChatRequest(message="hi"), request=NS(headers={}), auth=auth, db=request_db,
        )
        events.add("route returned")
        return threading.get_ident(), [frame async for frame in response.body_iterator]

    loop_thread, frames = asyncio.run(turn())
    return events, request_db, loop_thread, frames


def _event_names(frames):
    return [frame.split("\n", 1)[0].removeprefix("event: ") for frame in frames]


def test_the_request_transaction_ends_before_the_turn_streams(monkeypatch):
    events, _, _, _ = _one_turn(monkeypatch)

    assert events.names() == [
        "user looked up", "message saved", "request transaction ended", "route returned",
        "turn session opened", "turn produced", "turn session closed",
    ]


def test_the_turn_streams_on_its_own_session_not_the_requests(monkeypatch):
    events, request_db, _, _ = _one_turn(monkeypatch)

    assert isinstance(events.streamed_on, _TurnSession)
    assert events.streamed_on is not request_db


def test_the_turns_database_work_runs_off_the_event_loop(monkeypatch):
    events, _, loop_thread, _ = _one_turn(monkeypatch)

    for work in ("user looked up", "message saved", "request transaction ended"):
        assert events.thread_of(work) != loop_thread, f"{work} ran on the event loop"


def test_the_frames_are_unchanged(monkeypatch):
    _, _, _, frames = _one_turn(monkeypatch)

    assert _event_names(frames) == ["ai_disclosure", "message", "done"]
    message = json.loads(frames[1].strip("\n").split("\n")[1].removeprefix("data: "))
    assert message["content"] == REPLY and message["ai_generated"] is True
