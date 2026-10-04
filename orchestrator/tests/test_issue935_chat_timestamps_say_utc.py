"""#935 — every timestamp the chat API returns says it is UTC.

The chat tables store naive UTC (``DateTime`` columns filled by ``now()``), and the
routes serialised them with a bare ``.isoformat()``: no offset. ECMAScript reads an
offset-less date-time as LOCAL time, so in a UK browser (UTC+1) a message just sent
showed "1h ago" as soon as the messages were refetched. The routes now serialise
through ``core.utils.timestamps.utc_iso``.

The routes are called directly with a fake ChatService — no database.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import api.chat as chat_api
from services.chat_agent_switch import record_agent_switch

NAIVE_UTC = datetime(2026, 10, 4, 15, 49, 41, 311520)
CTX = SimpleNamespace(workspace_id="00000000-0000-0000-0000-0000000000c1")
USER_ID = 7


def _chat():
    return SimpleNamespace(
        id="chat-1", user_id=USER_ID, title="test", created_at=NAIVE_UTC, updated_at=NAIVE_UTC,
        visibility="private", last_context={}, kind="user",
    )


class _FakeChatService:
    def __init__(self, db):
        self.db = db

    def get_chat_history(self, user_id, limit, workspace_id):
        return [_chat()]

    def get_chat(self, chat_id, workspace_id=None):
        return _chat()

    def get_messages_by_chat_id(self, chat_id):
        return [SimpleNamespace(id="msg-1", role="user", parts=[], attachments=[], source=None, created_at=NAIVE_UTC)]


class _FakeTurns:
    async def is_in_flight(self, chat_id):
        return False


@pytest.fixture
def fake_chats(monkeypatch):
    monkeypatch.setattr(chat_api, "ChatService", _FakeChatService)
    monkeypatch.setattr(chat_api, "get_user_id", lambda db, ctx=None: USER_ID)
    monkeypatch.setattr(chat_api, "_last_message_previews", lambda db, ids: {})
    monkeypatch.setattr(chat_api, "get_turn_registry", lambda: _FakeTurns())


def _says_utc(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, f"{value!r} has no offset — a browser reads it as local time"
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.replace(tzinfo=None) == NAIVE_UTC, "the instant itself must not move"


def test_the_chat_list_says_utc(fake_chats):
    [chat] = asyncio.run(chat_api.get_chat_history(limit=1, ctx=CTX, db=object()))
    _says_utc(chat["createdAt"])
    _says_utc(chat["updatedAt"])


def test_one_chat_says_utc(fake_chats):
    chat = asyncio.run(chat_api.get_chat("chat-1", ctx=CTX, db=object()))
    _says_utc(chat["createdAt"])
    _says_utc(chat["updatedAt"])


def test_a_chats_messages_say_utc(fake_chats):
    [message] = asyncio.run(chat_api.get_chat_messages("chat-1", ctx=CTX, db=object()))
    _says_utc(message["createdAt"])


def test_message_search_says_utc(fake_chats):
    row = SimpleNamespace(
        id="msg-1", chat_id="chat-1", role="user", parts=[{"type": "text", "text": "hello"}],
        created_at=NAIVE_UTC, chat_title="test",
    )
    db = SimpleNamespace(execute=lambda *a, **k: SimpleNamespace(fetchall=lambda: [row]))
    found = asyncio.run(chat_api.search_chat_history(q="hello", limit=5, days=30, ctx=CTX, db=db))
    _says_utc(found["results"][0]["created_at"])


def test_no_chat_route_serialises_a_naive_timestamp():
    """A guard on the module: a new route that hand-rolls ``.isoformat()`` on a
    timestamp column brings the bug back. Use ``utc_iso`` instead."""
    from pathlib import Path

    source = Path(chat_api.__file__).read_text()
    assert ".created_at.isoformat()" not in source
    assert ".updated_at.isoformat()" not in source
    assert "utcnow().isoformat()" not in source


class _RecordingDb:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params):
        self.calls.append(params)


def test_an_agent_switch_is_recorded_in_utc_without_touching_the_chats_list():
    import json

    earlier = [{"timestamp": "2026-10-01T09:00:00+00:00", "from_agent_id": 1, "to_agent_id": 2, "reason": "x"}]
    chat = SimpleNamespace(id="chat-1", agent_switches=earlier)
    db = _RecordingDb()
    record_agent_switch(db, chat, 2, 3, None)

    [params] = db.calls
    written = json.loads(params["switches"])
    assert [s["to_agent_id"] for s in written] == [2, 3]
    assert datetime.fromisoformat(written[-1]["timestamp"]).utcoffset() == timedelta(0)
    assert written[-1]["reason"] == "User requested switch"
    assert chat.agent_switches == earlier and len(earlier) == 1, "the chat's own list is not mutated"


def test_an_agent_switch_reads_a_history_stored_as_json_text():
    import json

    chat = SimpleNamespace(id="chat-1", agent_switches=json.dumps([{"to_agent_id": 2}]))
    db = _RecordingDb()
    record_agent_switch(db, chat, 2, 4, "asked")

    written = json.loads(db.calls[0]["switches"])
    assert [s["to_agent_id"] for s in written] == [2, 4]
    assert written[-1]["reason"] == "asked"
