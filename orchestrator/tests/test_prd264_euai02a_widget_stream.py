"""PRD-264 EUAI-02a — the widget chat stream opens with the AI notice.

``POST /api/widgets/chat`` streams Server-Sent Events. Article 50(1): its
first event is ``{"type": "ai_disclosure", ...}``, before any token, so any
client (the SDK or a custom one) has the notice before the first answer.
Article 50(2): every assistant ``message`` event carries
``"ai_generated": true``. The stream's other events (tool-start, tool-end,
the callback-form bridge, error, done) keep their PRD-008-A.2 shapes.

Nothing here is gated by ``AUTH_EDITION``; the ``edition`` fixture runs each
case in both editions to prove it.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config as app_config  # noqa: E402

SHOP = "Harbour Coffee"
NOTICE = {
    "type": "ai_disclosure",
    "badge": "AI",
    "line": f"You are chatting with an AI assistant for {SHOP}.",
    "locale": "en",
}


@pytest.fixture(params=("local", "saas"))
def edition(request, monkeypatch):
    monkeypatch.setattr(app_config, "AUTH_EDITION", request.param)
    return request.param


def _parse(frames: list[str]) -> list[tuple[str, dict]]:
    """``event: X\\ndata: {...}\\n\\n`` frames as ``(event, payload)`` pairs."""
    parsed = []
    for frame in frames:
        assert frame.endswith("\n\n"), frame
        event_line, data_line = frame.strip("\n").split("\n")
        parsed.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return parsed


def _ctx(chat_id: str = "chat-1"):
    from api.widgets.chat_stream import WidgetStreamContext
    from modules.policy.ai_disclosure import build_disclosure

    return WidgetStreamContext(
        chat_id=chat_id, agent_id=7, team=None, is_proactive=False,
        log_extra="[widget_chat test]", started_at=0.0, disclosure=build_disclosure({}, SHOP),
    )


def _stream(chunks, fail_with=None) -> list[tuple[str, dict]]:
    from api.widgets.chat_stream import widget_event_stream

    async def _source():
        for chunk in chunks:
            yield chunk
        if fail_with is not None:
            raise fail_with

    async def _collect():
        return [frame async for frame in widget_event_stream(_ctx(), _source)]

    return _parse(asyncio.run(_collect()))


# ── the stream itself ──────────────────────────────────────────────────────

def test_the_first_event_is_the_notice_before_any_token(edition):
    events = _stream(['0:"Hello"\n', '0:" there"\n'])
    assert events[0] == ("ai_disclosure", {**NOTICE, "conversation_id": "chat-1"})
    assert [name for name, _ in events] == ["ai_disclosure", "message", "message", "done"]


def test_every_assistant_message_event_is_marked_ai_generated(edition):
    events = _stream(['0:"Hello"\n', {"_final_response": "x"}, 'e:{"finishReason":"stop"}\n', '0:" there"\n'])
    messages = [payload for name, payload in events if name == "message"]
    assert messages == [
        {"content": "Hello", "conversation_id": "chat-1", "ai_generated": True},
        {"content": " there", "conversation_id": "chat-1", "ai_generated": True},
    ]


def test_the_notice_comes_first_even_when_the_answer_fails(edition):
    events = _stream([], fail_with=RuntimeError("model down"))
    assert [name for name, _ in events] == ["ai_disclosure", "error", "done"]
    assert events[-1] == ("done", {"conversation_id": "chat-1"})


def test_tool_events_keep_their_shapes():
    start = json.dumps({"type": "tool-start", "data": {"toolName": "search_docs", "input": {"q": "beans"}}})
    end = json.dumps({"type": "tool-end", "data": {"toolName": "search_docs", "result": {"hits": 2}}})
    events = _stream([f"d:{start}\n", f"d:{end}\n", "d:not json\n"])
    assert events[1:] == [
        ("tool-start", {"tool": "search_docs", "arguments": {"q": "beans"}}),
        ("tool-end", {"tool": "search_docs", "result": {"hits": 2}}),
        ("done", {"conversation_id": "chat-1"}),
    ]


def test_the_callback_form_signal_becomes_one_event_with_the_tool_start_context():
    from modules.tools.widget_callback import (
        WIDGET_OPEN_CALLBACK_FORM_NAME,
        WIDGET_SIGNAL_KEY,
        WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
    )

    start = json.dumps({"type": "tool-start", "data": {
        "toolName": WIDGET_OPEN_CALLBACK_FORM_NAME, "input": {"product_context": "Blue mug"}}})
    signal = json.dumps({"type": "tool-data", "data": {WIDGET_SIGNAL_KEY: WIDGET_SIGNAL_OPEN_CALLBACK_FORM}})
    events = _stream([f"d:{start}\n", f"d:{signal}\n", f"d:{signal}\n"])
    forms = [payload for name, payload in events if name == "open-callback-form"]
    assert forms == [{"conversation_id": "chat-1", "product_context": "Blue mug"}]


# ── POST /api/widgets/chat end to end (chat services faked) ────────────────

def _db_with_site(site, workspace=None):
    """A db whose Site query returns ``site`` and Workspace query ``workspace``."""
    db = MagicMock()

    def query(model):
        chain = MagicMock()
        if "Site" in repr(model):
            chain.filter.return_value.order_by.return_value.first.return_value = site
        elif "Workspace" in repr(model):
            chain.filter.return_value.first.return_value = workspace
        else:
            chain.filter.return_value.first.return_value = None
        return chain

    db.query.side_effect = query
    return db


def _endpoint_events(monkeypatch, db) -> list[tuple[str, dict]]:
    """One ``POST /api/widgets/chat`` turn on ``db``, chat services faked."""
    import api.widgets.chat as chat_mod
    import consumers.chatbot as chatbot
    from api.widgets.auth import WidgetAuthContext

    class _Chats:
        def __init__(self, db):
            self.db = db

        def create_chat(self, **kwargs):
            return NS(id=uuid4())

        def save_message(self, **kwargs):
            return None

        def get_messages_by_chat_id(self, chat_id):
            return [NS(role="user", parts=[{"type": "text", "text": "hi"}])]

    class _Streams:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def stream_response_with_agent(self, **kwargs):
            yield '0:"Welcome to Harbour Coffee"\n'

    monkeypatch.setattr(chatbot, "ChatService", _Chats)
    monkeypatch.setattr(chatbot, "StreamingChatService", _Streams)
    monkeypatch.setattr(chat_mod, "_get_widget_user_id", lambda db: 1)
    monkeypatch.setattr(chat_mod, "_resolve_workspace_vertical", lambda db, workspace_id: "generic")
    monkeypatch.setattr(chat_mod, "_effective_agent_id", lambda *args: 7)
    monkeypatch.setattr(chat_mod, "_retrieval_team", lambda *args: None)

    auth = WidgetAuthContext(workspace_id=uuid4(), api_key_id=uuid4(), permissions=["chat"])

    async def _turn():
        response = await chat_mod.widget_chat(
            body=chat_mod.WidgetChatRequest(message="hi"), request=NS(headers={}),
            auth=auth, db=db,
        )
        return [frame async for frame in response.body_iterator]

    return _parse(asyncio.run(_turn()))


def test_the_widget_chat_endpoint_streams_the_notice_first(monkeypatch, edition):
    site = NS(display_name=SHOP, settings={"cart_idle": {"enabled": True}})
    events = _endpoint_events(monkeypatch, _db_with_site(site))
    name, payload = events[0]
    assert name == "ai_disclosure"
    assert {key: payload[key] for key in NOTICE} == NOTICE
    assert events[1] == ("message", {
        "content": "Welcome to Harbour Coffee", "conversation_id": payload["conversation_id"], "ai_generated": True,
    })
    assert events[-1][0] == "done"


# ── FR-15: the workspace name never reaches the stream ─────────────────────

PERSONAL_WORKSPACE_NAME = "jane@example.com's Workspace"


def test_the_first_event_never_carries_the_workspace_name(monkeypatch, edition):
    """An auto-created workspace is named after its owner; with no Site, the
    first event is the unnamed notice and holds neither the name nor the email."""
    workspace = NS(id=uuid4(), name=PERSONAL_WORKSPACE_NAME, settings={})
    events = _endpoint_events(monkeypatch, _db_with_site(None, workspace))

    name, payload = events[0]
    assert name == "ai_disclosure"
    assert payload["line"] == "You are chatting with an AI assistant."
    first = json.dumps(payload)
    for leaked in (PERSONAL_WORKSPACE_NAME, "jane", "example.com", "@"):
        assert leaked not in first
