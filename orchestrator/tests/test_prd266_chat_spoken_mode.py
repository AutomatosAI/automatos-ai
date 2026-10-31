"""PRD-266 US-007a (§4.7, D32) — ``spokenMode`` on ``POST /api/chat``: replies with no markdown.

* ``MessagePart``, ``ChatMessageRequest`` and ``ChatRequest`` live in ``api/chat_models.py``;
  ``api/chat.py`` imports them, keeps no copy, and is shorter than its 908 lines.
  ``ChatRequest.spokenMode`` parses, default false.
* ``stream_chat`` threads the flag through ``_Turn`` into
  ``stream_response_with_agent(spoken_mode=…)``: the real route, ``_produce_turn`` and
  ``_turn_chunks``, over a patched streaming service that replays a recorded turn.
* Spoken: the recorded markdown turn (a header, bullets, bold, a ``**`` split across two
  deltas) comes out with none of it, every other frame unchanged and in its order. A frame
  inside an open unit waits for it, so bold on both sides of it is still stripped.
* Off: every frame byte-identical to the recorded turn.
* Discovery lists ``spoken_mode``, mapped to ``POST /api/chat`` (a route in the manifest).

DB-free: no database, model or socket.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from api import chat, chat_models
from api.chat_models import ChatRequest
from config import HOSTED_PUBLIC_API_HOST, config
from consumers.chatbot.streaming import StreamingHandler
from modules.voice.spoken_stream import spoken_text_frames
from services.mobile import discovery

ORCH = Path(__file__).resolve().parents[1]
CHAT_PY_LINES_BEFORE = 908
WS = "6a0f3c1e-2b4d-4e8f-9a1b-266007a00001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-266007a0c4a7"
USER_PK = 7
AUTO = 11
HANDLER = StreamingHandler()


def _text(delta: str) -> str:
    return HANDLER.format_aisdk_text(delta)


CHAT_ID_FRAME = HANDLER.format_aisdk_chat_id(CHAT_ID)
PROGRESS = HANDLER.format_aisdk_progress("Checking the board")
USAGE = HANDLER.format_aisdk_usage(120, 48, 168)
FINISH = HANDLER.format_aisdk_finish()

# A recorded markdown turn: a header, bold split across deltas, bullets, and a "**" whose two
# asterisks arrive in two deltas ("- the *" then "*Acme** …").
RECORDED = (
    CHAT_ID_FRAME,
    _text("## Your morning\n"),
    _text("Two things need **you"),
    _text(" today**:\n"),
    _text("- the *"),
    _text("*Acme** invoice is overdue\n"),
    PROGRESS,
    _text("- the board has three new cards."),
    USAGE,
    FINISH,
)


@pytest.fixture(autouse=True)
def _unit_ceiling(monkeypatch):
    monkeypatch.setattr(config, "VOICE_LIVE_SPEECH_UNIT_MAX_CHARS", 180)


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` with its collaborators patched; the streaming service replays
    ``RECORDED`` and keeps the kwargs it was asked with."""
    asked = []

    class _ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kwargs):
            return None

    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = StreamingHandler()

        async def stream_response_with_agent(self, **kwargs):
            asked.append(kwargs)
            for frame in RECORDED:
                yield frame

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _auto_answers(*args, **kwargs):
        return NS(agent_id=AUTO, assessment=None, session_agent=None, suggest_mission=False,
                  skip_composio=False)

    async def _inline(*, chat_id, produce, on_complete=None):
        async for chunk in produce():
            yield chunk

    monkeypatch.setattr(chat, "ChatService", _ChatService)
    monkeypatch.setattr(chat, "StreamingChatService", _StreamingService)
    monkeypatch.setattr(chat, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Queue())
    monkeypatch.setattr(chat, "get_user_id", lambda db, ctx=None: USER_PK)
    monkeypatch.setattr(chat, "_open_chat", lambda *args, **kwargs: CHAT_ID)
    monkeypatch.setattr(chat, "turn_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat, "_turn_lane", _auto_answers)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)
    return asked


def _ask(body: dict) -> list:
    """The frames ``POST /api/chat`` streams for ``body``."""
    ctx = NS(workspace_id=WS, user=NS(id="owner@prd266.test", system_role="user"))

    async def _run() -> list:
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS())
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_run())


def _body(**over) -> dict:
    return {"message": {"role": "user", "parts": [{"type": "text", "text": "What needs me?"}]}, **over}


def _drain(chunks) -> list:
    async def _source():
        for chunk in chunks:
            yield chunk

    async def _run() -> list:
        return [frame async for frame in spoken_text_frames(_source(), HANDLER.format_aisdk_text)]

    return asyncio.run(_run())


def _spoken(frames: list) -> str:
    return "".join(json.loads(frame[2:]) for frame in frames if frame.startswith("0:"))


# --------------------------------------------------------------------------- the request model


def test_the_chat_request_and_its_parts_live_in_chat_models():
    source = (ORCH / "api" / "chat.py").read_text()

    assert (chat.MessagePart, chat.ChatMessageRequest, chat.ChatRequest) == (
        chat_models.MessagePart, chat_models.ChatMessageRequest, chat_models.ChatRequest)
    for name in ("MessagePart", "ChatMessageRequest", "ChatRequest"):
        assert f"class {name}(" not in source
    assert len(source.splitlines()) < CHAT_PY_LINES_BEFORE


def test_spoken_mode_parses_and_defaults_to_false():
    assert ChatRequest(**_body()).spokenMode is False
    assert ChatRequest(**_body(spokenMode=True)).spokenMode is True


# --------------------------------------------------------------------------- the route threads it


@pytest.mark.parametrize("spoken", [True, False])
def test_the_flag_reaches_the_streaming_service(route, spoken):
    _ask(_body(spokenMode=spoken))

    (kwargs,) = route
    assert kwargs["spoken_mode"] is spoken
    assert (kwargs["chat_id"], kwargs["agent_id"], kwargs["user_id"]) == (CHAT_ID, AUTO, USER_PK)


def test_a_request_without_the_flag_is_not_spoken(route):
    _ask(_body())

    (kwargs,) = route
    assert kwargs["spoken_mode"] is False


# --------------------------------------------------------------------------- spoken: no markdown


def test_a_spoken_turn_comes_out_with_no_markdown(route):
    frames = _ask(_body(spokenMode=True))

    assert frames == [
        CHAT_ID_FRAME,
        _text("Your morning."),
        _text(" Two things need you today:"),
        _text(" the Acme invoice is overdue."),
        PROGRESS,
        _text(" the board has three new cards."),
        USAGE,
        FINISH,
    ]
    spoken = _spoken(frames)
    assert "*" not in spoken and "#" not in spoken and "\n" not in spoken and "- " not in spoken
    assert "the Acme invoice is overdue" in spoken


def test_every_other_frame_keeps_its_bytes_and_its_order(route):
    frames = _ask(_body(spokenMode=True))

    assert [f for f in frames if not f.startswith("0:")] == [f for f in RECORDED if not f.startswith("0:")]
    assert frames[-1] == FINISH


def test_a_frame_inside_an_open_unit_waits_for_it():
    frames = _drain([_text("Your **inv"), PROGRESS, _text("oice** is paid"), FINISH])

    assert frames == [_text("Your invoice is paid"), PROGRESS, FINISH]


def test_a_unit_with_nothing_speakable_emits_no_frame():
    frames = _drain([_text("---\n"), FINISH])  # a horizontal rule: nothing to say

    assert frames == [FINISH]


# --------------------------------------------------------------------------- off: unchanged


def test_spoken_mode_off_leaves_every_frame_byte_identical(route):
    assert _ask(_body(spokenMode=False)) == list(RECORDED)


# --------------------------------------------------------------------------- discovery


def test_discovery_lists_spoken_mode_on_the_chat_route(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())

    document = discovery.build_discovery("http://192.168.1.20:8000/")

    assert "spoken_mode" in document["capabilities"]
    assert discovery.CAPABILITY_ROUTES["spoken_mode"] == ("POST", "/api/chat")
    assert {"method": "POST", "path": "/api/chat"} in manifest["routes"]
