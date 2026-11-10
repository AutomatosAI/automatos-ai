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
* Voice latency B-5: with the narration formatter, a narration or retraction frame closes its
  round's open text and carries the speechified text that was sent for it, so the client's
  exact-match removal takes it off; without the formatter the frame passes byte-identical.
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
from consumers.chatbot import on_screen
from consumers.chatbot.streaming import StreamingHandler
from modules.voice import spoken_stream
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
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS(rollback=lambda: None))
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


# --------------------------------------------------------------------------- B-5: narration frames

TOOL_START = HANDLER.format_aisdk_tool_start("call-1", "platform_list_tasks")
LEAD_IN = "Let me check that:"
LISTED = "**Checking** the board:\n\n- tasks\n- people"
DRAFT = "I've approved the mission.\n\nIt's now running."


def _heard(chunks, format_narration=HANDLER.narration_frame) -> list:
    """``_drain`` with the narration formatter: what a spoken turn sends with B-5's switch on."""
    async def _source():
        for chunk in chunks:
            yield chunk

    async def _run() -> list:
        return [frame async for frame in
                spoken_text_frames(_source(), HANDLER.format_aisdk_text, format_narration)]

    return asyncio.run(_run())


def _marks(frames: list) -> list:
    """The ``data`` of every narration frame, in order."""
    return [json.loads(frame[2:])["data"] for frame in frames
            if frame.startswith(spoken_stream.NARRATION_FRAME)]


def _without(content: str, text: str) -> str:
    """The clients' ``withoutNarration`` (frontend/lib/chat/narration.ts): ``content`` without
    the last occurrence of ``text``; unchanged when it is not there."""
    at = content.rfind(text)
    return content if not text or at < 0 else content[:at] + content[at + len(text):]


def _on_the_phone(frames: list) -> str:
    """What a client holds after the frames: text appended, each narration frame's text removed."""
    content = ""
    for frame in frames:
        if frame.startswith("0:"):
            content += json.loads(frame[2:])
        elif frame.startswith(spoken_stream.NARRATION_FRAME):
            content = _without(content, json.loads(frame[2:])["data"]["text"])
    return content


def test_narration_is_spoken_when_its_round_ends():
    frames = _heard([_text(LEAD_IN), HANDLER.narration_frame(LEAD_IN), TOOL_START,
                     _text("You have three open tasks."), FINISH])

    assert frames == [_text(LEAD_IN), HANDLER.narration_frame(LEAD_IN), TOOL_START,
                      _text(" You have three open tasks."), FINISH]
    assert _on_the_phone(frames) == " You have three open tasks."


def test_a_narration_frame_names_the_text_that_was_sent():
    deltas = ("**Check", "ing** the board:\n\n- ta", "sks\n- peo", "ple")
    assert "".join(deltas) == LISTED

    frames = _heard([*(_text(delta) for delta in deltas), HANDLER.narration_frame(LISTED),
                     TOOL_START, _text("Three tasks are open."), FINISH])

    (mark,) = _marks(frames)
    assert mark == {"text": "Checking the board: tasks. people"}
    assert mark["text"] in _spoken(frames)
    assert _spoken(frames) == "Checking the board: tasks. people Three tasks are open."
    left = _on_the_phone(frames)
    assert left == " Three tasks are open."
    assert not any(word in left for word in ("Checking", "board", "tasks.", "people"))


def test_a_retracted_two_paragraph_draft_can_be_taken_off():
    frames = _heard([_text("I've approved the mission.\n\nIt's now"), _text(" running."),
                     HANDLER.narration_frame(DRAFT, True),
                     _text("I haven't approved the mission yet."), FINISH])

    (mark,) = _marks(frames)
    assert mark == {"text": "I've approved the mission. It's now running.", "retracted": True}
    assert mark["text"] in _spoken(frames)
    assert _on_the_phone(frames) == " I haven't approved the mission yet."
    assert frames[-2:] == [_text(" I haven't approved the mission yet."), FINISH]


def test_a_retraction_sent_after_its_retry_matches_when_the_draft_ended_its_sentence():
    frames = _heard([_text("I've sent it."), _text(" Nothing was sent."),
                     HANDLER.narration_frame("I've sent it.", True), FINISH])

    assert frames == [_text("I've sent it."), _text(" Nothing was sent."),
                      HANDLER.narration_frame("I've sent it.", True), FINISH]
    assert _on_the_phone(frames) == " Nothing was sent."


def test_a_frame_for_text_that_said_nothing_is_not_sent():
    frames = _heard([_text("```\n"), HANDLER.narration_frame("```\n"), FINISH])  # a lone code fence

    assert frames == [FINISH]


def test_a_frame_with_no_sent_run_is_speechified_as_a_best_effort():
    frames = _heard([_text("Something else."), HANDLER.narration_frame("**Checking** now"), FINISH])

    assert frames == [_text("Something else."), HANDLER.narration_frame("Checking now"), FINISH]


def test_a_frame_that_is_not_narration_behaves_as_before_with_the_formatter():
    chunks = [_text("Your **inv"), PROGRESS, _text("oice** is paid"), FINISH]

    assert _heard(chunks) == _drain(chunks) == [_text("Your invoice is paid"), PROGRESS, FINISH]


def test_without_the_formatter_a_narration_frame_passes_unchanged():
    raw = HANDLER.narration_frame("**Let me** check that:")
    frames = _drain([_text("**Let me** check that:"), raw, _text(" You have three open tasks."), FINISH])

    # The open text waits for its sentence and the frame waits behind it, byte-identical.
    assert frames == [_text("Let me check that: You have three open tasks."), raw, FINISH]
    assert _heard([_text(LEAD_IN), raw, FINISH], None) == [_text(LEAD_IN), raw, FINISH]


def test_the_narration_prefix_is_the_chat_services():
    assert spoken_stream.NARRATION_FRAME == on_screen.NARRATION_FRAME
    assert StreamingHandler().narration_frame("x").startswith(spoken_stream.NARRATION_FRAME)
    assert StreamingHandler().narration_frame("x", True).startswith(spoken_stream.NARRATION_FRAME)


@pytest.mark.parametrize("switch", [True, False])
def test_the_route_passes_the_formatter_only_when_the_switch_is_on(route, monkeypatch, switch):
    raw = HANDLER.narration_frame("**Checking** the board:")
    turn = (CHAT_ID_FRAME, _text("**Checking** the board:"), raw, PROGRESS,
            _text(" Three tasks are open."), FINISH)
    monkeypatch.setitem(globals(), "RECORDED", turn)  # the fixture's service replays this global
    monkeypatch.setattr(config, "SPOKEN_MODE_NARRATION_AS_SPOKEN", switch)

    frames = _ask(_body(spokenMode=True))

    if switch:
        assert frames == [CHAT_ID_FRAME, _text("Checking the board:"),
                          HANDLER.narration_frame("Checking the board:"), PROGRESS,
                          _text(" Three tasks are open."), FINISH]
    else:
        assert frames == [CHAT_ID_FRAME, _text("Checking the board: Three tasks are open."),
                          raw, PROGRESS, FINISH]


def test_the_switch_is_on_by_default_and_named_on_the_config_surface():
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())

    assert "SPOKEN_MODE_NARRATION_AS_SPOKEN" in json.dumps(surface)
    assert 'os.getenv("SPOKEN_MODE_NARRATION_AS_SPOKEN", "true")' in (ORCH / "config.py").read_text()


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
