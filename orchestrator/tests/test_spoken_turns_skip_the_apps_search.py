"""Spoken turns skip the connected-apps search (voice latency C-2).

A turn sent with ``spokenMode`` ran the Composio search (bounded at 20 s) before its first model
call, like a typed turn. With ``SPOKEN_MODE_SKIP_COMPOSIO`` on it does not, as Auto Live does not.

* ``voice_skips_apps`` is true only when the spoken rule is the reason: a spoken turn, the switch
  on, and a lane that would have searched.
* ``spoken_skips_apps`` is what the turn passes as ``skip_composio``. Switch off: the lane's own
  value for every input. It logs one line, with no text of the turn, when the spoken rule skips.
* Through the real route, ``_produce_turn`` and ``_turn_chunks``: a spoken request reaches the
  service with ``skip_composio=True``; a typed one, and a spoken one with the switch off, with the
  lane's value. An agent chosen by ``agentId`` is treated the same.
* The real ``_inject_composio_tools`` with ``skip_composio=True`` makes no lookup, hands back the
  tools it was given (``composio_execute`` included) and adds no message.
* Auto Live keeps its own switch: ``api/voice_retell.py`` reads neither the module nor the setting.

DB-free: no database, model, SDK or socket.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from api import chat, chat_dispatch
from api.chat_models import ChatRequest
from config import config
from consumers.chatbot import spoken_apps
from consumers.chatbot.streaming import StreamingHandler

ORCH = Path(__file__).resolve().parents[1]
WS = "6a0f3c1e-2b4d-4e8f-9a1b-c20000000001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-c2000000c4a7"
USER_PK = 7
AUTO = 11
CHOSEN = 23
ASKED = "Email John the notes"
SPOKEN_APPS_LOGGER = "consumers.chatbot.spoken_apps"
PAIRS = [(False, False), (False, True), (True, False), (True, True)]  # (the lane skips, spoken)
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}
LIST_AGENTS = {"type": "function", "function": {
    "name": "platform_list_agents", "parameters": {"type": "object", "properties": {}},
}}


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", True)


@pytest.fixture
def switch_off(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", False)


# --------------------------------------------------------------------------- the setting


def test_the_switch_is_off_unless_the_environment_turns_it_on():
    source = (ORCH / "config.py").read_text()

    assert 'os.getenv("SPOKEN_MODE_SKIP_COMPOSIO", "false").strip().lower() == "true"' in source
    assert isinstance(config.SPOKEN_MODE_SKIP_COMPOSIO, bool)


# --------------------------------------------------------------------------- the rule


def test_the_spoken_rule_is_the_reason_only_on_a_spoken_turn_the_lane_would_have_searched(switch_on):
    assert spoken_apps.voice_skips_apps(False, True) is True
    assert spoken_apps.voice_skips_apps(True, True) is False  # the lane skipped it already
    assert spoken_apps.voice_skips_apps(False, False) is False  # typed
    assert spoken_apps.voice_skips_apps(True, False) is False


@pytest.mark.parametrize("lane_skips,spoken", PAIRS)
def test_switch_off_the_spoken_rule_is_never_the_reason(switch_off, lane_skips, spoken):
    assert spoken_apps.voice_skips_apps(lane_skips, spoken) is False


@pytest.mark.parametrize("lane_skips,spoken", PAIRS)
def test_switch_off_the_turn_passes_the_lanes_own_value(switch_off, caplog, lane_skips, spoken):
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        assert spoken_apps.spoken_skips_apps(lane_skips, spoken) is lane_skips

    assert [r for r in caplog.records if r.name == SPOKEN_APPS_LOGGER] == []


@pytest.mark.parametrize("lane_skips,spoken,skips", [
    (False, True, True),  # the spoken rule
    (True, True, True),  # the lane's own skip
    (False, False, False),  # typed: the lane's value
    (True, False, True),
])
def test_switch_on_a_spoken_turn_skips_and_a_typed_turn_keeps_the_lanes_value(switch_on, lane_skips, spoken, skips):
    assert spoken_apps.spoken_skips_apps(lane_skips, spoken) is skips


def test_the_spoken_rule_logs_one_line_and_only_when_it_is_the_reason(switch_on, caplog):
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        spoken_apps.spoken_skips_apps(False, True)
        said = [r.getMessage() for r in caplog.records if r.name == SPOKEN_APPS_LOGGER]
        caplog.clear()
        spoken_apps.spoken_skips_apps(True, True)
        spoken_apps.spoken_skips_apps(False, False)
        spoken_apps.spoken_skips_apps(True, False)
        quiet = [r for r in caplog.records if r.name == SPOKEN_APPS_LOGGER]

    assert len(said) == 1 and "SPOKEN_MODE_SKIP_COMPOSIO" in said[0]
    assert quiet == []


def test_the_route_reaches_the_rule_through_the_dispatch_module():
    assert chat_dispatch.spoken_skips_apps is spoken_apps.spoken_skips_apps
    assert chat.apps_skipped is spoken_apps.apps_skipped  # C-3: the route asks through apps_skipped
    assert "spoken_skips_apps" in chat_dispatch.__all__


# --------------------------------------------------------------------------- through the route


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` with its collaborators patched (the fixture of
    ``test_prd266_chat_spoken_mode.py``); the streaming service keeps the kwargs it was asked
    with, and the lane says whether it skips the apps search itself (``lane.skips``)."""
    asked = []
    lane = NS(skips=False, calls=[])
    handler = StreamingHandler()

    class _ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kwargs):
            return None

    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = handler

        async def stream_response_with_agent(self, **kwargs):
            asked.append(kwargs)
            yield handler.format_aisdk_text("Sent.")
            yield handler.format_aisdk_finish()

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _the_lane(db, ctx, request, *args, **kwargs):
        lane.calls.append(request.agentId)
        return NS(agent_id=request.agentId or AUTO, assessment=None, session_agent=None,
                  suggest_mission=False, skip_composio=lane.skips)

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
    monkeypatch.setattr(chat, "_turn_lane", _the_lane)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)
    return NS(asked=asked, lane=lane)


def _ask(body: dict) -> list:
    """The frames ``POST /api/chat`` streams for ``body``."""
    ctx = NS(workspace_id=WS, user=NS(id="owner@c2.test", system_role="user"))

    async def _run() -> list:
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS(rollback=lambda: None))
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_run())


def _body(**over) -> dict:
    return {"message": {"role": "user", "parts": [{"type": "text", "text": ASKED}]}, **over}


def test_a_spoken_request_reaches_the_service_skipping_the_apps_search(route, switch_on):
    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is True
    assert (kwargs["spoken_mode"], kwargs["agent_id"]) == (True, AUTO)


@pytest.mark.parametrize("lane_skips", [False, True])
def test_a_typed_request_reaches_the_service_with_the_lanes_value(route, switch_on, lane_skips):
    route.lane.skips = lane_skips

    _ask(_body(spokenMode=False))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is lane_skips
    assert kwargs["spoken_mode"] is False


def test_a_request_without_the_flag_keeps_the_lanes_value(route, switch_on):
    _ask(_body())

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is False


@pytest.mark.parametrize("lane_skips", [False, True])
def test_switch_off_a_spoken_request_reaches_the_service_with_the_lanes_value(route, switch_off, lane_skips):
    route.lane.skips = lane_skips

    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is lane_skips
    assert kwargs["spoken_mode"] is True


def test_a_spoken_turn_the_lane_already_skips_stays_skipped(route, switch_on):
    route.lane.skips = True

    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is True


def test_a_spoken_turn_for_a_chosen_agent_skips_the_search_too(route, switch_on):
    _ask(_body(spokenMode=True, agentId=CHOSEN))

    (kwargs,) = route.asked
    assert route.lane.calls == [CHOSEN]
    assert (kwargs["agent_id"], kwargs["skip_composio"]) == (CHOSEN, True)


# --------------------------------------------------------------------------- what the skip does


@pytest.fixture
def lookups(monkeypatch):
    """Every Composio lookup a turn asks for is counted here and then fails."""
    made = []

    async def _lookup(read, *, step="", **kwargs):
        made.append(step)
        raise RuntimeError("no Composio lookup was expected")

    monkeypatch.setattr("core.composio.off_loop.composio_lookup", _lookup)
    return made


def _inject(skip_composio: bool):
    """The real ``_inject_composio_tools`` on a bare service. Returns (messages, tools given,
    tools returned, the search's result)."""
    from consumers.chatbot.service import StreamingChatService

    service = StreamingChatService.__new__(StreamingChatService)
    service.db, service.workspace_id = MagicMock(name="the turn's session"), WS
    messages = [{"role": "system", "content": "identity"}, {"role": "user", "content": ASKED}]
    given = [LIST_AGENTS, COMPOSIO_EXECUTE]
    tools, found = asyncio.run(service._inject_composio_tools(
        messages, given, ASKED, AUTO, None, skip_composio, None))
    return messages, given, tools, found


def test_a_skipped_turn_makes_no_lookup_and_keeps_the_generic_tool(lookups):
    messages, given, tools, found = _inject(skip_composio=True)

    assert lookups == []
    assert tools is given and tools == [LIST_AGENTS, COMPOSIO_EXECUTE]
    assert found is None
    assert messages == [{"role": "system", "content": "identity"}, {"role": "user", "content": ASKED}]


def test_a_turn_that_does_not_skip_asks_for_the_tool_search(lookups):
    """The other side of the test above: the counted lookup is the one the turn would make."""
    _, _, tools, found = _inject(skip_composio=False)

    assert lookups == [f"chat turn (agent {AUTO}): tool search"]
    assert (tools, found) == ([LIST_AGENTS, COMPOSIO_EXECUTE], None)  # the failed lookup costs no tool


# --------------------------------------------------------------------------- Auto Live


def test_auto_live_keeps_its_own_switch():
    source = (ORCH / "api" / "voice_retell.py").read_text()

    assert "skip_composio=bool(config.VOICE_LIVE_SKIP_COMPOSIO)" in source
    assert "spoken_apps" not in source
    assert "SPOKEN_MODE_SKIP_COMPOSIO" not in source
