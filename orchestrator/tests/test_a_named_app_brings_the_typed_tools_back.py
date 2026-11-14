"""Naming a connected app brings the typed tools back (voice latency C-3).

With ``SPOKEN_MODE_SKIP_COMPOSIO`` on, a spoken turn skips the connected-apps search (C-2). With
``SPOKEN_MODE_APPS_WHEN_NAMED`` on too, a turn the spoken rule itself skipped runs the search
after all when the person's own words name an app Auto may use.

* ``names_an_app`` matches whole consecutive words only: "this lack of focus" does not name
  Slack, "one drive" names OneDrive. A real word that is also an app ("notion") matches.
* ``apps_skipped`` is what the turn passes as ``skip_composio``. The app list is read only when
  the spoken rule is the reason, no agent was chosen and the lane is not ASSIGN: never on a
  typed turn, with C-2 off, with C-3 off, with ``agentId``, or on a turn the lane already skips.
* The read is bounded: one slower than the bound, or one that raises, reads as not named and
  logs one warning. No log line carries the person's words.
* ``allowed_apps`` reads the search's own list on a session of its own and closes it.
* Through the real route: a spoken Auto turn naming a connected app reaches the service with
  ``skip_composio=False``.

DB-free: no database, model, SDK or socket.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from api import chat, chat_dispatch
from api.chat_models import ChatRequest
from config import config
from consumers.chatbot import spoken_apps
from consumers.chatbot.auto import Action
from consumers.chatbot.streaming import StreamingHandler

ORCH = Path(__file__).resolve().parents[1]
WS = "6a0f3c1e-2b4d-4e8f-9a1b-c30000000001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-c3000000c4a7"
USER_PK = 7
AUTO = 11
CHOSEN = 23
NAMED = "Check my Gmail"
NOT_NAMED = "What's on my calendar"
CONNECTED = ["GMAIL", "GOOGLECALENDAR"]
SPOKEN_APPS_LOGGER = "consumers.chatbot.spoken_apps"
BOUND_SECONDS = 0.05
HANG_SECONDS = 5.0
WELL_UNDER_THE_HANG_SECONDS = 2.0


@pytest.fixture
def both_on(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", True)
    monkeypatch.setattr(config, "SPOKEN_MODE_APPS_WHEN_NAMED", True)


@pytest.fixture
def apps(monkeypatch):
    """The app list the check reads: ``apps.connected``; every read is kept in ``apps.reads``."""
    state = NS(connected=list(CONNECTED), reads=[])

    def _allowed(agent_id, workspace_id):
        state.reads.append((agent_id, workspace_id))
        return list(state.connected)

    monkeypatch.setattr(spoken_apps, "allowed_apps", _allowed)
    return state


def _lane(skips: bool = False, assessment=None) -> NS:
    return NS(agent_id=AUTO, skip_composio=skips, assessment=assessment)


def _skipped(lane, *, spoken=True, agent_chosen=False, text=NAMED) -> bool:
    return asyncio.run(spoken_apps.apps_skipped(
        lane, spoken=spoken, agent_chosen=agent_chosen, workspace_id=WS, text=text))


def _said(caplog, level: int) -> list:
    return [r.getMessage() for r in caplog.records if r.name == SPOKEN_APPS_LOGGER and r.levelno == level]


# --------------------------------------------------------------------------- the setting


def test_the_switch_is_off_unless_the_environment_turns_it_on():
    source = (ORCH / "config.py").read_text()

    assert 'os.getenv("SPOKEN_MODE_APPS_WHEN_NAMED", "false").strip().lower() == "true"' in source
    assert isinstance(config.SPOKEN_MODE_APPS_WHEN_NAMED, bool)


def test_the_check_is_bounded_at_half_a_second():
    assert spoken_apps.APP_CHECK_TIMEOUT_SECONDS == 0.5


# --------------------------------------------------------------------------- the name rule


@pytest.mark.parametrize("text,app", [
    ("check my Gmail", "GMAIL"),
    ("what's in google calendar today", "GOOGLECALENDAR"),
    ("put it on one drive", "ONEDRIVE"),
    ("Post it in Slack.", "SLACK"),
    ("what's in google calendar today", "GOOGLE_CALENDAR"),  # the app's name is squashed too
])
def test_an_app_said_as_whole_words_is_named(text, app):
    assert spoken_apps.names_an_app(text, [app]) is True


@pytest.mark.parametrize("text,app", [
    ("what's on my calendar", "GOOGLECALENDAR"),
    ("this lack of focus", "SLACK"),  # "thislack" is two words, not Slack
    ("open the canvas", "CANVA"),
    ("a nonlinear plan", "LINEAR"),
    ("line around the block", "LINEAR"),
    ("two stripes", "STRIPE"),
    ("", "GMAIL"),
])
def test_a_word_that_only_contains_an_apps_name_does_not_name_it(text, app):
    assert spoken_apps.names_an_app(text, [app]) is False


def test_no_apps_name_nothing():
    assert spoken_apps.names_an_app("check my Gmail", []) is False


def test_a_three_letter_app_name_never_matches():
    assert spoken_apps.names_an_app("put it in box", ["BOX"]) is False


def test_a_name_longer_than_three_words_is_not_looked_for():
    assert spoken_apps.names_an_app("a b c d", ["ABCD"]) is False
    assert spoken_apps.names_an_app("ab c d", ["ABCD"]) is True


@pytest.mark.parametrize("text,app", [("I was slack today", "SLACK"), ("I have no notion", "NOTION")])
def test_a_real_word_that_is_also_an_app_names_it(text, app):
    """Documented: whole-word matching cannot tell these apart; the cost is one slower turn."""
    assert spoken_apps.names_an_app(text, [app]) is True


def test_one_named_app_among_several_is_enough():
    assert spoken_apps.names_an_app("check my Gmail", ["SLACK", "GMAIL", "NOTION"]) is True


# --------------------------------------------------------------------------- what the turn passes


def test_a_named_app_brings_the_search_back(both_on, apps):
    assert _skipped(_lane()) is False
    assert apps.reads == [(AUTO, WS)]


def test_a_turn_that_names_no_app_stays_skipped(both_on, apps):
    assert _skipped(_lane(), text=NOT_NAMED) is True
    assert apps.reads == [(AUTO, WS)]


def test_no_allowed_apps_names_nothing(both_on, apps):
    """The local edition with no Composio: the list is empty."""
    apps.connected = []

    assert _skipped(_lane()) is True


@pytest.mark.parametrize("lane_skips", [False, True])
def test_a_typed_turn_passes_the_lanes_value_and_reads_no_apps(both_on, apps, lane_skips):
    assert _skipped(_lane(skips=lane_skips), spoken=False) is lane_skips
    assert apps.reads == []


@pytest.mark.parametrize("lane_skips", [False, True])
def test_with_the_skip_off_a_spoken_turn_passes_the_lanes_value_and_reads_no_apps(monkeypatch, apps, lane_skips):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", False)
    monkeypatch.setattr(config, "SPOKEN_MODE_APPS_WHEN_NAMED", True)

    assert _skipped(_lane(skips=lane_skips)) is lane_skips
    assert apps.reads == []


def test_a_turn_the_lane_already_skips_is_never_unskipped(both_on, apps):
    """A platform-hint or recall turn: the spoken rule is not the reason, so no check."""
    assert _skipped(_lane(skips=True)) is True
    assert apps.reads == []


def test_a_chosen_agent_gets_no_check(both_on, apps):
    assert _skipped(_lane(), agent_chosen=True) is True
    assert apps.reads == []


def test_a_hand_off_gets_no_check(both_on, apps):
    assert _skipped(_lane(assessment=NS(action=Action.ASSIGN))) is True
    assert apps.reads == []


def test_autos_own_answer_with_an_assessment_is_checked(both_on, apps):
    assert _skipped(_lane(assessment=NS(action=Action.RESPOND))) is False
    assert apps.reads == [(AUTO, WS)]


def test_with_the_switch_off_naming_an_app_changes_nothing(monkeypatch, apps):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", True)
    monkeypatch.setattr(config, "SPOKEN_MODE_APPS_WHEN_NAMED", False)

    assert _skipped(_lane()) is True
    assert apps.reads == []


# --------------------------------------------------------------------------- the bound


def test_a_check_slower_than_its_bound_reads_as_not_named(both_on, monkeypatch, caplog):
    released = threading.Event()

    def _slow(agent_id, workspace_id):
        released.wait(HANG_SECONDS)
        return list(CONNECTED)

    async def _run():
        started = time.monotonic()
        try:
            skipped = await spoken_apps.apps_skipped(
                _lane(), spoken=True, agent_chosen=False, workspace_id=WS, text=NAMED)
            return skipped, time.monotonic() - started
        finally:
            released.set()  # free the read's thread: the loop waits for it on its way out

    monkeypatch.setattr(spoken_apps, "allowed_apps", _slow)
    monkeypatch.setattr(spoken_apps, "APP_CHECK_TIMEOUT_SECONDS", BOUND_SECONDS)
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        skipped, took = asyncio.run(_run())

    assert skipped is True
    assert BOUND_SECONDS <= took < WELL_UNDER_THE_HANG_SECONDS
    assert len(_said(caplog, logging.WARNING)) == 1


def test_a_check_that_raises_reads_as_not_named(both_on, monkeypatch, caplog):
    def _broken(agent_id, workspace_id):
        raise RuntimeError("connection refused for owner@c3.test")

    monkeypatch.setattr(spoken_apps, "allowed_apps", _broken)
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        skipped = _skipped(_lane())

    warnings = _said(caplog, logging.WARNING)
    assert skipped is True
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0] and "owner@c3.test" not in warnings[0]


# --------------------------------------------------------------------------- what it logs


def test_a_check_logs_one_line_with_a_flag_and_a_duration_and_none_of_the_words(both_on, apps, caplog):
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        _skipped(_lane())

    said = [r.getMessage() for r in caplog.records if r.name == SPOKEN_APPS_LOGGER]
    assert len(said) == 1
    assert said[0].startswith("[spoken] apps check: named=True check_ms=")
    assert "gmail" not in said[0].lower()


def test_a_turn_that_names_no_app_logs_the_check_and_then_the_skip(both_on, apps, caplog):
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        _skipped(_lane(), text=NOT_NAMED)

    said = [r.getMessage() for r in caplog.records if r.name == SPOKEN_APPS_LOGGER]
    assert len(said) == 2
    assert said[0].startswith("[spoken] apps check: named=False check_ms=")
    assert "SPOKEN_MODE_SKIP_COMPOSIO" in said[1]
    assert all("calendar" not in line.lower() for line in said)


def test_a_turn_with_no_check_logs_no_check_line(both_on, apps, caplog):
    with caplog.at_level(logging.INFO, logger=SPOKEN_APPS_LOGGER):
        _skipped(_lane(), agent_chosen=True)
        _skipped(_lane(skips=True))
        _skipped(_lane(), spoken=False)

    said = [r.getMessage() for r in caplog.records if r.name == SPOKEN_APPS_LOGGER]
    assert [line for line in said if "apps check" in line] == []


# --------------------------------------------------------------------------- the app list


@pytest.fixture
def resolver(monkeypatch):
    """The search's own resolver, answered here; the session the read opens is counted."""
    import modules.tools.services.composio_tool_service as tool_service

    state = NS(asked=[], closed=0, opened=[], fails=False)

    def _session():
        def _close():
            state.closed += 1

        session = NS(close=_close)
        state.opened.append(session)
        return session

    def _resolve(self, agent_id, workspace_id):
        state.asked.append((self.db, agent_id, workspace_id))
        if state.fails:
            raise RuntimeError("the resolver failed")
        return ["GMAIL", "SLACK"]

    monkeypatch.setattr("core.database.database.SessionLocal", _session)
    monkeypatch.setattr(tool_service.ComposioToolService, "_resolve_allowed_apps", _resolve)
    return state


def test_the_app_list_is_the_searchs_own_on_a_session_it_closes(resolver):
    assert spoken_apps.allowed_apps(AUTO, WS) == ["GMAIL", "SLACK"]

    (session,) = resolver.opened
    assert resolver.asked == [(session, AUTO, WS)]
    assert resolver.closed == 1


def test_a_failed_read_still_closes_its_session(resolver):
    resolver.fails = True

    with pytest.raises(RuntimeError):
        spoken_apps.allowed_apps(AUTO, WS)

    assert len(resolver.opened) == 1 and resolver.closed == 1


# --------------------------------------------------------------------------- the wiring


def test_the_route_reaches_the_check_through_the_dispatch_module():
    assert chat_dispatch.apps_skipped is spoken_apps.apps_skipped
    assert chat.apps_skipped is spoken_apps.apps_skipped
    assert "apps_skipped" in chat_dispatch.__all__


def test_a_turn_is_not_a_chosen_agents_unless_it_says_so():
    (field,) = [f for f in dataclasses.fields(chat._Turn) if f.name == "agent_chosen"]

    assert field.default is False


def test_auto_live_reads_neither_the_module_nor_the_setting():
    source = (ORCH / "api" / "voice_retell.py").read_text()

    assert "spoken_apps" not in source
    assert "SPOKEN_MODE_APPS_WHEN_NAMED" not in source


# --------------------------------------------------------------------------- through the route


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` with its collaborators patched (the fixture of
    ``test_spoken_turns_skip_the_apps_search.py``); the streaming service keeps the kwargs it
    was asked with, and the lane says whether it skips the apps search itself (``lane.skips``)."""
    asked = []
    lane = NS(skips=False)
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
            yield handler.format_aisdk_text("Done.")
            yield handler.format_aisdk_finish()

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _the_lane(db, ctx, request, *args, **kwargs):
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
    ctx = NS(workspace_id=WS, user=NS(id="owner@c3.test", system_role="user"))

    async def _run() -> list:
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS(rollback=lambda: None))
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_run())


def _body(text: str = NAMED, **over) -> dict:
    return {"message": {"role": "user", "parts": [{"type": "text", "text": text}]}, **over}


def test_a_spoken_turn_naming_a_connected_app_reaches_the_service_searching(route, both_on, apps):
    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is False
    assert (kwargs["spoken_mode"], kwargs["agent_id"]) == (True, AUTO)
    assert apps.reads == [(AUTO, WS)]


def test_a_spoken_turn_naming_no_app_reaches_the_service_skipping(route, both_on, apps):
    _ask(_body(NOT_NAMED, spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is True
    assert apps.reads == [(AUTO, WS)]


def test_a_spoken_turn_for_a_chosen_agent_is_never_checked(route, both_on, apps):
    _ask(_body(spokenMode=True, agentId=CHOSEN))

    (kwargs,) = route.asked
    assert (kwargs["agent_id"], kwargs["skip_composio"]) == (CHOSEN, True)
    assert apps.reads == []


@pytest.mark.parametrize("lane_skips", [False, True])
def test_a_typed_turn_naming_an_app_keeps_the_lanes_value(route, both_on, apps, lane_skips):
    route.lane.skips = lane_skips

    _ask(_body(spokenMode=False))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is lane_skips
    assert apps.reads == []


def test_with_the_switch_off_a_spoken_turn_naming_an_app_stays_skipped(route, monkeypatch, apps):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", True)
    monkeypatch.setattr(config, "SPOKEN_MODE_APPS_WHEN_NAMED", False)

    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert kwargs["skip_composio"] is True
    assert apps.reads == []
