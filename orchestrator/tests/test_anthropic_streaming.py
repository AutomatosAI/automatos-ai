"""Anthropic streaming (voice latency B): the switch and the spoken-surface flag (B-1).

``ANTHROPIC_STREAMING`` chooses which turns' Anthropic calls hand their text over as it is
generated: ``off`` (the default), ``quick-ask``, ``spoken`` or ``all``. The client knows a turn's
surface from a context variable (``core.llm.request_scope``), set where the turn starts.

* The setting is ``off`` by default and is named in the config surface.
* Outside any turn the flag is ``""``. ``set_spoken_turn`` hands back what was there before.
* A turn sent with ``spoken_mode`` says ``quick-ask`` to its model calls, also from a task the turn
  creates (the service's model calls run in tasks). A turn that is read says ``""``.
* A spoken turn started inside Auto Live stays ``live``; a turn that is read inside it says ``""``.
* The flag is put back when the turn ends: exhausted, raised, or closed early. A service built
  without an agent factory is marked and put back the same way.
* Auto Live marks its turn ``live`` around its frame loop and puts the flag back in a ``finally``.

DB-free: no database, model, SDK or socket.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from config import config
from consumers.chatbot import spoken_model as sm
from core.llm import request_scope
from core.llm.request_scope import LIVE, QUICK_ASK, set_spoken_turn, spoken_turn

ORCH = Path(__file__).resolve().parents[1]
AUTO = 337
USER_PK = 7
CHAT_ID = "chat-1"
ASKED = "Read me what needs me today"
NO_SURFACE = ""
BROKE = "the turn broke"


def _chat(*, factory=True):
    """The service the wrapper is given: with a stand-in agent factory, or built without one."""
    return NS(agent_factory=NS(name="the service's own factory") if factory else None)


async def _surface_in_a_task():
    """The surface a model call sees: the service makes its calls in tasks the turn creates."""
    async def _model_call():
        return spoken_turn()

    return await asyncio.create_task(_model_call())


async def _stub_turn(chat, chat_id, messages, agent_id, user_id, **kwargs):
    """A stand-in turn of two frames: each is the surface a task created by the turn sees."""
    yield await _surface_in_a_task()
    yield await _surface_in_a_task()


async def _raising_turn(chat, chat_id, messages, agent_id, user_id, **kwargs):
    yield await _surface_in_a_task()
    raise RuntimeError(BROKE)


def _wrapped_turn(turn=_stub_turn, *, chat=None, **kwargs):
    """One turn through ``answers_on_the_turns_model``, as an async generator not yet started."""
    wrapped = sm.answers_on_the_turns_model(turn)
    messages = [{"role": "user", "content": ASKED}]
    return wrapped(chat or _chat(), CHAT_ID, messages, AUTO, USER_PK, **kwargs)


async def _surfaces(**kwargs):
    return [surface async for surface in _wrapped_turn(**kwargs)]


# --------------------------------------------------------------------------- the switch


def test_the_switch_is_off_by_default_and_named_in_the_config_surface(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_STREAMING", raising=False)
    source = (ORCH / "config.py").read_text(encoding="utf-8")
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))

    assert 'ANTHROPIC_STREAMING: str = os.getenv("ANTHROPIC_STREAMING", "off").strip().lower()' in source
    assert isinstance(config.ANTHROPIC_STREAMING, str)
    assert "ANTHROPIC_STREAMING" in json.dumps(surface)


# --------------------------------------------------------------------------- the flag


def test_no_surface_outside_a_turn_and_set_hands_back_what_was_there():
    async def run():
        seen = [spoken_turn()]
        seen.append(set_spoken_turn(LIVE))
        seen.append(spoken_turn())
        seen.append(set_spoken_turn(None))
        seen.append(spoken_turn())
        return seen

    assert asyncio.run(run()) == [NO_SURFACE, NO_SURFACE, LIVE, LIVE, NO_SURFACE]
    assert (QUICK_ASK, LIVE) == ("quick-ask", "live")


def test_a_quick_ask_says_so_to_its_model_calls():
    assert asyncio.run(_surfaces(spoken_mode=True)) == [QUICK_ASK, QUICK_ASK]
    assert asyncio.run(_surfaces()) == [NO_SURFACE, NO_SURFACE]
    assert asyncio.run(_surfaces(spoken_mode=False)) == [NO_SURFACE, NO_SURFACE]


def test_an_auto_live_turn_stays_live():
    async def run(**kwargs):
        set_spoken_turn(LIVE)
        return await _surfaces(**kwargs)

    assert asyncio.run(run(spoken_mode=True)) == [LIVE, LIVE]
    assert asyncio.run(run()) == [NO_SURFACE, NO_SURFACE]


@pytest.mark.parametrize("outside", [NO_SURFACE, LIVE])
@pytest.mark.parametrize("spoken", [True, False])
def test_the_flag_is_put_back_when_the_turn_ends(outside, spoken):
    async def run():
        set_spoken_turn(outside)
        await _surfaces(spoken_mode=spoken)
        exhausted = spoken_turn()

        turn = _wrapped_turn(spoken_mode=spoken)
        await turn.__anext__()
        await turn.aclose()
        closed_early = spoken_turn()

        with pytest.raises(RuntimeError, match=BROKE):
            [surface async for surface in _wrapped_turn(_raising_turn, spoken_mode=spoken)]
        return exhausted, closed_early, spoken_turn()

    assert asyncio.run(run()) == (outside, outside, outside)


def test_a_service_without_a_factory_is_marked_and_put_back_too():
    async def run():
        set_spoken_turn(LIVE)
        chat = _chat(factory=False)
        surfaces = [surface async for surface in _wrapped_turn(chat=chat, spoken_mode=True)]
        typed = [surface async for surface in _wrapped_turn(chat=chat)]
        return surfaces, typed, spoken_turn(), chat.agent_factory

    assert asyncio.run(run()) == ([LIVE, LIVE], [NO_SURFACE, NO_SURFACE], LIVE, None)


def test_the_services_own_factory_is_back_after_a_marked_turn():
    async def run():
        chat = _chat()
        own = chat.agent_factory
        await _surfaces(chat=chat, spoken_mode=True)
        return chat.agent_factory is own

    assert asyncio.run(run()) is True


# --------------------------------------------------------------------------- Auto Live


def test_auto_live_marks_its_turn():
    from api import voice_retell

    source = inspect.getsource(voice_retell._agent_retell_stream_inner)
    marked = source.index("before = set_spoken_turn(LIVE)")
    frames = source.index("retell_response_frames(req.response_id, tap)")
    put_back = source.index("set_spoken_turn(before)")

    assert marked < frames < put_back
    assert source[frames:put_back].rstrip().endswith("finally:")
    assert voice_retell.set_spoken_turn is request_scope.set_spoken_turn
    assert voice_retell.LIVE == LIVE
