"""PRD-264 EUAI-04 — Auto Live says it is an AI at the start of the call and reminds during it.

EU AI Act Art. 50(1), Guidelines ¶37: a voice says in words, at the start,
that it is an AI, and reminds in longer calls.

* The begin frame (response 0) carries ``VOICE_AI_GREETING`` (default "Hi,
  this is Auto, your AI assistant."). Retell speaks it as the opening line,
  so it is never empty, and a configured greeting without the AI term is
  never spoken.
* After an automatic reconnect (a new socket for a call that already had one,
  marked in Redis) the next response opens with "Just so you know, I'm an AI
  assistant."
* With a fake clock past ``VOICE_AI_REMINDER_MINUTES`` the next response opens
  with the reminder, once.

The route (``api/voice_retell.py``) builds the greeting and the reminder in
``modules/policy/ai_disclosure.py`` and lends them to the socket through
``SocketPorts``; every socket here runs on the route's own ports, so the
wiring is what is tested.

Nothing here is gated by ``AUTH_EDITION``; the ``edition`` fixture runs the
socket cases in both editions to prove it. No Retell, no model, no network,
no database.
"""
from __future__ import annotations

import ast
import asyncio
import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from config import config as app_config  # noqa: E402
from modules.policy import ai_disclosure  # noqa: E402
from modules.voice import ai_notice, live_socket  # noqa: E402

T0 = 5_000.0
MINUTE = 60.0
CALL_ID = "call_euai04"
REMINDER = "Just so you know, I'm an AI assistant."
DEFAULT_GREETING = "Hi, this is Auto, your AI assistant."


@pytest.fixture(params=("local", "saas"))
def edition(request, monkeypatch):
    monkeypatch.setattr(app_config, "AUTH_EDITION", request.param)
    return request.param


@pytest.fixture(autouse=True)
def voice_settings(monkeypatch):
    """The shipped defaults, and no slow-turn ack, so every frame here is ours."""
    monkeypatch.setattr(app_config, "VOICE_AI_GREETING", DEFAULT_GREETING)
    monkeypatch.setattr(app_config, "VOICE_AI_REMINDER_MINUTES", 10.0)
    monkeypatch.setattr(app_config, "VOICE_LIVE_LANGUAGE", "en-GB")
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", 0)


class Clock:
    """Call time the test moves by hand."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> Callable[[], None]:
        """A script step that moves the clock on by ``seconds``."""

        def step() -> None:
            self.now += seconds

        return step


class FakeRedis:
    """The one call the reconnect mark makes: ``set(key, value, nx=True, ex=ttl)``."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, Optional[int]] = {}

    def set(self, key: str, value: str, nx: bool = False, ex: Optional[int] = None) -> Optional[bool]:
        if nx and key in self.values:
            return None
        self.values = {**self.values, key: value}
        self.ttls = {**self.ttls, key: ex}
        return True


class BrokenRedis:
    def set(self, key: str, value: str, nx: bool = False, ex: Optional[int] = None) -> Optional[bool]:
        raise ConnectionError("redis down")


class ScriptedSocket:
    """A Retell socket double. A script step is an inbound message, or a callable
    run between messages (moving the clock). Before each step the loop runs a
    few ticks, so a just-created turn finishes first."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.sent: list[dict] = []
        self.close_code: Optional[int] = None

    async def accept(self) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.close_code = code

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive_json(self) -> dict:
        from fastapi import WebSocketDisconnect

        while True:
            for _ in range(3):
                await asyncio.sleep(0)
            if not self.script:
                raise WebSocketDisconnect(1000)
            step = self.script.pop(0)
            if callable(step):
                step()
                continue
            return step


CALL_DETAILS = {
    "interaction_type": "call_details",
    "call": {"call_id": CALL_ID, "retell_llm_dynamic_variables": {"workspace_id": "ws-euai04"}},
}


def _turn(rid: int) -> dict:
    return {
        "interaction_type": "response_required",
        "response_id": rid,
        "transcript": [{"role": "user", "content": f"question {rid}"}],
    }


async def _answer(req: Any):
    yield {"response_id": req.response_id, "content": f"answer {req.response_id}", "content_complete": False}
    yield {"response_id": req.response_id, "content": "", "content_complete": True}


def _reconnect_says(value: bool):
    async def socket_reconnected(call_id: str) -> bool:
        return value

    return socket_reconnected


def _route_ports(stream_turn: Callable, *, clock: Clock, socket_reconnected=None) -> live_socket.SocketPorts:
    """The ports the route lends a call (its AI notice texts included), with a
    fake turn stream, status read and clock."""
    import api.voice_retell as vr

    extra = {} if socket_reconnected is None else {"socket_reconnected": socket_reconnected}
    return dataclasses.replace(
        vr._socket_ports(), stream_turn=stream_turn, call_status=lambda cid: "started", clock=clock, **extra
    )


def _run_call(script: list, *, clock: Clock, socket_reconnected=None) -> ScriptedSocket:
    """Drive one socket of a call through ``script``; return what it sent."""
    ports = _route_ports(_answer, clock=clock, socket_reconnected=socket_reconnected)
    ws = ScriptedSocket(script)
    asyncio.run(live_socket.LiveCallSocket(ws, CALL_ID, ports).run())
    return ws


def _contents(ws: ScriptedSocket, rid: int) -> list[str]:
    """The content of every response frame sent for response ``rid``, in order."""
    return [
        m["content"] for m in ws.sent
        if m.get("response_type") == "response" and m.get("response_id") == rid
    ]


def _begin(ws: ScriptedSocket) -> dict:
    begins = [m for m in ws.sent if m.get("response_type") == "response" and m.get("response_id") == 0]
    assert len(begins) == 1, "exactly one begin frame per socket"
    return begins[0]


# ---------------------------------------------------------------------------
# Auto speaks first, and says it is an AI
# ---------------------------------------------------------------------------

def test_the_begin_frame_is_not_empty_and_says_ai(edition):
    ws = _run_call([CALL_DETAILS, _turn(1)], clock=Clock(T0), socket_reconnected=_reconnect_says(False))

    begin = _begin(ws)
    assert begin["content"].strip(), "response 0 is Auto's opening line: never empty"
    assert ai_disclosure.contains_badge_term(begin["content"], "AI")
    assert begin["content"] == DEFAULT_GREETING
    assert begin["content_complete"] is True
    # the config frame still opens the socket; the greeting answers call_details
    assert ws.sent[0]["response_type"] == "config"
    assert ws.sent.index(begin) == 1


def test_a_configured_greeting_is_spoken_when_it_says_ai(monkeypatch):
    monkeypatch.setattr(app_config, "VOICE_AI_GREETING", "  Hello, Auto here, the AI assistant for Harbour Coffee.  ")
    ws = _run_call([CALL_DETAILS], clock=Clock(T0), socket_reconnected=_reconnect_says(False))
    assert _begin(ws)["content"] == "Hello, Auto here, the AI assistant for Harbour Coffee."


@pytest.mark.parametrize("configured", ["", "   ", "Hello, this is Auto.", "Check your MAIL"])
def test_a_greeting_that_does_not_say_ai_is_never_spoken(monkeypatch, configured):
    monkeypatch.setattr(app_config, "VOICE_AI_GREETING", configured)
    ws = _run_call([CALL_DETAILS], clock=Clock(T0), socket_reconnected=_reconnect_says(False))
    assert _begin(ws)["content"] == DEFAULT_GREETING


def test_the_greeting_rule_follows_the_call_language():
    assert ai_disclosure.voice_greeting("Hallo, hier ist Auto, dein KI-Assistent.", "de-DE") == (
        "Hallo, hier ist Auto, dein KI-Assistent."
    )
    assert ai_disclosure.voice_greeting("Olá, sou o Auto, o seu assistente de IA.", "pt-PT") == (
        "Olá, sou o Auto, o seu assistente de IA."
    )
    assert ai_disclosure.voice_greeting("Hallo, hier ist Auto.", "de") == ai_disclosure.VOICE_GREETING
    assert ai_disclosure.voice_greeting(None, "en") == ai_disclosure.VOICE_GREETING
    assert ai_disclosure.VOICE_GREETING == DEFAULT_GREETING


def test_a_first_socket_adds_no_reminder(edition):
    ws = _run_call(
        [CALL_DETAILS, _turn(1), _turn(2)], clock=Clock(T0), socket_reconnected=_reconnect_says(False)
    )
    assert _contents(ws, 1) == ["answer 1", ""]
    assert _contents(ws, 2) == ["answer 2", ""]


# ---------------------------------------------------------------------------
# After an automatic reconnect, the next response opens with the reminder
# ---------------------------------------------------------------------------

def test_a_reconnect_adds_the_reminder_to_the_next_response_once(edition):
    ws = _run_call(
        [CALL_DETAILS, _turn(7), _turn(8)], clock=Clock(T0), socket_reconnected=_reconnect_says(True)
    )
    first = _contents(ws, 7)
    assert first[0].startswith(REMINDER), "the reminder opens the response"
    assert first[1:] == ["answer 7", ""]
    assert _contents(ws, 8) == ["answer 8", ""], "one reminder per reconnect"


def test_the_second_socket_of_a_call_is_the_reconnect(edition, monkeypatch):
    """Through the real Redis mark: the first socket of a call has no reminder,
    the reconnect (a new socket for the same call) does."""
    redis = FakeRedis()
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: redis)

    first = _run_call([CALL_DETAILS, _turn(1)], clock=Clock(T0))
    again = _run_call([CALL_DETAILS, _turn(2)], clock=Clock(T0))

    assert _contents(first, 1) == ["answer 1", ""]
    assert _contents(again, 2)[0].startswith(REMINDER)
    assert _begin(again)["content"] == DEFAULT_GREETING, "the begin frame is never empty, reconnect or not"


def test_the_reconnect_mark_is_per_call_hashed_and_expires(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: redis)
    monkeypatch.setattr(app_config, "VOICE_LIVE_MAX_CALL_MINUTES", 30)

    assert asyncio.run(ai_notice.socket_reconnected("call_a")) is False
    assert asyncio.run(ai_notice.socket_reconnected("call_a")) is True
    assert asyncio.run(ai_notice.socket_reconnected("call_b")) is False

    assert not any("call_a" in key for key in redis.values), "the call id is the socket's credential"
    assert set(redis.ttls.values()) == {30 * 60 + ai_notice.SOCKET_MARK_GRACE_SECONDS}


@pytest.mark.parametrize("store", [None, BrokenRedis()])
def test_an_unreadable_mark_counts_as_a_reconnect(monkeypatch, store):
    """Losing the state means one reminder too many, never one too few."""
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: store)
    assert asyncio.run(ai_notice.socket_reconnected(CALL_ID)) is True


class SlowReminderSocket(ScriptedSocket):
    """A socket whose send of the AI reminder is slow (a loaded event loop)."""

    async def send_json(self, data: dict) -> None:
        if str(data.get("content", "")).startswith(REMINDER):
            await asyncio.sleep(SLOW_SEND_SECONDS)
        self.sent.append(data)


SLOW_SEND_SECONDS = 0.05
ACK_TEXT = "One moment."


def test_the_reminder_comes_before_the_slow_turn_ack(monkeypatch):
    """A slow reminder send never lets the slow-turn ack speak first."""
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", SLOW_SEND_SECONDS / 5)
    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_TEXT", ACK_TEXT)

    async def slow_answer(req: Any):
        await asyncio.sleep(SLOW_SEND_SECONDS * 2)
        yield {"response_id": req.response_id, "content": "answer", "content_complete": True}

    ports = _route_ports(slow_answer, clock=Clock(T0), socket_reconnected=_reconnect_says(True))
    ws = SlowReminderSocket([CALL_DETAILS, _turn(1)])
    asyncio.run(live_socket.LiveCallSocket(ws, CALL_ID, ports).run())

    contents = _contents(ws, 1)
    assert contents[0].startswith(REMINDER), "the reminder opens the response"
    assert contents[1].startswith(ACK_TEXT)
    assert contents[2:] == ["answer"]


# ---------------------------------------------------------------------------
# Every VOICE_AI_REMINDER_MINUTES of call time, the reminder once
# ---------------------------------------------------------------------------

def test_a_clock_past_the_interval_adds_the_reminder_once(edition):
    clock = Clock(T0)
    script = [
        CALL_DETAILS,
        _turn(1),
        clock.advance(9 * MINUTE),
        _turn(2),  # 9 minutes in: not yet
        clock.advance(1 * MINUTE + 1),
        _turn(3),  # past 10 minutes since the greeting: the reminder
        _turn(4),  # the same moment: not again
        clock.advance(5 * MINUTE),
        _turn(5),  # 5 minutes after the reminder: not yet
    ]
    ws = _run_call(script, clock=clock, socket_reconnected=_reconnect_says(False))

    assert _contents(ws, 1) == ["answer 1", ""]
    assert _contents(ws, 2) == ["answer 2", ""]
    third = _contents(ws, 3)
    assert third[0].startswith(REMINDER)
    assert third[1:] == ["answer 3", ""]
    assert _contents(ws, 4) == ["answer 4", ""]
    assert _contents(ws, 5) == ["answer 5", ""]
    reminders = [m for m in ws.sent if str(m.get("content", "")).startswith(REMINDER)]
    assert len(reminders) == 1


def test_the_interval_follows_the_setting(monkeypatch):
    monkeypatch.setattr(app_config, "VOICE_AI_REMINDER_MINUTES", 2.0)
    clock = Clock(T0)
    ws = _run_call(
        [CALL_DETAILS, clock.advance(2 * MINUTE), _turn(1)],
        clock=clock,
        socket_reconnected=_reconnect_says(False),
    )
    assert _contents(ws, 1)[0].startswith(REMINDER)


def test_the_reminder_rule_never_turns_off(monkeypatch):
    due = ai_notice.reminder_due
    assert due(reconnected=True, last_notice_at=T0, now=T0) is True
    assert due(reconnected=False, last_notice_at=T0, now=T0 + 599) is False
    assert due(reconnected=False, last_notice_at=T0, now=T0 + 600) is True
    # 0 (or less) is not an off switch: every response reminds
    for minutes in (0.0, -1.0):
        monkeypatch.setattr(app_config, "VOICE_AI_REMINDER_MINUTES", minutes)
        assert due(reconnected=False, last_notice_at=T0, now=T0) is True


# ---------------------------------------------------------------------------
# The texts come from the route: the voice module imports no other feature module
# ---------------------------------------------------------------------------

def test_the_route_lends_the_texts_from_the_one_rules_module():
    import api.voice_retell as vr

    ports = vr._socket_ports()
    assert ports.ai_greeting == DEFAULT_GREETING
    assert ports.ai_reminder == ai_disclosure.VOICE_REMINDER == REMINDER
    assert ai_notice.reminder_frame_text(ports.ai_reminder).startswith(REMINDER)


def test_a_socket_cannot_be_built_without_the_ai_texts():
    with pytest.raises(TypeError):
        live_socket.SocketPorts(stream_turn=_answer, call_status=lambda cid: "started")  # type: ignore[call-arg]


def _imported_modules(module: Any) -> set[str]:
    """Every module name a source file imports (top level or inside a function)."""
    nodes = list(ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))))
    froms = {node.module for node in nodes if isinstance(node, ast.ImportFrom) and node.module}
    plain = {alias.name for node in nodes if isinstance(node, ast.Import) for alias in node.names}
    return froms | plain


def test_the_voice_module_does_not_import_the_policy_module():
    """PRD-211 independence: modules.voice never imports modules.policy."""
    for module in (ai_notice, live_socket):
        imported = _imported_modules(module)
        assert not any(name.startswith("modules.policy") for name in imported), (module.__name__, imported)
