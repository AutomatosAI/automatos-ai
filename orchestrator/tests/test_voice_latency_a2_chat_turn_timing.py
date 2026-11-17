"""A chat turn's marks: on its reply's trace and in one ``chat_turn_timing`` line.

``consumers/chatbot/turn_timing.py`` feeds the turn's clock (``core/monitoring/turn_clock.py``)
from the two wrappers it already had, ``api/chat.py`` and ``services/chat_turns.py`` mark the
moments around the service, and the producer writes the line:

* five of the service's steps are timed by shadowing their bound methods for the turn; no
  shadow is left when the turn ends, raises or is closed early, and the frames shadow closes
  the generator it wraps;
* a model call is marked when it starts, a second one too, and the model's time is summed;
* a turn adopts the clock its request began, once; any other turn begins and ends its own;
* the saved reply's ``context_trace["timing"]`` is the clock at the save;
* through the real route and the real ``run_detached_turn``: one line per request, with
  ``origin=request``, the marks in order, the outcome, and the phone's ``turnRef`` when it fits;
  the turns that never reach the service get their line too;
* the line is ids and numbers: never the person's words or the reply's;
* ``CHAT_TURN_TIMING_ENABLED`` false: no ``timing`` key, no line, no shadow, and
  ``first_token_ms`` as before. ``CHAT_TURN_TIMING_STAGES_ENABLED`` false: no shadow only.

DB-free: no database, model, Redis or socket.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.auth.workspace_permission as permission
from api import chat
from api.chat_models import ChatRequest
from config import config
from consumers.chatbot import turn_timing
from consumers.chatbot.streaming import StreamingHandler
from core.monitoring import turn_clock
from services import chat_turns
from services.chat_turns import TurnRegistry

WS = "6a0f3c1e-2b4d-4e8f-9a1b-a2c10c400001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-a2c10c40c4a7"
USER_PK = 7
AUTO = 11
STEP = 0.02
STEP_MS = 20
ASKED_WORD = "NIGHTINGALE"
REPLY_WORD = "KINGFISHER"
ASKED = f"What is the status of Project {ASKED_WORD}?"
REF = "vt-test-0001"
HANDLER = StreamingHandler()
FINISH = HANDLER.format_aisdk_finish()
REPLY = (HANDLER.format_aisdk_text(f"Project {REPLY_WORD} ships on Friday."), FINISH)
ONE_UNIT_REPLY = (HANDLER.format_aisdk_text("It's five o'clock"), FINISH)  # no sentence end
STAGE_METHODS = tuple(name for _, name, _ in turn_timing.STAGES)
STAGE_NAMES = {stage for stage, _, _ in turn_timing.STAGES}
REQUEST_MARKS = ("authorised", "saved", "history", "routed", "response_started", "producer", "locked", "turn")
TOKEN = '0:"Hi"\n'


@pytest.fixture(autouse=True)
def _no_clock_left_behind():
    """A test that begins a clock in the test's own context does not hand it to the next."""
    turn_clock.leave(None)
    yield
    turn_clock.leave(None)


@pytest.fixture(autouse=True)
def _timing_on(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", True)
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_STAGES_ENABLED", True)
    monkeypatch.setattr(config, "VOICE_LIVE_SPEECH_UNIT_MAX_CHARS", 180)


@pytest.fixture
def lines(caplog):
    """The ``chat_turn_timing`` lines logged while the test runs, read when called."""
    caplog.set_level(logging.INFO, logger=turn_clock.logger.name)
    return lambda: [r.getMessage() for r in caplog.records
                    if r.getMessage().startswith(f"{turn_clock.TIMING_LOG} ")]


def _read(line: str):
    """A line's facts (chat, origin, spoken, ref, outcome) and its marks, in the line's order."""
    pairs = [token.split("=", 1) for token in line.split(" | ")[0].split()[1:]]
    return dict(pairs[:5]), {name: int(ms) for name, ms in pairs[5:]}


class _Saved:
    """A chat service that keeps what it was asked to save."""

    def __init__(self) -> None:
        self.rows: list = []

    def save_message(self, **kwargs):
        self.rows.append(kwargs)
        return NS(id=f"message-{len(self.rows)}")

    @property
    def reply(self) -> dict:
        (row,) = [row for row in self.rows if row.get("role") == "assistant"]
        return row

    @property
    def timing(self) -> dict:
        return self.reply["context_trace"]["timing"]


class _Chat:
    """The streaming service's shape, as far as the timing wrappers see it: the five staged
    steps, the model call and the turn."""

    def __init__(self) -> None:
        self.chat_service = _Saved()
        self.shadowed: list = []
        self.closed: list = []

    async def _load_agent_context(self, runtime):
        await asyncio.sleep(STEP)
        return {"agent": runtime}

    async def _get_tools(self):
        await asyncio.sleep(STEP)
        return ["tool"]

    async def _prepare_messages(self):
        await asyncio.sleep(STEP)
        return [], [], None

    async def _inject_composio_tools(self):
        await asyncio.sleep(STEP)
        return [], None

    async def _retrieval_first(self):
        try:
            await asyncio.sleep(STEP)
            yield "frame-1"
            yield "frame-2"
        finally:
            self.closed.append("retrieval")

    @turn_timing.records_the_first_token
    async def _stream_llm_call(self, items):
        await asyncio.sleep(STEP)
        for item in items:
            yield item

    @turn_timing.keeps_the_turns_timing
    async def turn(self, chat_id, *, calls=(), tools_twice=False, fail=False, messages=None, **kwargs):
        self.shadowed = [name for name in STAGE_METHODS if name in vars(self)]
        await self._load_agent_context(None)
        await self._get_tools()
        if tools_twice:
            await self._get_tools()
        await self._prepare_messages()
        await self._inject_composio_tools()
        async for frame in self._retrieval_first():
            yield frame
        for items in calls:
            async for item in self._stream_llm_call(items):
                if not isinstance(item, dict):
                    yield item
        if fail:
            raise RuntimeError("the turn failed")
        self.chat_service.save_message(
            role="assistant", parts=[{"type": "text", "text": f"Project {REPLY_WORD} ships."}],
            context_trace={"mode": "full"},
        )


def _run(chat, *args, **kwargs):
    """Drain ``chat.turn(...)`` in a task of its own: its frames, and the clock left in that
    task's context afterwards."""
    async def scenario():
        frames = [frame async for frame in chat.turn(*args, **kwargs)]
        return frames, turn_clock.current()

    return asyncio.run(scenario())


def _shadows(chat) -> list:
    return [name for name in STAGE_METHODS if name in vars(chat)]


# --------------------------------------------------------------------------- stages


def test_the_five_steps_are_timed_on_the_saved_replys_trace():
    chat = _Chat()

    frames, _ = _run(chat, "chat-1")

    assert frames == ["frame-1", "frame-2"]
    assert chat.shadowed == list(STAGE_METHODS), "all five were shadowed for the turn"
    stages = chat.chat_service.timing["stages"]
    assert STAGE_NAMES <= set(stages)
    assert all(stages[stage] >= STEP_MS for stage in STAGE_NAMES), stages
    assert chat.chat_service.reply["context_trace"]["mode"] == "full", "the trace's own keys are kept"
    assert chat.chat_service.timing["v"] == turn_clock.TRACE_VERSION
    assert _shadows(chat) == [], "no shadow is left on the service"


def test_a_step_called_twice_sums_into_its_stage():
    chat = _Chat()

    _run(chat, "chat-1", tools_twice=True)

    stages = chat.chat_service.timing["stages"]
    assert stages["tools"] >= 2 * STEP_MS


def test_no_shadow_is_left_when_the_turn_raises():
    chat = _Chat()

    with pytest.raises(RuntimeError, match="the turn failed"):
        _run(chat, "chat-1", fail=True)

    assert chat.shadowed == list(STAGE_METHODS)
    assert _shadows(chat) == []
    assert isinstance(chat.chat_service, _Saved), "the service's own chat service is put back"


def test_no_shadow_is_left_when_the_turn_is_closed_early():
    chat = _Chat()

    async def scenario():
        turn = chat.turn("chat-1")
        first = await turn.__anext__()
        during = _shadows(chat)
        await turn.aclose()  # Stop
        return first, during, _shadows(chat), turn_clock.current()

    first, during, after, clock = asyncio.run(scenario())

    assert first == "frame-1"
    assert during == list(STAGE_METHODS)
    assert after == [], "the shadows are off by the time aclose() returns"
    assert clock is None, "the turn's own clock is left when it is closed"
    assert isinstance(chat.chat_service, _Saved)


def test_closing_a_timed_frames_step_closes_the_generator_it_wraps():
    """The frames shadow adds no generator layer that outlives a Stop: by the time its
    ``aclose()`` returns, the step's own ``finally`` has run and its time is on the clock."""
    chat = _Chat()
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_TURN)

    async def scenario():
        restore = turn_timing._staged(chat, clock)
        frames = chat._retrieval_first()
        first = await frames.__anext__()
        await frames.aclose()
        closed = list(chat.closed)
        turn_timing._unstaged(chat, restore)
        return first, closed

    first, closed = asyncio.run(scenario())

    assert first == "frame-1"
    assert closed == ["retrieval"]
    assert clock.snapshot()["stages"]["knowledge"] >= STEP_MS
    assert _shadows(chat) == []


def test_an_attribute_the_instance_already_had_is_put_back():
    chat = _Chat()

    async def its_own_tools():
        return ["its own"]

    chat._get_tools = its_own_tools
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_TURN)

    restore = turn_timing._staged(chat, clock)
    assert chat._get_tools is not its_own_tools
    assert asyncio.run(chat._get_tools()) == ["its own"]
    turn_timing._unstaged(chat, restore)

    assert chat._get_tools is its_own_tools
    assert _shadows(chat) == ["_get_tools"], "its own attribute, not a shadow"
    assert "tools" in clock.snapshot()["stages"]


def test_a_chat_without_those_steps_runs_and_saves_as_before():
    saved = []
    chats = NS(save_message=lambda **kwargs: saved.append(kwargs))
    chat = NS(chat_service=chats)

    @turn_timing.records_the_first_token
    async def stream_llm_call(chat):
        await asyncio.sleep(STEP)
        yield TOKEN
        yield {"_response": object()}

    @turn_timing.keeps_the_turns_timing
    async def turn(chat):
        async for frame in stream_llm_call(chat):
            yield frame
        chat.chat_service.save_message(role="user", context_trace=None)
        chat.chat_service.save_message(role="assistant", context_trace={"mode": "lean"})

    async def drain():
        return [frame async for frame in turn(chat)]

    assert len(asyncio.run(drain())) == 2
    assert saved[0] == {"role": "user", "context_trace": None}, "only the reply's trace is touched"
    trace = saved[1]["context_trace"]
    assert trace["mode"] == "lean" and trace["first_token_ms"] >= STEP_MS
    assert trace["timing"]["stages"] == {"model": trace["timing"]["stages"]["model"]}, "no staged step"
    assert chat.chat_service is chats
    assert set(vars(chat)) == {"chat_service", "_turn_started", "_first_token_ms"}


def test_a_step_of_another_kind_is_left_alone():
    """A plain function where an ``async def`` is expected (a mock, a stub) is not shadowed."""
    chat = NS(_get_tools=lambda: ["plain"], _retrieval_first=lambda: iter(()))
    its_tools = chat._get_tools

    restore = turn_timing._staged(chat, turn_clock.TurnClock(turn_clock.ORIGIN_TURN))

    assert restore == ()
    assert chat._get_tools is its_tools


def test_with_the_stages_switch_off_nothing_is_shadowed_and_the_marks_stay(monkeypatch, lines):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_STAGES_ENABLED", False)
    chat = _Chat()

    _run(chat, "chat-1", calls=((TOKEN, {"_response": object()}),))

    assert chat.shadowed == [], "nothing was shadowed during the turn"
    timing = chat.chat_service.timing
    assert "turn" in timing["marks"] and "model_start" in timing["marks"]
    assert set(timing["stages"]) == {"model"}, "the model's time is not a method shadow"
    assert len(lines()) == 1


# --------------------------------------------------------------------------- model calls


def test_two_model_calls_are_a_start_a_second_start_a_sum_and_a_count():
    chat = _Chat()
    whole = {"_response": object()}

    _run(chat, "chat-1", calls=((whole,), (TOKEN, whole)))

    timing = chat.chat_service.timing
    marks = timing["marks"]
    assert marks["model_again"] >= marks["model_start"] + STEP_MS - 1  # each mark is rounded
    assert timing["counts"] == {"model_calls": 2}
    assert timing["stages"]["model"] >= 2 * STEP_MS


def test_a_turn_with_one_model_call_has_no_second_start():
    chat = _Chat()

    _run(chat, "chat-1", calls=((TOKEN, {"_response": object()}),))

    timing = chat.chat_service.timing
    assert "model_start" in timing["marks"] and "model_again" not in timing["marks"]
    assert timing["counts"] == {"model_calls": 1}


def test_a_first_call_that_answers_whole_has_a_first_response_and_no_first_token():
    chat = _Chat()

    _run(chat, "chat-1", calls=(({"_response": object()},),))

    trace = chat.chat_service.reply["context_trace"]
    marks = trace["timing"]["marks"]
    assert marks["first_response"] >= marks["model_start"] + STEP_MS - 1  # each mark is rounded
    assert "first_token" not in marks
    assert turn_timing.FIRST_TOKEN_KEY not in trace


def test_a_streamed_first_call_has_both_and_first_token_ms_is_from_the_turns_start():
    chat = _Chat()
    wait_ms = 40

    async def scenario():
        clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
        await asyncio.sleep(wait_ms / 1000)  # the request's time before the turn starts
        return [frame async for frame in chat.turn("chat-1", calls=((TOKEN, {"_response": object()}),))], clock

    asyncio.run(scenario())

    trace = chat.chat_service.reply["context_trace"]
    marks = trace["timing"]["marks"]
    assert marks["first_token"] <= marks["first_response"]
    assert trace[turn_timing.FIRST_TOKEN_KEY] >= STEP_MS
    assert marks["turn"] >= wait_ms
    # The mark counts from the request; first_token_ms still counts from the turn (a millisecond of rounding apart).
    assert marks["first_token"] - trace[turn_timing.FIRST_TOKEN_KEY] >= wait_ms - 2


# --------------------------------------------------------------------------- origin and claiming


def test_a_turn_with_no_clock_begins_its_own_and_leaves_it():
    chat = _Chat()

    _, clock_after = _run(chat, "chat-1")

    assert chat.chat_service.timing["origin"] == turn_clock.ORIGIN_TURN
    assert clock_after is None


def test_a_requests_clock_is_adopted_and_left_for_the_producer_to_finish(lines):
    chat = _Chat()

    async def scenario():
        clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
        clock.mark("authorised")
        frames = [frame async for frame in chat.turn("chat-1")]
        return clock, turn_clock.current(), frames

    clock, clock_after, _ = asyncio.run(scenario())

    timing = chat.chat_service.timing
    assert timing["origin"] == turn_clock.ORIGIN_REQUEST
    assert {"authorised", "turn", "reply_saved"} <= set(timing["marks"])
    assert clock_after is clock, "still the context's clock after the turn"
    assert lines() == [], "the producer writes a request clock's line, not the turn"
    assert "done" not in clock.snapshot()["marks"]
    assert clock.close() is True, "nobody has written its line yet"


def test_two_turns_under_one_requests_clock_get_two_clocks(lines):
    first, second = _Chat(), _Chat()

    async def scenario():
        clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
        [frame async for frame in first.turn("chat-1")]
        [frame async for frame in second.turn("chat-2")]
        return clock, turn_clock.current()

    clock, clock_after = asyncio.run(scenario())

    assert first.chat_service.timing["origin"] == turn_clock.ORIGIN_REQUEST
    assert second.chat_service.timing["origin"] == turn_clock.ORIGIN_TURN
    assert clock_after is clock, "the second turn left its own clock and put the request's back"
    (line,) = lines()
    facts, _ = _read(line)
    assert facts["chat"] == "chat-2" and facts["origin"] == turn_clock.ORIGIN_TURN


def test_a_turn_clock_left_behind_in_the_context_is_not_adopted():
    chat = _Chat()

    async def scenario():
        stale = turn_clock.begin(turn_clock.ORIGIN_TURN)  # never left
        stale.mark("authorised")
        [frame async for frame in chat.turn("chat-1")]
        return stale, turn_clock.current()

    stale, clock_after = asyncio.run(scenario())

    timing = chat.chat_service.timing
    assert timing["origin"] == turn_clock.ORIGIN_TURN
    assert "authorised" not in timing["marks"], "a new clock, not the one left behind"
    assert "turn" not in stale.snapshot()["marks"]
    assert clock_after is stale


# --------------------------------------------------------------------------- spoken, the line, the switch


def test_a_spoken_turn_says_so_on_its_trace():
    spoken, typed = _Chat(), _Chat()

    _run(spoken, "chat-1", spoken_mode=True)
    _run(typed, "chat-2")

    assert spoken.chat_service.timing["spoken"] is True
    assert typed.chat_service.timing["spoken"] is False


def test_a_turn_that_began_its_clock_writes_one_line_with_ids_and_numbers_only(caplog, lines):
    chat = _Chat()
    reply = HANDLER.format_aisdk_text(f"Project {REPLY_WORD} ships on Friday.")

    with caplog.at_level(logging.DEBUG):
        _run(chat, "chat-1", spoken_mode=True, calls=((reply, {"_response": object()}),),
             messages=[{"role": "user", "content": ASKED}])

    (line,) = lines()
    facts, marks = _read(line)
    assert facts == {"chat": "chat-1", "origin": "turn", "spoken": "true", "ref": "-", "outcome": "-"}
    assert {"turn", "model_start", "first_token", "first_response", "reply_saved", "done"} <= set(marks)
    assert list(marks.values()) == sorted(marks.values()), "marks are in time order"
    assert "model_calls=1" in line and "knowledge=" in line
    for record in caplog.records:
        assert ASKED_WORD not in record.getMessage() and REPLY_WORD not in record.getMessage()
    assert ASKED_WORD not in str(chat.chat_service.timing) and REPLY_WORD not in str(chat.chat_service.timing)


def test_a_turn_that_raises_still_writes_its_line(lines):
    chat = _Chat()

    with pytest.raises(RuntimeError):
        _run(chat, "chat-1", fail=True)

    (line,) = lines()
    facts, marks = _read(line)
    assert facts["chat"] == "chat-1" and facts["origin"] == "turn"
    assert "done" in marks and "reply_saved" not in marks


def test_with_the_switch_off_the_turn_is_as_before(monkeypatch, lines):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", False)
    chat = _Chat()

    _, clock_after = _run(chat, "chat-1", calls=((TOKEN, {"_response": object()}),))

    trace = chat.chat_service.reply["context_trace"]
    assert set(trace) == {"mode", turn_timing.FIRST_TOKEN_KEY}, "no timing key"
    assert trace[turn_timing.FIRST_TOKEN_KEY] >= STEP_MS
    assert chat.shadowed == [] and _shadows(chat) == []
    assert lines() == []
    assert clock_after is None


def test_with_timing_adds_the_clock_as_a_new_dict():
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_TURN)
    clock.mark("turn")
    trace = {"mode": "lean"}

    timed = turn_timing.with_timing(trace, clock)

    assert timed == {"mode": "lean", "timing": clock.snapshot()}
    assert trace == {"mode": "lean"}, "the trace handed in is not changed"
    assert turn_timing.with_timing(None, clock) == {"timing": clock.snapshot()}
    assert turn_timing.with_timing(trace, None) is trace
    assert turn_timing.with_timing(None, None) is None


# --------------------------------------------------------------------------- the request body


def test_a_turn_ref_that_fits_is_kept_and_any_other_is_dropped():
    message = {"role": "user", "parts": [{"type": "text", "text": "Hi"}]}

    assert ChatRequest(message=message).turnRef is None
    assert ChatRequest(message=message, turnRef=REF).turnRef == REF
    for bad in ("not a ref!", "vt-1a.b-3", "short", "x" * 41, 12345678, None, ["vt-test-0001"]):
        assert ChatRequest(message=message, turnRef=bad).turnRef is None, bad


# --------------------------------------------------------------------------- the route, end to end


class _NoRedis(TurnRegistry):
    """The registry of a server with no Redis: process-local, nothing dialled."""

    def _client(self):
        return None


class _RouteChats:
    """The request's chat service: the person's message is saved, nothing is read."""

    def __init__(self, db):
        self.db = db

    def save_message(self, **kwargs):
        return NS(id="message-1")


def _streaming_service(state):
    """A streaming service class whose turn replays ``state.reply`` and saves the reply after
    its first frame, under the real ``keeps_the_turns_timing``."""
    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = HANDLER
            self.chat_service = state.saved

        @turn_timing.keeps_the_turns_timing
        async def stream_response_with_agent(self, **kwargs):
            yield state.reply[0]
            self.chat_service.save_message(role="assistant", context_trace={"mode": "full"})
            for frame in state.reply[1:]:
                yield frame

    return _StreamingService


class _Lock:
    """The per-chat lock. With ``stop_at_the_lock`` the turn is stopped while it waits for it."""

    def __init__(self, state) -> None:
        self._state = state

    @asynccontextmanager
    async def acquire(self, key):
        if self._state.stop_at_the_lock:
            stop = self._state.registry.request_cancel(CHAT_ID)
            self._state.held.append(asyncio.get_running_loop().create_task(stop))
            await asyncio.Event().wait()
        yield


def _session_local(state):
    def _session():
        if state.no_session:
            raise RuntimeError("no database")
        return NS(close=lambda: None)

    return _session


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` through the real router, dependencies and ``run_detached_turn``; the
    chat service, the lane, the lock and the streaming service are fakes. The streaming
    service's turn is wrapped in the real ``keeps_the_turns_timing``."""
    import services.cli_ticket_lane as cli_lane

    state = NS(saved=_Saved(), reply=REPLY, session_agent=None, stop_at_the_lock=False, no_session=False,
               registry=_NoRedis(), noted=[], held=[])

    async def _lane(*args, **kwargs):
        return NS(agent_id=AUTO, assessment=None, session_agent=state.session_agent, suggest_mission=False,
                  skip_composio=False)

    async def _note(workspace_id, chat_id, message_id, *, cancelled):
        state.noted.append(cancelled)
        return True

    monkeypatch.setattr(permission, "resolve_workspace_role", lambda db, ctx: "editor")
    monkeypatch.setattr(chat, "ChatService", _RouteChats)
    monkeypatch.setattr(chat, "StreamingChatService", _streaming_service(state))
    monkeypatch.setattr(chat, "SessionLocal", _session_local(state))
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Lock(state))
    monkeypatch.setattr(chat, "get_user_id", lambda db, ctx=None: USER_PK)
    monkeypatch.setattr(chat, "_open_chat", lambda *args, **kwargs: CHAT_ID)
    monkeypatch.setattr(chat, "turn_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat, "_turn_lane", _lane)
    monkeypatch.setattr(chat, "note_an_unanswered_turn", _note)
    monkeypatch.setattr(chat, "save_turn_note", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli_lane, "session_agent_terminal_message", lambda db, agent_id: "Use the terminal.")
    monkeypatch.setattr(chat_turns, "get_turn_registry", lambda: state.registry)
    return state


def _post(**over):
    app = FastAPI()
    app.include_router(chat.router)
    ctx = NS(workspace_id=WS, user=NS(id=USER_PK, system_role="user", clerk_user_id="user_c1"), auth_type="clerk")
    app.dependency_overrides[chat.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[chat.get_db] = lambda: NS(rollback=lambda: None)
    body = {"message": {"role": "user", "parts": [{"type": "text", "text": ASKED}]}, **over}
    return TestClient(app).post("/api/chat", json=body)


def test_a_request_logs_one_line_with_its_marks_in_order(route, lines):
    response = _post(turnRef=REF)

    assert response.status_code == 200, response.text
    assert response.text == "".join(REPLY)
    (line,) = lines()
    facts, marks = _read(line)
    assert facts == {"chat": CHAT_ID, "origin": "request", "spoken": "false", "ref": REF, "outcome": "completed"}
    in_order = [marks[name] for name in REQUEST_MARKS]  # KeyError: a mark is missing
    assert in_order == sorted(in_order), marks
    assert marks["first_text"] >= marks["turn"] and marks["done"] >= marks["reply_saved"]
    assert list(marks.values()) == sorted(marks.values())
    stored = route.saved.timing
    assert stored["origin"] == "request" and stored["ref"] == REF and stored["spoken"] is False
    assert "first_text" in stored["marks"], "a reply whose text left before the save has it stored"
    assert "done" not in stored["marks"] and "outcome" not in stored, "the trace is the clock at the save"
    assert ASKED_WORD not in line and REPLY_WORD not in line
    assert ASKED_WORD not in str(stored) and REPLY_WORD not in str(stored)
    assert route.noted == []


@pytest.mark.parametrize("bad", ["not a ref!", "vt-1a.b-3"])
def test_a_turn_ref_that_does_not_fit_is_left_off_and_the_request_still_answers(route, lines, bad):
    response = _post(turnRef=bad)

    assert response.status_code == 200, response.text
    (line,) = lines()
    facts, _ = _read(line)
    assert facts["ref"] == "-" and facts["outcome"] == "completed"
    assert bad not in line
    assert "ref" not in route.saved.timing


def test_a_request_with_no_turn_ref_has_none(route, lines):
    assert _post().status_code == 200

    (line,) = lines()
    assert _read(line)[0]["ref"] == "-"
    assert "ref" not in route.saved.timing


def test_a_one_unit_spoken_reply_has_its_first_text_on_the_line_only(route, lines):
    """A spoken reply with no sentence end leaves after the turn has ended (the speech-unit
    hold), so the trace saved before that cannot hold it; the line can."""
    route.reply = ONE_UNIT_REPLY

    response = _post(spokenMode=True, turnRef=REF)

    assert response.status_code == 200, response.text
    assert response.text.startswith(chat_turns.TEXT_FRAME_PREFIX) and response.text.endswith(FINISH)
    assert "first_text" not in route.saved.timing["marks"]
    assert route.saved.timing["spoken"] is True
    (line,) = lines()
    facts, marks = _read(line)
    assert facts["spoken"] == "true" and facts["outcome"] == "completed"
    assert marks["first_text"] >= marks["reply_saved"]


def test_the_session_agent_lane_logs_its_line_without_a_turn(route, lines):
    route.session_agent = True

    response = _post()

    assert response.status_code == 200, response.text
    assert response.text.startswith("e:")
    (line,) = lines()
    facts, marks = _read(line)
    assert facts["origin"] == "request" and facts["outcome"] == "completed"
    assert "locked" in marks and "turn" not in marks and "first_text" not in marks
    assert route.saved.rows == []


def test_a_stop_while_waiting_for_the_lock_logs_a_cancelled_line(route, lines):
    route.stop_at_the_lock = True

    response = _post()

    assert response.status_code == 200, response.text
    assert response.text == ""
    (line,) = lines()
    facts, marks = _read(line)
    assert facts["origin"] == "request" and facts["outcome"] == "cancelled"
    assert "producer" in marks and "locked" not in marks and "turn" not in marks


def test_a_turn_that_fails_before_the_service_logs_a_failed_line(route, lines):
    route.no_session = True

    response = _post()

    assert response.status_code == 200, response.text
    assert response.text.startswith("e:")
    (line,) = lines()
    facts, marks = _read(line)
    assert facts["origin"] == "request" and facts["outcome"] == "failed"
    assert "producer" in marks and "locked" not in marks


def test_with_the_switch_off_a_request_logs_no_line_and_stores_no_timing(route, lines, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", False)

    response = _post(turnRef=REF)

    assert response.status_code == 200, response.text
    assert response.text == "".join(REPLY)
    assert lines() == []
    assert route.saved.reply["context_trace"] == {"mode": "full"}


def test_the_clock_starts_before_the_permission_gate_on_the_chat_route():
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    (route,) = [r for r in chat.router.routes if r.path == "/api/chat" and "POST" in r.methods]
    calls = [dep.call for dep in route.dependant.dependencies]
    markers = [getattr(call, PERMISSION_MARKER_ATTR, None) for call in calls]

    assert calls[0] is turn_clock.arrives
    assert "agents:execute" in markers
    assert calls.index(turn_clock.authorised) > markers.index("agents:execute")


def test_how_a_producer_ended_reads_as_one_word():
    assert chat_turns._ended_as({"completed": True, "cancelled": False}) == "completed"
    assert chat_turns._ended_as({"completed": False, "cancelled": True}) == "cancelled"
    assert chat_turns._ended_as({"completed": False, "cancelled": False}) == "failed"
