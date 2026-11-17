"""A chat turn's clock (core/monitoring/turn_clock.py) and the two ids it may carry
(core/monitoring/turn_ref.py).

The clock is what every later voice-latency change is measured with, so its own rules are
pinned here:

* a mark is whole milliseconds since the clock started, and the FIRST mark of a name wins;
  a stage sums over its calls; a snapshot is new dicts, never a view of the clock;
* with no clock in the context every module-level call does nothing;
* the clock rides the context: two tasks each see their own, a child task and a thread
  (``asyncio.to_thread``) write to their parent's, and marks from several threads are all kept;
* ``claim()`` and ``close()`` are True once, so one turn adopts a clock and one line is written;
* the line is ids and numbers only: marks in time order, then stages, then counts;
* a turn reference or a request id that does not fit its pattern is dropped, never stored;
* ``CHAT_TURN_TIMING_ENABLED`` false starts no clock.

Pure: no database, no network, no server.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from config import config
from core.monitoring import turn_clock
from core.monitoring.automatos_logging import request_id_var
from core.monitoring.turn_ref import clean_ref, clean_request_id

LONG_ID = "a" * 5000


@pytest.fixture(autouse=True)
def _no_clock_left_behind():
    """A test that begins a clock in the test's own context does not hand it to the next."""
    turn_clock.leave(None)
    yield
    turn_clock.leave(None)


@pytest.fixture
def timing_on(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", True)
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_STAGES_ENABLED", True)


class _Time:
    """A clock the test moves by hand, in place of ``time.perf_counter``."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def at(monkeypatch):
    moved = _Time()
    monkeypatch.setattr(turn_clock, "_now", moved)
    return moved


def _timing_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(f"{turn_clock.TIMING_LOG} ")]


# --------------------------------------------------------------------------- #
# The clock itself
# --------------------------------------------------------------------------- #


def test_a_mark_is_whole_milliseconds_since_the_start_and_the_first_of_a_name_wins(at):
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_REQUEST)

    at.now = 100.0414
    assert clock.mark("authorised") is True
    at.now = 100.5
    assert clock.mark("authorised") is False

    marks = clock.snapshot()["marks"]
    assert marks == {"authorised": 41}
    assert type(marks["authorised"]) is int


def test_a_clock_can_be_given_its_start(at):
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_TURN, started=99.0)

    assert clock.mark("turn") is True
    assert clock.snapshot()["marks"] == {"turn": 1000}
    assert clock.elapsed_seconds() == pytest.approx(1.0)


def test_a_stage_sums_over_its_calls_and_a_count_counts():
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_REQUEST)

    clock.add("tools", 0.2)
    clock.add("tools", 0.18)
    clock.add("context", 0.61)
    clock.count("model_calls")
    clock.count("model_calls")

    snapshot = clock.snapshot()
    assert snapshot["stages"] == {"tools": 380, "context": 610}
    assert snapshot["counts"] == {"model_calls": 2}


def test_a_snapshot_is_new_dicts_and_says_what_was_noted(at):
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_REQUEST)
    at.now = 100.041
    clock.mark("authorised")
    clock.add("tools", 0.38)
    clock.count("model_calls")
    clock.note(spoken=True, ref="vt-mhq2k1ab-3", request_id=None, outcome=None, text="never kept")

    first = clock.snapshot()
    assert first == {
        "v": 1,
        "origin": "request",
        "spoken": True,
        "ref": "vt-mhq2k1ab-3",
        "marks": {"authorised": 41},
        "stages": {"tools": 380},
        "counts": {"model_calls": 1},
    }

    first["marks"]["authorised"] = 9
    first["stages"]["tools"] = 9
    first["counts"]["model_calls"] = 9
    first["spoken"] = False
    assert clock.snapshot()["marks"] == {"authorised": 41}
    assert clock.snapshot()["stages"] == {"tools": 380}
    assert clock.snapshot()["counts"] == {"model_calls": 1}
    assert clock.snapshot()["spoken"] is True


def test_a_spoken_false_is_a_fact_and_a_none_is_not():
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_TURN)

    clock.note(spoken=False, ref=None)

    snapshot = clock.snapshot()
    assert snapshot["spoken"] is False
    assert "ref" not in snapshot and "request_id" not in snapshot and "outcome" not in snapshot


def test_claim_and_close_are_each_true_once():
    clock = turn_clock.TurnClock(turn_clock.ORIGIN_REQUEST)

    assert [clock.claim(), clock.claim(), clock.claim()] == [True, False, False]
    assert [clock.close(), clock.close(), clock.close()] == [True, False, False]


# --------------------------------------------------------------------------- #
# No clock: nothing happens
# --------------------------------------------------------------------------- #


def test_with_no_clock_every_call_does_nothing(caplog):
    assert turn_clock.current() is None

    with caplog.at_level(logging.DEBUG, logger=turn_clock.logger.name):
        assert turn_clock.mark("authorised") is False
        assert turn_clock.add("tools", 0.2) is None
        assert turn_clock.count("model_calls") is None
        assert turn_clock.note(spoken=True) is None
        assert turn_clock.finish("c1", "completed") is None
        assert turn_clock.elapsed_seconds() is None

    assert turn_clock.current() is None
    assert _timing_lines(caplog) == []


# --------------------------------------------------------------------------- #
# The clock rides the context
# --------------------------------------------------------------------------- #


def test_begin_then_leave_puts_the_previous_clock_back():
    assert turn_clock.current() is None
    outer = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
    assert turn_clock.current() is outer

    previous = turn_clock.current()
    inner = turn_clock.begin(turn_clock.ORIGIN_TURN)
    assert turn_clock.current() is inner and inner is not outer
    assert inner.origin == "turn" and outer.origin == "request"

    turn_clock.leave(previous)
    assert turn_clock.current() is outer
    turn_clock.leave(None)
    assert turn_clock.current() is None


def test_the_module_calls_write_to_the_contexts_clock(at):
    clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)

    at.now = 100.077
    assert turn_clock.mark("saved") is True
    assert turn_clock.mark("saved") is False
    turn_clock.add("model", 2.56)
    turn_clock.count("model_calls")
    turn_clock.note(spoken=True)
    at.now = 100.25

    assert turn_clock.elapsed_seconds() == pytest.approx(0.25)
    snapshot = clock.snapshot()
    assert snapshot["marks"] == {"saved": 77}
    assert snapshot["stages"] == {"model": 2560}
    assert snapshot["counts"] == {"model_calls": 1}
    assert snapshot["spoken"] is True


def test_two_tasks_each_see_their_own_clock():
    async def turn(name: str, started: asyncio.Event, other: asyncio.Event):
        clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
        started.set()
        await other.wait()  # both tasks have begun before either marks
        turn_clock.mark(name)
        return clock, turn_clock.current()

    async def go():
        first, second = asyncio.Event(), asyncio.Event()
        return await asyncio.gather(turn("one", first, second), turn("two", second, first))

    (clock_one, seen_one), (clock_two, seen_two) = asyncio.run(go())

    assert clock_one is seen_one and clock_two is seen_two and clock_one is not clock_two
    assert list(clock_one.snapshot()["marks"]) == ["one"]
    assert list(clock_two.snapshot()["marks"]) == ["two"]


def test_a_child_task_and_a_thread_write_to_their_parents_clock():
    async def child():
        turn_clock.mark("from_task")
        return turn_clock.current()

    def in_thread():
        turn_clock.mark("from_thread")
        return turn_clock.current()

    async def go():
        clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)
        seen_by_task = await asyncio.create_task(child())
        seen_by_thread = await asyncio.to_thread(in_thread)
        return clock, seen_by_task, seen_by_thread

    clock, seen_by_task, seen_by_thread = asyncio.run(go())

    assert seen_by_task is clock and seen_by_thread is clock
    assert set(clock.snapshot()["marks"]) == {"from_task", "from_thread"}


def test_two_hundred_marks_from_eight_threads_are_all_there():
    threads, each = 8, 25
    clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)

    def marks(thread: int) -> None:
        for number in range(each):
            turn_clock.mark(f"m{thread}_{number}")
            turn_clock.add("tools", 0.001)
            turn_clock.count("model_calls")

    with ThreadPoolExecutor(max_workers=threads) as pool:
        # Each thread runs in its own copy of this context, as asyncio.to_thread gives it.
        futures = [pool.submit(contextvars.copy_context().run, marks, thread) for thread in range(threads)]
        for future in futures:
            future.result()

    snapshot = clock.snapshot()
    assert set(snapshot["marks"]) == {f"m{thread}_{number}" for thread in range(threads) for number in range(each)}
    assert len(snapshot["marks"]) == 200
    assert snapshot["counts"] == {"model_calls": 200}
    assert snapshot["stages"] == {"tools": 200}


# --------------------------------------------------------------------------- #
# The line
# --------------------------------------------------------------------------- #


def test_finish_writes_one_line_with_the_outcome_and_marks_done(caplog):
    clock = turn_clock.begin(turn_clock.ORIGIN_REQUEST)

    with caplog.at_level(logging.INFO, logger=turn_clock.logger.name):
        turn_clock.finish("c1", "completed")
        turn_clock.finish("c1", "completed")

    lines = _timing_lines(caplog)
    assert len(lines) == 1, "a second finish writes nothing"
    assert lines[0].startswith("chat_turn_timing chat=c1 origin=request ")
    assert " outcome=completed " in lines[0] and " done=" in lines[0]
    assert "done" in clock.snapshot()["marks"]
    assert clock.snapshot()["outcome"] == "completed"
    assert clock.close() is False, "finish closed it"


def test_finish_with_no_outcome_says_so(caplog):
    turn_clock.begin(turn_clock.ORIGIN_TURN)

    with caplog.at_level(logging.INFO, logger=turn_clock.logger.name):
        turn_clock.finish("c2")

    lines = _timing_lines(caplog)
    assert len(lines) == 1
    assert lines[0].startswith("chat_turn_timing chat=c2 origin=turn spoken=- ref=- outcome=- done=")


def test_the_line_is_marks_in_time_order_then_stages_then_counts():
    snapshot = {
        "v": 1,
        "origin": "request",
        "spoken": True,
        "ref": "vt-mhq2k1ab-3",
        "request_id": "0123456789ab",
        "outcome": "completed",
        "marks": {"done": 6400, "saved": 77, "authorised": 41, "first_text": 4236},
        "stages": {"tools": 380, "model": 2560},
        "counts": {"model_calls": 1},
    }

    assert turn_clock.timing_line("c1", snapshot) == (
        "chat_turn_timing chat=c1 origin=request spoken=true ref=vt-mhq2k1ab-3 outcome=completed"
        " authorised=41 saved=77 first_text=4236 done=6400 | tools=380 model=2560 | model_calls=1"
    )


def test_a_line_with_no_stages_or_counts_has_no_empty_segment():
    bare = {"v": 1, "origin": "turn", "spoken": False, "marks": {"turn": 0, "done": 12}, "stages": {}, "counts": {}}
    assert turn_clock.timing_line("c1", bare) == (
        "chat_turn_timing chat=c1 origin=turn spoken=false ref=- outcome=- turn=0 done=12"
    )

    only_counts = {**bare, "counts": {"model_calls": 2}}
    assert turn_clock.timing_line("c1", only_counts).endswith(" turn=0 done=12 | model_calls=2")

    nothing = {"v": 1, "origin": "request", "marks": {}, "stages": {}, "counts": {}}
    assert turn_clock.timing_line("c1", nothing) == (
        "chat_turn_timing chat=c1 origin=request spoken=- ref=- outcome=-"
    )


def test_the_line_never_carries_text_that_is_not_an_id():
    said = "what is on my calendar today"
    snapshot = {"v": 1, "origin": "request", "ref": said, "outcome": said, "marks": {"done": 5}}

    line = turn_clock.timing_line(said, snapshot)

    assert line == "chat_turn_timing chat=- origin=request spoken=- ref=- outcome=- done=5"
    assert "calendar" not in turn_clock.timing_line(LONG_ID + " calendar", snapshot)


# --------------------------------------------------------------------------- #
# The ids
# --------------------------------------------------------------------------- #


def test_a_turn_ref_that_fits_is_kept():
    assert clean_ref("vt-mhq2k1ab-3") == "vt-mhq2k1ab-3"
    assert clean_ref("abcdef") == "abcdef"
    assert clean_ref("a" * 40) == "a" * 40


@pytest.mark.parametrize(
    "value",
    [
        "what is on my calendar today",  # a sentence
        "vt-1a",  # 5 characters
        "a" * 41,
        "vt-1a.b-3",  # a dot: what a fractional clock's base 36 would give
        "vt-mhq2k1ab-3\n",
        12345678,
        None,
        ["vt-mhq2k1ab-3"],
    ],
)
def test_a_turn_ref_that_does_not_fit_is_dropped(value):
    assert clean_ref(value) is None


def test_a_request_id_that_fits_is_kept():
    assert clean_request_id("0123456789ab") == "0123456789ab"


@pytest.mark.parametrize("value", [LONG_ID, "0123 456789ab", "", 123456789012, None, b"0123456789ab"])
def test_a_request_id_that_does_not_fit_is_dropped(value):
    assert clean_request_id(value) is None


# --------------------------------------------------------------------------- #
# The route dependencies and the switches
# --------------------------------------------------------------------------- #


def _arrive(request_id: str):
    """``arrives`` then ``authorised`` awaited in one task, as a route's dependencies are."""
    async def go():
        request_id_var.set(request_id)  # this task's context only
        await turn_clock.arrives()
        await turn_clock.authorised()
        return turn_clock.current()

    return asyncio.run(go())


def test_arrives_then_authorised_leave_a_request_clock_with_its_request_id(timing_on):
    clock = _arrive("0123456789ab")

    assert clock is not None
    snapshot = clock.snapshot()
    assert snapshot["origin"] == "request" == turn_clock.ORIGIN_REQUEST
    assert snapshot["request_id"] == "0123456789ab"
    assert list(snapshot["marks"]) == ["authorised"]
    assert turn_clock.current() is None, "the clock stayed in the request's own context"


def test_a_request_id_of_five_thousand_characters_is_not_stored(timing_on):
    clock = _arrive(LONG_ID)

    assert clock is not None
    assert "request_id" not in clock.snapshot()
    assert "authorised" in clock.snapshot()["marks"]


def test_with_timing_off_arrives_starts_no_clock(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", False)

    assert _arrive("0123456789ab") is None


def test_authorised_alone_starts_nothing():
    async def go():
        await turn_clock.authorised()
        return turn_clock.current()

    assert asyncio.run(go()) is None


@pytest.mark.parametrize(
    "timing, stages, on, stages_on",
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, False, False),
        (False, False, False, False),
    ],
)
def test_the_stages_need_both_switches(monkeypatch, timing, stages, on, stages_on):
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_ENABLED", timing)
    monkeypatch.setattr(config, "CHAT_TURN_TIMING_STAGES_ENABLED", stages)

    assert turn_clock.enabled() is on
    assert turn_clock.stages_enabled() is stages_on


def test_both_switches_are_on_by_default_and_named_in_the_config_surface():
    import json
    from pathlib import Path

    from config import Config

    assert Config.CHAT_TURN_TIMING_ENABLED is True
    assert Config.CHAT_TURN_TIMING_STAGES_ENABLED is True
    surface = json.loads((Path(__file__).resolve().parents[1] / "reports" / "config-surface.json").read_text())
    assert {"CHAT_TURN_TIMING_ENABLED", "CHAT_TURN_TIMING_STAGES_ENABLED"} <= set(surface["settings"])
