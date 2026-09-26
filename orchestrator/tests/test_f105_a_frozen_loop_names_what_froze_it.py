"""F105 (26 Sep): a frozen event loop names what froze it.

Every F105 freezer so far was found by reading gaps in the backend log, and one
(4.6 s in the first memory turn after a restart) could not be named that way.
The loop watchdog logs one WARNING per stall, with the stall's length and the
loop thread's stack at the time (code lines only), at most so many a minute.

The tests freeze their own loop with a sync sleep and read the log. No app.
"""
import asyncio
import logging
import re
import threading
import time
from pathlib import Path

import pytest

try:
    from core import loop_watchdog
except ImportError:  # the code before the watchdog: nothing watches the loop
    loop_watchdog = None

MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"


def _freeze_the_loop_for(seconds):
    time.sleep(seconds)


def _stall_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "[loop-watchdog]" in r.getMessage()]


@pytest.fixture
def watch():
    """Start a watchdog on the running loop; every one started is stopped."""
    started = []

    def _start(**options):
        dog = loop_watchdog.start_loop_watchdog(**options) if loop_watchdog else None
        started.append(dog)
        return dog

    yield _start
    for dog in started:
        if dog is not None:
            dog.stop()


@pytest.mark.asyncio
async def test_a_stall_is_logged_once_with_the_code_that_stalled_it(watch, caplog):
    with caplog.at_level(logging.WARNING):
        watch(stall_s=0.3, ping_s=0.05)
        await asyncio.sleep(0.2)
        _freeze_the_loop_for(0.8)
        await asyncio.sleep(0.3)

    warnings = _stall_warnings(caplog)
    assert len(warnings) == 1, f"one stall, {len(warnings)} warnings: {warnings}"
    assert float(re.search(r"stood still for (\d+\.\d)s", warnings[0]).group(1)) >= 0.6
    assert "in _freeze_the_loop_for" in warnings[0] and "time.sleep(seconds)" in warnings[0]


@pytest.mark.asyncio
async def test_nothing_is_logged_while_the_loop_answers(watch, caplog):
    with caplog.at_level(logging.WARNING):
        watch(stall_s=0.3, ping_s=0.05)
        for _ in range(5):
            _freeze_the_loop_for(0.1)
            await asyncio.sleep(0.05)

    assert _stall_warnings(caplog) == []


@pytest.mark.asyncio
async def test_off_starts_nothing(watch):
    before = {t.name for t in threading.enumerate()}

    assert watch(stall_s=0) is None
    assert "loop-watchdog" not in {t.name for t in threading.enumerate()} - before


@pytest.mark.asyncio
async def test_stalls_over_the_rate_are_counted_not_dumped(watch, caplog):
    with caplog.at_level(logging.WARNING):
        watch(stall_s=0.1, ping_s=0.02, dumps_per_window=1, window_s=2.0)
        for _ in range(3):
            _freeze_the_loop_for(0.3)
            await asyncio.sleep(0.15)
        await asyncio.sleep(2.2)
        _freeze_the_loop_for(0.3)
        await asyncio.sleep(0.15)

    first, later = _stall_warnings(caplog)
    assert "more stalls" not in first
    assert "(2 more stalls since the last one were not dumped: over 1 a minute)" in later


@pytest.mark.asyncio
async def test_a_hang_is_named_while_it_lasts(watch, caplog, monkeypatch):
    monkeypatch.setattr(loop_watchdog, "STILL_STALLED_S", 0.5)
    with caplog.at_level(logging.WARNING):
        watch(stall_s=0.2, ping_s=0.05)
        await asyncio.sleep(0.1)
        _freeze_the_loop_for(1.2)
        (warning,) = _stall_warnings(caplog)  # logged while the loop was still frozen
        await asyncio.sleep(0.3)

    assert "has stood still, and still does," in warning and "in _freeze_the_loop_for" in warning
    assert len(_stall_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_stop_on_the_loops_own_thread_ends_it_at_once(watch):
    dog = watch(stall_s=5, ping_s=0.05)
    await asyncio.sleep(0.1)

    started = time.monotonic()
    dog.stop()

    assert time.monotonic() - started < 0.5
    assert not dog.is_alive()


def test_the_watchdog_ends_quietly_when_its_loop_closes(caplog):
    loop = asyncio.new_event_loop()

    async def _start():
        return loop_watchdog.start_loop_watchdog(stall_s=0.2, ping_s=0.02)

    with caplog.at_level(logging.WARNING):
        dog = loop.run_until_complete(_start())
        loop.close()
        deadline = time.monotonic() + 2
        while dog.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)

    assert not dog.is_alive()
    assert _stall_warnings(caplog) == []  # a closed loop is no stall


def test_the_app_watches_its_loop_once_ready_and_stops_at_shutdown():
    source = MAIN_PY.read_text()
    ready, started = source.index("app.state.ready = True"), source.index("start_loop_watchdog()")
    shutdown, stopped = source.index("Shutting down Automotas AI API Server"), source.index("_watchdog.stop()")

    assert ready < started < shutdown < stopped
