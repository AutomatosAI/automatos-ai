"""PRD-282 US-201: GET /health/ready answers 200 only when start-up has finished and the
last database check passed.

* The check is health_probe.database_status (F458), run in the background by a
  ``Watch`` (core/health_watch.py) on its own timer and time limit; the route reads
  its last result (core/health_readiness.py) and does no database work itself.
* "busy" is ready; "unhealthy" (unreachable, or no answer in time) and "stuck" are not.
* Start-up order: the READY stage waits for the first check, so the route never
  answers ready before a check was made.

Driven on main's real lifespan with every boot step stubbed (test_prd262_ready_after_core's
harness), on a real one-connection pool (test_f458's fixture), and on the Watch itself.
"""
from __future__ import annotations

import ast
import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import config
from core import health_probe, health_readiness
from core.health_watch import Reading, Watch
from tests.test_f458_health_never_waits_on_a_full_pool import one_connection_pool  # noqa: F401  (the fixture)
from tests.test_prd262_ready_after_core import WAIT_S, _Boot

ORCH = Path(__file__).resolve().parents[1]
AT_ONCE_S = 0.5  # what the route, or a check on a full pool, may take
LIMIT_S = 0.2    # a test watch's time limit


def _state(reading_state=None, *, ready=True):
    watch = health_readiness.database_watch()
    if reading_state is not None:
        watch.last = Reading(reading_state, datetime.now(timezone.utc))
    return SimpleNamespace(ready=ready, database_watch=watch)


def _route(state):
    """GET /health/ready through main's handler: the status code and the body."""
    import main

    answer = asyncio.run(main.readiness_probe(SimpleNamespace(app=SimpleNamespace(state=state))))
    if hasattr(answer, "status_code"):
        return answer.status_code, json.loads(answer.body)
    return 200, answer


# --------------------------------------------------------------------------- what each state answers


@pytest.mark.parametrize("database,code", [
    (health_probe.HEALTHY, 200),
    (health_probe.DATABASE_BUSY, 200),   # the pool is full, the database answers: still ready
    (health_probe.UNHEALTHY, 503),       # unreachable, or no answer within the time limit
    (health_probe.DATABASE_STUCK, 503),
])
def test_what_each_database_state_makes_ready_answer(database, code):
    status, body = health_readiness.answer(_state(database))
    assert status == code
    if code == 200:
        assert body == {"status": "ready"}
    else:
        assert body["status"] == "not_ready" and body["database"] == database and body["checked_at"]


def test_not_ready_while_starting_and_while_no_check_has_finished():
    assert health_readiness.answer(_state(health_probe.HEALTHY, ready=False)) == (
        503, {"status": "starting", "message": "Boot in progress"})
    assert health_readiness.answer(_state(None)) == (503, {"status": "not_ready", "database": "unchecked"})
    assert health_readiness.answer(SimpleNamespace(ready=True))[0] == 503  # no watch at all: never ready


# --------------------------------------------------------------------------- the route does no work


def test_with_a_full_pool_the_check_reads_busy_and_the_route_answers_at_once(
        one_connection_pool, monkeypatch):  # noqa: F811
    import core.database.database as database

    monkeypatch.setattr(config, "HEALTH_POOL_STUCK_SECONDS", 10 ** 9)  # busy, never stuck, here
    state = _state(None)
    held = one_connection_pool.connect()  # every connection checked out
    try:
        started = time.monotonic()
        reading = asyncio.run(state.database_watch.check_now())
        check_took = time.monotonic() - started

        def no_database_work(*_args, **_kwargs):
            raise AssertionError("the route checked the database itself")

        monkeypatch.setattr(database, "SessionLocal", no_database_work)
        monkeypatch.setattr(health_probe, "database_status", no_database_work)
        started = time.monotonic()
        answered = _route(state)
        route_took = time.monotonic() - started
    finally:
        held.close()
    assert reading.state == health_probe.DATABASE_BUSY and check_took < AT_ONCE_S
    assert answered == (200, {"status": "ready"})
    assert route_took < AT_ONCE_S


def test_the_route_reads_a_failed_check_as_503():
    status, body = _route(_state(health_probe.UNHEALTHY))
    assert status == 503 and body["database"] == health_probe.UNHEALTHY


def test_redis_is_not_part_of_readiness():
    """PRD-282 D-5: core serves without Redis, so readiness never asks for it."""
    for path in (ORCH / "core" / "health_readiness.py", ORCH / "core" / "health_watch.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                 for alias in node.names}
        modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("redis" in name for name in names | modules), path.name


# --------------------------------------------------------------------------- the watch


def _watch(check, **kwargs):
    return Watch("test", check, every_s=kwargs.pop("every_s", 60), limit_s=LIMIT_S, failed_state="down", **kwargs)


async def _until(condition, within_s=WAIT_S):
    """Waits for ``condition`` itself, never a fixed sleep; fails after ``within_s``."""
    deadline = time.monotonic() + within_s
    while not condition():
        assert time.monotonic() < deadline, "the awaited condition never came"
        await asyncio.sleep(0.01)


def test_a_hung_check_ends_its_wait_at_the_limit_holds_one_thread_and_its_late_answer_is_used():
    release, calls = threading.Event(), []

    def hangs():
        calls.append(threading.get_ident())
        release.wait(WAIT_S)
        return "up"

    watch = _watch(hangs)

    async def scenario():
        started = time.monotonic()
        first = await watch.check_now()
        took = time.monotonic() - started
        second = await watch.check_now()  # the first is still on its thread
        release.set()
        await _until(lambda: watch.last.state == "up")  # the late answer, recorded when it came
        late = watch.last
        third = await watch.check_now()
        return first, took, second, late, third

    try:
        first, took, second, late, third = asyncio.run(scenario())
    finally:
        release.set()
    assert first.state == "down" and LIMIT_S * 0.9 <= took < LIMIT_S + AT_ONCE_S
    assert second is first, "no new reading while the hung check runs: it ages"
    assert late.state == "up" and late.checked_at > first.checked_at
    assert third.state == "up"
    assert len(calls) == 2, "no second thread while the first check hung"


def test_a_check_runs_off_the_event_loop_and_one_that_raises_reads_as_failed(caplog):
    threads = []

    def records():
        threads.append(threading.get_ident())
        return "up"

    def raises():
        raise RuntimeError("refused")

    async def scenario():
        return threading.get_ident(), await _watch(records).check_now(), await _watch(raises).check_now()

    with caplog.at_level("WARNING", logger="core.health_watch"):
        loop_thread, ran, failed = asyncio.run(scenario())
    assert ran.state == "up" and threads and threads[0] != loop_thread
    assert failed.state == "down"
    assert [record.exc_info[1].args for record in caplog.records
            if record.name == "core.health_watch" and record.exc_info] == [("refused",)]


def test_a_failure_is_logged_when_the_state_changes_and_when_it_recovers_never_every_tick(caplog):
    answers = iter([RuntimeError("one"), RuntimeError("two"), RuntimeError("three"), "up", "up"])

    def check():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    watch = _watch(check)

    async def scenario():
        for _ in range(5):
            await watch.check_now()

    with caplog.at_level("INFO", logger="core.health_watch"):
        asyncio.run(scenario())
    lines = [(record.levelname, record.getMessage(), bool(record.exc_info)) for record in caplog.records
             if record.name == "core.health_watch"]
    assert lines == [("WARNING", "Health: test is down", True), ("INFO", "Health: test is up", False)]


def test_from_the_watch_a_failed_select_raises_and_logs_nothing_and_a_direct_probe_logs_as_before(
        monkeypatch, caplog):
    import core.database.database as database

    class Broken:
        def execute(self, *_args):
            raise RuntimeError("no database")

        def close(self):
            pass

    monkeypatch.setattr(database, "SessionLocal", Broken)
    monkeypatch.setattr(health_probe.pool_watch, "pool_has_room", lambda: True)
    with caplog.at_level("ERROR", logger="core.health_probe"):
        with pytest.raises(RuntimeError):
            health_probe.database_status(health_probe.FullPoolClock(), raise_errors=True)
        assert [record for record in caplog.records if record.name == "core.health_probe"] == []
        assert health_probe.database_status(health_probe.FullPoolClock()) == health_probe.UNHEALTHY
    assert [record.getMessage() for record in caplog.records
            if record.name == "core.health_probe"] == ["Health: database check failed"]


def test_a_callback_that_raises_once_neither_loses_the_reading_nor_stops_the_watch(caplog):
    told = []

    def on_reading(name, reading):
        told.append(reading)
        if len(told) == 1:
            raise RuntimeError("gauge down")

    watch = _watch(lambda: "up", every_s=0.01, on_reading=on_reading)

    async def scenario():
        task = asyncio.create_task(watch.run())
        await _until(lambda: len(told) >= 3)
        task.cancel()
        await asyncio.wait({task})
        return task

    with caplog.at_level("ERROR", logger="core.health_watch"):
        task = asyncio.run(scenario())
    assert watch.last is not None and watch.last.state == "up"
    assert task.cancelled(), "only the cancel ended it"
    assert any("the reading stands" in record.getMessage() for record in caplog.records)


def test_a_check_that_fails_outside_its_thread_does_not_stop_the_watch(monkeypatch):
    import core.health_watch as health_watch

    real, starts = health_watch._on_a_thread, []

    def fails_once(name, check):
        starts.append(name)
        if len(starts) == 1:
            raise RuntimeError("no thread")
        return real(name, check)

    monkeypatch.setattr(health_watch, "_on_a_thread", fails_once)
    watch = _watch(lambda: "up", every_s=0.01)

    async def scenario():
        task = asyncio.create_task(watch.run())
        await _until(lambda: watch.last is not None)
        task.cancel()
        await asyncio.wait({task})

    asyncio.run(scenario())
    assert watch.last.state == "up" and len(starts) >= 2


def test_a_stale_healthy_reading_is_no_passed_check():
    state = _state(health_probe.HEALTHY)
    watch = state.database_watch
    watch.last = Reading(health_probe.HEALTHY, datetime.now(timezone.utc) - timedelta(seconds=watch.stale_after_s + 1))
    assert watch.stale_after_s == 3 * config.HEALTH_READY_CHECK_SECONDS + config.HEALTH_READY_CHECK_TIMEOUT_SECONDS
    assert health_readiness.answer(state) == (503, {"status": "not_ready", "database": "unchecked"})


# --------------------------------------------------------------------------- start-up order, on main's lifespan


def _boot_until_ready(boot):
    """The lifespan, the background boot let through, and what readiness says at READY."""
    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            at_ready = (boot.app.state.ready, boot.app.state.database_watch.last, await boot.ready_probe())
            task = boot.app.state.database_watch_task
        return at_ready, task

    return asyncio.run(scenario())


def test_ready_is_never_answered_before_the_first_database_check(monkeypatch):
    """The first check takes longer than the whole (stubbed) boot: READY waits for it."""
    boot = _Boot(monkeypatch)

    def slow_but_healthy(**_kwargs):
        time.sleep(LIMIT_S)
        return health_probe.HEALTHY

    monkeypatch.setattr(health_probe, "database_status", slow_but_healthy)
    (ready, reading, probe), task = _boot_until_ready(boot)
    assert ready is True and reading is not None and reading.state == health_probe.HEALTHY
    assert probe == (200, {"status": "ready"})
    assert task.cancelled(), "the shutdown stopped the database watch"


def test_an_install_whose_database_does_not_answer_is_not_ready_after_its_boot(monkeypatch):
    boot = _Boot(monkeypatch)
    boot.database = health_probe.UNHEALTHY
    (ready, reading, (status, body)), _ = _boot_until_ready(boot)
    assert ready is True and reading.state == health_probe.UNHEALTHY
    assert status == 503 and json.loads(body)["database"] == health_probe.UNHEALTHY


def test_readiness_follows_the_database_back_on_the_watchs_timer(monkeypatch):
    boot = _Boot(monkeypatch)
    boot.database = health_probe.UNHEALTHY
    monkeypatch.setattr(config, "HEALTH_READY_CHECK_SECONDS", 0.05)

    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            down = (await boot.ready_probe())[0]
            boot.database = health_probe.DATABASE_BUSY  # back, and busy: ready
            await _until(lambda: boot.app.state.database_watch.last.state == health_probe.DATABASE_BUSY)
            return down, await boot.ready_probe()

    down, back = asyncio.run(scenario())
    assert down == 503
    assert back == (200, {"status": "ready"})
