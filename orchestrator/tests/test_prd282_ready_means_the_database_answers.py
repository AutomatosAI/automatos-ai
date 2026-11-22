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
from datetime import datetime, timezone
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


def _watch(check):
    return Watch("test", check, every_s=60, limit_s=LIMIT_S, failed_state="down")


def test_a_check_that_hangs_ends_its_wait_at_the_time_limit_and_is_never_started_twice():
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
        await asyncio.sleep(0.05)
        third = await watch.check_now()
        return first, took, second, third

    try:
        first, took, second, third = asyncio.run(scenario())
    finally:
        release.set()
    assert first.state == "down" and LIMIT_S * 0.9 <= took < LIMIT_S + AT_ONCE_S
    assert second.state == "down"
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

    with caplog.at_level("ERROR", logger="core.health_watch"):
        loop_thread, ran, failed = asyncio.run(scenario())
    assert ran.state == "up" and threads and threads[0] != loop_thread
    assert failed.state == "down"
    assert any("the test check failed" in record.getMessage() for record in caplog.records)


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

    def slow_but_healthy():
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
            await asyncio.sleep(0.3)
            return down, await boot.ready_probe()

    down, back = asyncio.run(scenario())
    assert down == 503
    assert back == (200, {"status": "ready"})
