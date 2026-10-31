"""F458 (night 18): a full connection pool names who holds it.

At 08:01Z the pool (10 + 20) ran dry; the backend log had the TimeoutError and the
shutdown, but not one line about who held the 30 connections, so the cause could
not be proven from the night's data. Now, when a checkout leaves no connection
free, one WARNING (at most every DB_POOL_FULL_LOG_SECONDS) lists every holder,
oldest first: an asyncio task by its name, coroutine and await chain; a thread by
its name and stack; a holder that has ended without giving its connection back as
exactly that. Driven on a real QueuePool (SQLite file).
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest
import sqlalchemy as sa
from sqlalchemy.pool import QueuePool

from core.database import pool_watch

WAIT_S = 5
THIS_FILE = "tests/test_f458_full_pool_names_its_holders.py"


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def make(size: int):
        engine = sa.create_engine(
            f"sqlite:///{tmp_path / f'pool{len(engines)}.db'}", poolclass=QueuePool,
            pool_size=size, max_overflow=0, pool_timeout=1, connect_args={"check_same_thread": False},
        )
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.dispose()


def _reports(caplog) -> list:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[pool-watch] the connection pool is full")]


def test_a_full_pool_names_a_thread_and_a_task_that_hold_it_across_an_await(make_engine, caplog):
    engine = make_engine(3)
    watch = pool_watch.watch(engine, every_s=60)
    release, thread_holds = threading.Event(), threading.Event()

    def thread_holder():
        with engine.connect():
            thread_holds.set()
            release.wait(WAIT_S)

    holder = threading.Thread(target=thread_holder, name="holder-thread")
    holder.start()
    assert thread_holds.wait(WAIT_S)

    async def heartbeat_holder(stop: asyncio.Event):
        with engine.connect():  # the night-18 shape: a connection held across a long await
            await stop.wait()

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(heartbeat_holder(stop), name="hb-holder")
        await asyncio.sleep(0.05)
        with caplog.at_level(logging.WARNING, logger=pool_watch.__name__):
            with engine.connect():  # the checkout that leaves no connection free
                pass
        stop.set()
        await task

    try:
        asyncio.run(main())
    finally:
        release.set()
        holder.join(WAIT_S)

    (report,) = _reports(caplog)
    assert "(3 of 3 checked out)" in report
    assert "thread holder-thread" in report and f"{THIS_FILE}:" in report and "thread_holder" in report
    assert "task hb-holder (test_a_full_pool_names_a_thread_and_a_task_that_hold_it_across_an_await.<locals>.heartbeat_holder)" in report
    assert " heartbeat_holder" in report.split("task hb-holder", 1)[1], "the task's await chain names its own frame"
    assert watch.holder_count() == 0, "every connection given back is forgotten"


def test_a_holder_that_ended_without_giving_its_connection_back_is_named_as_such(make_engine, caplog):
    engine = make_engine(2)
    pool_watch.watch(engine, every_s=60)
    leaked = []

    async def leaks():
        leaked.append(engine.connect())  # never closed: the leak a restart is for

    asyncio.run(leaks())
    with caplog.at_level(logging.WARNING, logger=pool_watch.__name__):
        with engine.connect():
            pass
    leaked[0].close()
    (report,) = _reports(caplog)
    assert "a task that has finished without giving the connection back" in report


def test_one_report_per_interval(make_engine, caplog):
    engine = make_engine(1)
    pool_watch.watch(engine, every_s=60)
    with caplog.at_level(logging.WARNING, logger=pool_watch.__name__):
        for _ in range(3):
            with engine.connect():
                pass
    assert len(_reports(caplog)) == 1


def test_a_pool_with_room_is_never_reported(make_engine, caplog):
    engine = make_engine(2)
    pool_watch.watch(engine, every_s=60)
    with caplog.at_level(logging.WARNING, logger=pool_watch.__name__):
        with engine.connect():
            pass
    assert _reports(caplog) == []


def test_zero_attaches_nothing(make_engine, caplog):
    engine = make_engine(1)
    assert pool_watch.watch(engine, every_s=0) is None
    with caplog.at_level(logging.WARNING, logger=pool_watch.__name__):
        with engine.connect():
            pass
    assert _reports(caplog) == []


def test_bookkeeping_that_fails_never_fails_the_checkout(make_engine, monkeypatch):
    engine = make_engine(1)
    watch = pool_watch.watch(engine, every_s=60)

    def broken(_record):
        raise RuntimeError("bookkeeping broke")

    monkeypatch.setattr(watch, "_remember", broken)
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT 1")).scalar() == 1


def test_the_app_engine_is_watched():
    from config import config
    import core.database.database as database

    assert config.DB_POOL_FULL_LOG_SECONDS > 0
    assert isinstance(database.pool_holders, pool_watch.PoolWatch)
    assert sa.event.contains(database.engine, "checkout", database.pool_holders.on_checkout)
    assert sa.event.contains(database.engine, "checkin", database.pool_holders.on_return)
