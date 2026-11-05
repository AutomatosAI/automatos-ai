"""One worker owns every polling-mode channel adapter.

The backend runs several uvicorn workers. Every worker used to start every
polling adapter at boot (``ChannelManager.start_all()`` outside the scheduler's
lock), so with N workers a Telegram bot had N getUpdates pollers: Telegram
answered 409 Conflict and messages were split or lost. Now the worker that runs
the unified scheduler reconciles the adapters with the database at boot and on
a tick (``channels/polling.py``), and publishes the running ones so every
worker's ``is_running`` answers truly (``channels/polling_status.py``).

No database and no Redis: the desired set, the adapters and Redis are fakes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from channels import polling, polling_status  # noqa: E402
from channels.manager import START_RETRY_SECONDS, AdapterSpec, ChannelManager  # noqa: E402
from config import config  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000aa"


def _spec(token: str = "tok-1", platform: str = "telegram") -> AdapterSpec:
    return AdapterSpec(WS, platform, {"bot_token": token})


class _Adapter:
    """Stands in for TelegramAdapter: records start/stop, can be told to fail."""

    def __init__(self, connection_id, config, log, fail):
        self.connection_id, self.config, self._log, self._fail = connection_id, config, log, fail
        self.is_running = False

    async def start(self):
        self._log.append(("start", self.connection_id, self.config.get("bot_token")))
        if self._fail:
            raise RuntimeError("Unauthorized: invalid token")
        self.is_running = True

    async def stop(self):
        self._log.append(("stop", self.connection_id))
        self.is_running = False


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _manager(monkeypatch, *, failing=(), owner=False):
    """A ChannelManager whose adapter factory builds fakes; ``failing`` tokens fail to start."""
    clock = _Clock()
    manager = ChannelManager(clock=clock)
    manager.log = []
    manager.clock = clock

    def _factory(connection_id, workspace_id, platform, config):
        return _Adapter(connection_id, config, manager.log, config.get("bot_token") in failing)

    monkeypatch.setattr(manager, "_create_adapter", _factory)
    if owner:
        manager.take_polling_ownership()
    return manager


class _Pipe:
    def __init__(self, redis):
        self._redis, self._ops = redis, []

    def set(self, key, value, ex=None):
        self._ops.append(("set", key, ex))
        return self

    def delete(self, key):
        self._ops.append(("delete", key, None))
        return self

    def execute(self):
        for op, key, ex in self._ops:
            if op == "set":
                self._redis.keys[key] = ex
            else:
                self._redis.keys.pop(key, None)


class _Redis:
    """The redis-py slice polling_status uses; ``keys`` maps a key to its TTL."""

    def __init__(self):
        self.keys: dict = {}
        self.down = False

    def pipeline(self, transaction=False):
        return _Pipe(self)

    def exists(self, key):
        if self.down:
            raise ConnectionError("redis unavailable")
        return int(key in self.keys)


@pytest.fixture()
def redis(monkeypatch):
    fake = _Redis()
    monkeypatch.setattr(polling_status, "_redis", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _no_drivers(monkeypatch):
    """is_running also asks every channel driver; none runs a poller here."""
    import channels.drivers as drivers

    monkeypatch.setattr(drivers, "list_platforms", lambda: [])


def _desired(monkeypatch, manager, desired):
    """Point channels.polling at ``manager`` and a desired set (a dict, mutated by the test)."""
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)
    monkeypatch.setattr(polling, "load_polling_specs", lambda: dict(desired))


# ── ChannelManager.reconcile ────────────────────────────────────────────────


def test_reconcile_starts_what_should_run_and_stops_what_should_not(monkeypatch):
    manager = _manager(monkeypatch)

    first = asyncio.run(manager.reconcile({"a": _spec("tok-a"), "b": _spec("tok-b")}))
    assert first.started == ("a", "b") and not first.stopped
    assert set(manager.running_ids()) == {"a", "b"}

    second = asyncio.run(manager.reconcile({"b": _spec("tok-b")}))
    assert second.stopped == ("a",) and not second.started and not second.restarted
    assert manager.running_ids() == ("b",)
    assert manager.log[-1] == ("stop", "a")


def test_reconcile_restarts_an_adapter_whose_config_changed_and_leaves_the_rest(monkeypatch):
    manager = _manager(monkeypatch)
    asyncio.run(manager.reconcile({"a": _spec("tok-1"), "b": _spec("tok-b")}))
    manager.log.clear()

    same = asyncio.run(manager.reconcile({"a": _spec("tok-1"), "b": _spec("tok-b")}))
    assert not same.changed and manager.log == [], "an unchanged spec must not restart anything"

    rotated = asyncio.run(manager.reconcile({"a": _spec("tok-2"), "b": _spec("tok-b")}))
    assert rotated.restarted == ("a",) and not rotated.started
    assert manager.log == [("stop", "a"), ("start", "a", "tok-2")]


def test_a_failed_start_waits_before_it_is_tried_again(monkeypatch):
    manager = _manager(monkeypatch, failing={"revoked"})
    desired = {"a": _spec("revoked")}

    assert asyncio.run(manager.reconcile(desired)).failed == ("a",)
    manager.clock.now += START_RETRY_SECONDS - 1
    assert not asyncio.run(manager.reconcile(desired)).changed, "retried before the wait was up"

    manager.clock.now += 1
    assert asyncio.run(manager.reconcile(desired)).failed == ("a",), "not retried after the wait"


def test_a_failed_start_is_tried_at_once_when_its_spec_changes_or_someone_acts_on_it(monkeypatch):
    manager = _manager(monkeypatch, failing={"revoked"})
    asyncio.run(manager.reconcile({"a": _spec("revoked")}))

    assert asyncio.run(manager.reconcile({"a": _spec("revoked")}, retry=("a",))).failed == ("a",)
    assert asyncio.run(manager.reconcile({"a": _spec("fresh")})).started == ("a",)


def test_a_failed_restart_leaves_nothing_reading_as_running(monkeypatch, redis):
    manager = _manager(monkeypatch, failing={"revoked"})
    asyncio.run(manager.reconcile({"a": _spec("good")}))

    report = asyncio.run(manager.reconcile({"a": _spec("revoked")}))
    assert report.failed == ("a",)
    assert manager.running_ids() == ()
    assert manager.is_running("a") is False


# ── channels.polling: the pass, the tick, the published set ─────────────────


def test_load_polling_specs_reads_only_active_polling_rows(monkeypatch):
    seen = {}
    rows = [
        SimpleNamespace(id="c-1", workspace_id=WS, platform="telegram", config={"bot_token": "t"}),
        SimpleNamespace(id="c-2", workspace_id=WS, platform="telegram", config='{"bot_token": "s"}'),
    ]

    class _Query:
        def filter(self, *criteria):
            seen["criteria"] = sorted(
                str(c.compile(compile_kwargs={"literal_binds": True})) for c in criteria
            )
            return self

        def all(self):
            return rows

    class _Session:
        def query(self, *columns):
            return _Query()

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr("core.database.database.SessionLocal", _Session)

    specs = polling.load_polling_specs()

    assert seen["criteria"] == [
        "channel_connections.mode = 'polling'",
        "channel_connections.status = 'active'",
    ], "webhook-mode and inactive rows must never get a poller"
    assert seen["closed"] is True
    assert specs == {"c-1": _spec("t"), "c-2": _spec("s")}


def test_a_pass_publishes_the_running_adapters_and_drops_the_stopped(monkeypatch, redis):
    manager = _manager(monkeypatch, owner=True)
    desired = {"a": _spec("tok-a")}
    _desired(monkeypatch, manager, desired)

    asyncio.run(polling.reconcile_polling())
    key = polling_status.running_key("a")
    assert redis.keys == {key: config.CHANNEL_POLLING_RECONCILE_SECONDS * polling_status.TTL_INTERVALS}

    desired.clear()
    asyncio.run(polling.reconcile_polling())
    assert redis.keys == {}


def test_a_database_error_stops_no_adapter(monkeypatch, redis):
    manager = _manager(monkeypatch, owner=True)
    _desired(monkeypatch, manager, {"a": _spec()})
    asyncio.run(polling.reconcile_polling())

    def _db_down():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(polling, "load_polling_specs", _db_down)
    with pytest.raises(RuntimeError):
        asyncio.run(polling.reconcile_polling())
    assert manager.running_ids() == ("a",), "an unreadable desired set must not read as empty"


def test_the_tick_runs_a_pass_and_survives_a_failed_one(monkeypatch, caplog):
    manager = _manager(monkeypatch, owner=True)
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)
    passes = []

    async def _pass(**kwargs):
        passes.append(kwargs)
        if len(passes) == 2:
            raise RuntimeError("database unreachable")

    monkeypatch.setattr(polling, "reconcile_polling", _pass)
    asyncio.run(polling._tick())
    with caplog.at_level(logging.ERROR, logger="channels.polling"):
        asyncio.run(polling._tick())

    assert len(passes) == 2
    assert any("reconcile pass failed" in r.getMessage() and r.exc_info for r in caplog.records)


def test_a_kept_tick_does_nothing_on_a_worker_that_does_not_own_polling(monkeypatch):
    """A Redis job store keeps the job across restarts, also after channels were turned off."""
    manager = _manager(monkeypatch, owner=False)
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)

    async def _pass(**kwargs):
        raise AssertionError("no pass on a worker that does not own polling")

    monkeypatch.setattr(polling, "reconcile_polling", _pass)
    asyncio.run(polling._tick())


# ── Boot: who owns polling ───────────────────────────────────────────────────


class _Scheduler:
    running = True

    def __init__(self):
        self.jobs = {}

    def add_job(self, func, trigger=None, id=None, **kwargs):  # noqa: A002 — APScheduler's name
        self.jobs[id] = (func, trigger, kwargs)


@pytest.fixture()
def interval_trigger(monkeypatch):
    """APScheduler's IntervalTrigger, faked so the test needs no real scheduler."""
    module = types.ModuleType("apscheduler.triggers.interval")
    module.IntervalTrigger = lambda seconds: SimpleNamespace(seconds=seconds)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers.interval", module)


def _scheduler_flags(monkeypatch, on: bool):
    for name in ("HEARTBEAT_ENABLED", "RECIPE_SCHEDULER_ENABLED", "COORDINATOR_ENABLED"):
        monkeypatch.setattr(config, name, on)
    monkeypatch.setattr(config, "CHANNELS_ENABLED", True)


def test_the_scheduler_worker_owns_polling_starts_its_adapters_and_ticks(monkeypatch, redis, interval_trigger):
    _scheduler_flags(monkeypatch, True)
    scheduler = _Scheduler()
    monkeypatch.setattr(polling, "_scheduler_on_this_worker", lambda: scheduler)
    manager = _manager(monkeypatch)
    _desired(monkeypatch, manager, {"a": _spec()})

    assert asyncio.run(polling.start_channel_polling()) is True

    assert manager.owns_polling is True
    assert manager.running_ids() == ("a",), "the boot pass must start the adapters"
    func, trigger, kwargs = scheduler.jobs[polling.RECONCILE_JOB_ID]
    assert func is polling._tick
    assert trigger.seconds == config.CHANNEL_POLLING_RECONCILE_SECONDS
    assert "args" not in kwargs and kwargs["replace_existing"] is True and kwargs["max_instances"] == 1


def test_another_worker_starts_no_poller_at_boot(monkeypatch, redis):
    _scheduler_flags(monkeypatch, True)
    monkeypatch.setattr(polling, "_scheduler_on_this_worker", lambda: None)
    manager = _manager(monkeypatch)
    _desired(monkeypatch, manager, {"a": _spec()})

    assert asyncio.run(polling.start_channel_polling()) is True

    assert manager.owns_polling is False
    assert manager.running_ids() == () and manager.log == []


def test_without_the_scheduler_every_worker_polls_as_before_and_says_so(monkeypatch, redis, caplog):
    _scheduler_flags(monkeypatch, False)

    def _no_scheduler():
        raise AssertionError("no lock is taken when the scheduler is disabled")

    monkeypatch.setattr(polling, "_scheduler_on_this_worker", _no_scheduler)
    manager = _manager(monkeypatch)
    _desired(monkeypatch, manager, {"a": _spec()})

    with caplog.at_level(logging.WARNING, logger="channels.polling"):
        assert asyncio.run(polling.start_channel_polling()) is True

    assert manager.owns_polling is True and manager.running_ids() == ("a",)
    warning = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "more than one uvicorn worker" in warning and "409" in warning


def test_channels_disabled_starts_nothing(monkeypatch):
    monkeypatch.setattr(config, "CHANNELS_ENABLED", False)

    def _untouched():
        raise AssertionError("nothing may be read with channels disabled")

    monkeypatch.setattr(polling, "get_channel_manager", _untouched)
    assert asyncio.run(polling.start_channel_polling()) is True


def test_a_failed_boot_pass_reports_false(monkeypatch, redis, interval_trigger):
    _scheduler_flags(monkeypatch, True)
    monkeypatch.setattr(polling, "_scheduler_on_this_worker", lambda: _Scheduler())
    manager = _manager(monkeypatch)
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)

    def _db_down():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(polling, "load_polling_specs", _db_down)
    assert asyncio.run(polling.start_channel_polling()) is False
    assert manager.owns_polling is True, "the tick still owns polling and retries"


# ── A change written on some worker ─────────────────────────────────────────


def test_a_change_on_another_worker_starts_no_poller_there(monkeypatch, redis):
    manager = _manager(monkeypatch)

    def _no_read():
        raise AssertionError("a non-owner must not even read the desired set")

    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)
    monkeypatch.setattr(polling, "load_polling_specs", _no_read)

    assert asyncio.run(polling.apply_polling_change("a")) is False
    assert manager.running_ids() == () and manager.log == []


def test_a_change_on_the_owner_applies_now_and_retries_that_connection(monkeypatch, redis):
    manager = _manager(monkeypatch, failing={"revoked"}, owner=True)
    desired = {"a": _spec("revoked")}
    _desired(monkeypatch, manager, desired)
    asyncio.run(polling.reconcile_polling())
    manager.log.clear()

    assert asyncio.run(polling.apply_polling_change("a")) is True
    assert manager.log == [("start", "a", "revoked")], "the owner retries what the user acted on"

    desired["a"] = _spec("fixed")
    asyncio.run(polling.apply_polling_change("a"))
    assert manager.running_ids() == ("a",)


def test_a_failed_apply_on_the_owner_does_not_fail_the_request(monkeypatch, caplog):
    manager = _manager(monkeypatch, owner=True)
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)

    async def _pass(**kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(polling, "reconcile_polling", _pass)
    with caplog.at_level(logging.ERROR, logger="channels.polling"):
        assert asyncio.run(polling.apply_polling_change("a")) is False
    assert any(r.exc_info for r in caplog.records)


# ── Status on every worker ───────────────────────────────────────────────────


def test_another_worker_sees_the_owners_running_adapters(monkeypatch, redis):
    owner = _manager(monkeypatch, owner=True)
    desired = {"a": _spec()}
    _desired(monkeypatch, owner, desired)
    asyncio.run(polling.reconcile_polling())

    elsewhere = ChannelManager()
    assert elsewhere.running_ids() == ()
    assert elsewhere.is_running("a") is True, "a non-owner must not report a running bot as stopped"
    assert elsewhere.is_running("b") is False

    desired.clear()
    asyncio.run(polling.reconcile_polling())
    assert elsewhere.is_running("a") is False


def test_a_redis_error_does_not_hide_a_local_adapter_and_is_logged(monkeypatch, redis, caplog):
    manager = _manager(monkeypatch)
    asyncio.run(manager.reconcile({"a": _spec()}))
    redis.down = True

    with caplog.at_level(logging.WARNING, logger="channels.manager"):
        assert manager.is_running("a") is True
        assert manager.is_running("b") is False

    assert any("running set" in r.getMessage() and r.exc_info for r in caplog.records)


def test_without_redis_nothing_is_published_and_nothing_reads_as_running(monkeypatch):
    monkeypatch.setattr(polling_status, "_redis", lambda: None)
    polling_status.publish(["a"], ["b"])
    assert polling_status.is_running("a") is False


def test_shutdown_stops_the_owners_adapters_and_drops_them(monkeypatch, redis):
    manager = _manager(monkeypatch, owner=True)
    _desired(monkeypatch, manager, {"a": _spec(), "b": _spec("tok-b")})
    asyncio.run(polling.reconcile_polling())
    assert len(redis.keys) == 2

    asyncio.run(polling.stop_channel_polling())

    assert manager.running_ids() == ()
    assert ("stop", "a") in manager.log and ("stop", "b") in manager.log
    assert redis.keys == {}

    manager.log.clear()
    asyncio.run(polling.apply_polling_change("a"))
    asyncio.run(polling.reconcile_polling())
    assert manager.log == [] and manager.running_ids() == (), "nothing may start again after shutdown"
