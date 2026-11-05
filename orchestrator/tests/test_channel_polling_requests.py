"""A request that changes a channel starts or stops no poller in its own worker.

Connect, start, stop and delete used to start or stop a polling adapter in
whichever uvicorn worker served the request (connect even started the driver's
own handler-less poller there), so a connect added a second poller beside the
polling worker's and a disconnect stopped nothing unless it reached the same
worker. Now they write the row and hand it to ``channels.polling``: the owner
applies it at once, any other worker leaves it to the owner's tick. The list
answers from the running set every worker can read and no longer writes a
running adapter's state back into the row (that undid a stop saved elsewhere).

Fake sessions, drivers and adapters: no database, no Redis, no Telegram.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import api.channels as channels_api  # noqa: E402
from channels import polling, polling_status  # noqa: E402
from channels.drivers import ConnectivityMode, VerifyResult  # noqa: E402
from channels.manager import AdapterSpec, ChannelManager  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000aa"
CHANNEL = "11111111-2222-3333-4444-555555555555"
CTX = SimpleNamespace(workspace_id=WS)


class _Adapter:
    def __init__(self, connection_id, log):
        self.connection_id, self._log, self.is_running = connection_id, log, False

    async def start(self):
        self._log.append(("start", self.connection_id))
        self.is_running = True

    async def stop(self):
        self._log.append(("stop", self.connection_id))


def _manager(monkeypatch, *, owner: bool) -> ChannelManager:
    manager = ChannelManager()
    manager.log = []
    monkeypatch.setattr(
        manager, "_create_adapter",
        lambda connection_id, workspace_id, platform, config: _Adapter(connection_id, manager.log),
    )
    if owner:
        manager.take_polling_ownership()
    monkeypatch.setattr(polling, "get_channel_manager", lambda: manager)
    return manager


class _Driver:
    """A polling-capable driver that records every platform call."""

    calls: list = []

    def default_mode(self):
        return ConnectivityMode.POLLING

    def supports(self, mode):
        return True

    async def verify(self, *, workspace_id, config):
        return VerifyResult(ok=True, identity="bot")

    async def start_polling(self, **kwargs):
        _Driver.calls.append("start_polling")
        return True

    async def stop_polling(self, **kwargs):
        _Driver.calls.append("stop_polling")
        return True

    async def uninstall_webhook(self, **kwargs):
        _Driver.calls.append("uninstall_webhook")


class _DB:
    """Records db.execute(text(...), params); fetchone/fetchall are scripted."""

    def __init__(self, row=None, rows=()):
        self.calls: list = []
        self._row, self._rows = row, list(rows)
        self.commits = 0

    def execute(self, clause, params=None):
        self.calls.append((" ".join(str(clause).split()), dict(params or {})))
        return SimpleNamespace(fetchone=lambda: self._row, fetchall=lambda: self._rows)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def inserted_id(self) -> str:
        return next(p["id"] for sql, p in self.calls if sql.startswith("INSERT INTO channel_connections"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    _Driver.calls = []
    monkeypatch.setattr(channels_api, "get_driver", lambda platform: _Driver)
    monkeypatch.setattr(polling_status, "_redis", lambda: None)

    def _no_read():
        raise AssertionError("only the polling worker reads the desired set")

    monkeypatch.setattr(polling, "load_polling_specs", _no_read)


def _connect(db):
    return asyncio.run(channels_api.connect_channel_for_workspace(
        db, WS, "telegram", {"bot_token": "123:abc"}, mode="polling",
    ))


def test_a_polling_connect_on_another_worker_starts_no_poller_there(monkeypatch):
    manager = _manager(monkeypatch, owner=False)
    db = _DB()

    result = _connect(db)

    assert (result["status"], result["mode"]) == ("active", "polling")
    assert _Driver.calls == [], "the driver's own poller must not start in the request's worker"
    assert manager.log == [] and manager.running_ids() == ()


def test_a_polling_connect_on_the_polling_worker_starts_the_poller_at_once(monkeypatch):
    manager = _manager(monkeypatch, owner=True)
    db = _DB()
    monkeypatch.setattr(
        polling, "load_polling_specs",
        lambda: {db.inserted_id(): AdapterSpec(WS, "telegram", {"bot_token": "123:abc"})},
    )

    _connect(db)

    assert manager.running_ids() == (db.inserted_id(),)
    assert _Driver.calls == []


@pytest.mark.parametrize(
    ("route", "status", "reply"),
    [
        (channels_api.start_channel, "'active'", {"status": "started"}),
        (channels_api.stop_channel, "'inactive'", {"status": "stopped"}),
    ],
)
def test_start_and_stop_write_the_row_and_leave_the_poller_to_its_owner(monkeypatch, route, status, reply):
    manager = _manager(monkeypatch, owner=False)
    db = _DB(row=SimpleNamespace(id=CHANNEL))

    assert asyncio.run(route(CHANNEL, ctx=CTX, db=db)) == reply

    select_sql, select_params = db.calls[0]
    assert "workspace_id = :ws_id" in select_sql and select_params["ws_id"] == WS
    update_sql, update_params = db.calls[1]
    assert status in update_sql and update_params == {"id": CHANNEL}
    assert db.commits == 1
    assert manager.log == [], "no adapter may start or stop in the request's worker"


def test_delete_removes_the_row_before_the_owner_stops_its_poller(monkeypatch):
    db = _DB(row=SimpleNamespace(id=CHANNEL, platform="telegram", mode="polling", config={}))
    applied = []

    async def _apply(channel_id):
        applied.append((channel_id, [sql for sql, _ in db.calls]))
        return False

    monkeypatch.setattr(channels_api, "apply_polling_change", _apply)

    assert asyncio.run(channels_api.delete_channel(CHANNEL, ctx=CTX, db=db)) == {"status": "deleted"}

    assert _Driver.calls == [], "the driver's poller is not stopped from the request's worker"
    ((channel_id, sql_before_apply),) = applied
    assert channel_id == CHANNEL
    assert sql_before_apply[-1].startswith("DELETE FROM channel_connections")


def test_delete_of_a_webhook_channel_still_uninstalls_its_webhook(monkeypatch):
    _manager(monkeypatch, owner=False)
    db = _DB(row=SimpleNamespace(id=CHANNEL, platform="telegram", mode="webhook", config={}))

    asyncio.run(channels_api.delete_channel(CHANNEL, ctx=CTX, db=db))

    assert _Driver.calls == ["uninstall_webhook"]


def _list_row(status):
    return SimpleNamespace(
        id=CHANNEL, platform="telegram", status=status, mode="polling", webhook_url=None,
        last_verified=None, last_error=None, metadata={}, config={}, default_agent_id=None,
        message_count=0, last_activity_at=None, created_at=None,
    )


def test_the_list_reports_the_owners_running_poller_and_writes_nothing(monkeypatch):
    import channels.drivers as drivers

    monkeypatch.setattr(drivers, "list_platforms", lambda: [])
    monkeypatch.setattr(polling_status, "is_running", lambda connection_id: connection_id == CHANNEL)
    db = _DB(rows=[_list_row("inactive")])

    out = channels_api.list_channels(ctx=CTX, db=db)

    assert out[0]["status"] == "active", "a poller running on the polling worker reads as active anywhere"
    assert len(db.calls) == 1 and db.calls[0][0].startswith("SELECT"), "the list must not write the row back"
    assert db.commits == 0


def test_phase_2_starts_channel_polling_after_the_scheduler(monkeypatch):
    from fastapi import FastAPI

    import main
    from core.models.bootstrap import BootstrapReport, DeferredInitResult

    order = []

    async def _phase_2(app):
        order.append("scheduler")
        return DeferredInitResult(scheduler_started=True, dashboard_initialized=True)

    async def _channels():
        order.append("channels")
        return False

    monkeypatch.setattr(main, "_boot_phase_2_extensions", _phase_2)
    monkeypatch.setattr(polling, "start_channel_polling", _channels)
    app = FastAPI()

    asyncio.run(main._boot_phase_2_stage(app, BootstrapReport(), True))

    assert order == ["scheduler", "channels"], "the scheduler's lock decides who polls, so it goes first"
    assert app.state.deferred_init.scheduler_started is True
    assert app.state.deferred_init.channels_connected is False
