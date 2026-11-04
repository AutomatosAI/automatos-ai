"""F105 lane H — a helper a threadpool route reaches starts its background work on the event loop.

The F105 sweep made 479 routes plain ``def``: FastAPI runs them in its threadpool (anyio's
worker threads), where no event loop runs. A sync helper those routes reach that started
background work with ``create_task`` / ``call_later`` raised "no running event loop" there,
or skipped the work behind a ``get_running_loop`` check: a board ticket was never
dispatched, an agent never re-embedded, a Shopify catalog webhook never re-synced, a
Knowledge Graph rebuild never scheduled, the AI credit never looked at when a suggestion
was started. ``core.utils.background_tasks.on_the_event_loop`` (F296's fix, moved to core)
now runs the starting step on the loop's own thread.

Each helper is called the way a route calls it (``anyio.to_thread.run_sync`` inside a
running loop) and the work must run on the loop's thread; on the loop, and with no loop at
all (a script, a plain test), it behaves as before.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest

import core.utils.background_tasks as bg

SETTLE_SECONDS = 2.0
WS = "ws-f105-h"


async def _until(done, seconds: float = SETTLE_SECONDS) -> None:
    """Let the loop run until ``done()``; fail after ``seconds``."""
    with anyio.fail_after(seconds):
        while not done():
            await anyio.sleep(0.01)


def _from_a_route_thread(call, done):
    """Run ``call`` the way FastAPI runs a plain ``def`` route, then let the loop run until
    ``done()``. Returns the loop's thread id and ``call``'s answer."""
    async def _route():
        loop_thread = threading.get_ident()
        answer = await anyio.to_thread.run_sync(call)
        await _until(done)
        return loop_thread, answer

    return anyio.run(_route)


# ---------------------------------------------------------------------------
# on_the_event_loop (core)
# ---------------------------------------------------------------------------


def test_from_a_route_thread_the_start_runs_on_the_loop_and_answers():
    seen = []

    def start():
        asyncio.get_running_loop()  # what create_task needs; raises in a worker thread
        seen.append(threading.get_ident())
        return "started"

    loop_thread, answer = _from_a_route_thread(lambda: bg.on_the_event_loop(start), lambda: bool(seen))
    assert answer == "started" and seen == [loop_thread]


def test_on_the_loop_and_with_no_loop_the_start_runs_where_it_is_asked():
    async def on_the_loop():
        return bg.on_the_event_loop(threading.get_ident), threading.get_ident()

    ran_in, loop_thread = anyio.run(on_the_loop)
    assert ran_in == loop_thread
    assert bg.on_the_event_loop(threading.get_ident) == threading.get_ident()


def test_the_playbook_engine_starts_through_the_core_helper():
    """Replace, don't shim: the engine's own copy is gone; it uses the one in core."""
    import services.playbook_engine as engine

    assert engine.on_the_event_loop is bg.on_the_event_loop
    assert "def on_the_event_loop" not in open(engine.__file__, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# launch_guarded (core): board dispatch, socials renders, wizard scans, copies
# ---------------------------------------------------------------------------


def test_a_guarded_launch_from_a_route_thread_runs_its_task_on_the_loop(monkeypatch):
    monkeypatch.setattr(bg, "record_error", MagicMock())
    ran = []

    async def work():
        ran.append(threading.get_ident())

    def launch():
        return bg.launch_guarded(work(), subsystem="board", operation="execute_task")

    loop_thread, task = _from_a_route_thread(launch, lambda: bool(ran))
    assert ran == [loop_thread]
    assert isinstance(task, asyncio.Task)


def test_a_guarded_launch_on_the_loop_is_held_until_done(monkeypatch):
    monkeypatch.setattr(bg, "record_error", MagicMock())

    async def on_the_loop():
        task = bg.launch_guarded(asyncio.sleep(0), subsystem="x", operation="y")
        held = task in bg._BACKGROUND_TASKS
        await task
        await asyncio.sleep(0)
        return held, task

    held, task = anyio.run(on_the_loop)
    assert held and task not in bg._BACKGROUND_TASKS


def test_a_guarded_launch_with_no_loop_raises_like_create_task():
    coro = asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        bg.launch_guarded(coro, subsystem="x", operation="y")
    coro.close()


# ---------------------------------------------------------------------------
# api/agents._reindex_agent_embedding
# ---------------------------------------------------------------------------


@pytest.fixture
def embeds(monkeypatch):
    """The re-embed with its session and the embedding call stood in for."""
    import core.database.database as database
    import core.routing.semantic_indexer as indexer

    agent = SimpleNamespace(id=7)
    session = MagicMock()
    session.query.return_value.get.return_value = agent
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    seen = []

    async def embed_agent(found, db):
        seen.append((found.id, threading.get_ident()))

    monkeypatch.setattr(indexer, "embed_agent", embed_agent)
    return SimpleNamespace(agent=agent, seen=seen)


def test_an_agent_saved_from_a_route_thread_is_re_embedded_on_the_loop(embeds):
    from api.agents import _reindex_agent_embedding

    loop_thread, _ = _from_a_route_thread(lambda: _reindex_agent_embedding(embeds.agent, None),
                                          lambda: bool(embeds.seen))
    assert embeds.seen == [(7, loop_thread)]


def test_an_agent_saved_on_the_loop_is_re_embedded_there(embeds):
    from api.agents import _reindex_agent_embedding

    async def on_the_loop():
        _reindex_agent_embedding(embeds.agent, None)
        await _until(lambda: bool(embeds.seen))
        return threading.get_ident()

    assert embeds.seen == [(7, anyio.run(on_the_loop))]


def test_an_agent_saved_with_no_loop_is_not_re_embedded_and_nothing_raises(embeds):
    from api.agents import _reindex_agent_embedding

    _reindex_agent_embedding(embeds.agent, None)
    assert embeds.seen == []


# ---------------------------------------------------------------------------
# api/shopify._schedule_catalog_sync (the /events webhook)
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog(monkeypatch):
    from api import shopify

    shopify._catalog_debounce_handles.clear()
    shopify._catalog_pending_events.clear()
    shopify._catalog_sync_tasks.clear()
    monkeypatch.setattr(shopify.config, "SHOPIFY_SYNC_DEBOUNCE_SECONDS", 0.01)
    synced = []

    async def sync(workspace_id, reason):
        synced.append((workspace_id, reason, threading.get_ident()))

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", sync)
    yield SimpleNamespace(module=shopify, synced=synced)
    shopify._catalog_debounce_handles.clear()
    shopify._catalog_pending_events.clear()
    shopify._catalog_sync_tasks.clear()


def test_a_catalog_webhook_from_a_route_thread_re_syncs_on_the_loop(catalog):
    loop_thread, _ = _from_a_route_thread(
        lambda: catalog.module._schedule_catalog_sync(WS, "products/update"), lambda: bool(catalog.synced))
    assert catalog.synced == [(WS, "products/update", loop_thread)]


def test_a_catalog_webhook_on_the_loop_re_syncs_there(catalog):
    async def on_the_loop():
        catalog.module._schedule_catalog_sync(WS, "products/update")
        await _until(lambda: bool(catalog.synced))
        return threading.get_ident()

    assert catalog.synced == [(WS, "products/update", anyio.run(on_the_loop))]


def test_a_catalog_webhook_with_no_loop_raises_as_before(catalog):
    with pytest.raises(RuntimeError):
        catalog.module._schedule_catalog_sync(WS, "products/update")
    assert catalog.synced == []


# ---------------------------------------------------------------------------
# modules/knowledge/graph_service.schedule_incremental_update
# ---------------------------------------------------------------------------


@pytest.fixture
def graph(monkeypatch):
    from modules.knowledge import graph_service

    service = graph_service.GraphifyService()
    monkeypatch.setattr(graph_service, "_DEBOUNCE_SECONDS", 0)
    rebuilt = []

    async def rebuild(workspace_id):
        rebuilt.append((workspace_id, threading.get_ident()))

    monkeypatch.setattr(service, "_debounced_rebuild", rebuild)
    return SimpleNamespace(service=service, rebuilt=rebuilt)


def test_a_graph_rebuild_asked_from_a_route_thread_runs_on_the_loop(graph):
    pending = [{"type": "roster", "path": "Analyst", "id": 3}]
    loop_thread, _ = _from_a_route_thread(
        lambda: graph.service.schedule_incremental_update(WS, pending), lambda: bool(graph.rebuilt))
    assert graph.rebuilt == [(WS, loop_thread)]
    assert graph.service._pending_sources[WS] == pending


def test_a_graph_rebuild_asked_on_the_loop_runs_there(graph):
    async def on_the_loop():
        graph.service.schedule_incremental_update(WS, [{"type": "document", "id": 1}])
        await _until(lambda: bool(graph.rebuilt))
        return threading.get_ident()

    assert graph.rebuilt == [(WS, anyio.run(on_the_loop))]


def test_a_graph_rebuild_asked_with_no_loop_raises_as_before(graph):
    with pytest.raises(RuntimeError):
        graph.service.schedule_incremental_update(WS, [{"type": "document", "id": 1}])
    assert graph.rebuilt == []


# ---------------------------------------------------------------------------
# services/credit_watch (refuse_new_work's balance check, local edition)
# ---------------------------------------------------------------------------


@pytest.fixture
def credit(monkeypatch):
    from config import config
    from services import credit_watch

    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", True)
    monkeypatch.setattr(credit_watch, "_last_check", {})
    checked = []

    async def check(workspace_id):
        checked.append((workspace_id, threading.get_ident()))

    monkeypatch.setattr(credit_watch, "check_credit", check)
    guard = credit_watch.watches_the_credit(lambda db, ws, what: None)
    return SimpleNamespace(module=credit_watch, checked=checked, guard=guard)


def test_work_started_from_a_route_thread_has_the_credit_looked_at_on_the_loop(credit):
    loop_thread, answer = _from_a_route_thread(
        lambda: credit.guard(None, WS, "suggestion 1"), lambda: bool(credit.checked))
    assert answer is None                       # the guard's answer is unchanged
    assert credit.checked == [(WS, loop_thread)]
    assert WS in credit.module._last_check


def test_work_started_twice_from_route_threads_looks_once(credit):
    async def two_routes():
        await anyio.to_thread.run_sync(credit.guard, None, WS, "suggestion 1")
        await anyio.to_thread.run_sync(credit.guard, None, WS, "suggestion 2")
        await _until(lambda: bool(credit.checked))
        await anyio.sleep(0.05)

    anyio.run(two_routes)
    assert len(credit.checked) == 1


def test_work_started_with_no_loop_looks_at_nothing_and_leaves_the_check_due(credit):
    assert credit.guard(None, WS, "a script") is None
    assert credit.checked == [] and credit.module._last_check == {}


# ---------------------------------------------------------------------------
# api/recipe_executor.request_execution_cancel (Cancel on a playbook's card)
# ---------------------------------------------------------------------------


class _RunningRun:
    """A playbook run's task on this worker: where its cancel was made."""

    def __init__(self):
        self.cancelled_in = []

    def done(self):
        return bool(self.cancelled_in)

    def cancel(self):
        self.cancelled_in.append(threading.get_ident())
        return True


@pytest.fixture
def running_run(monkeypatch):
    import api.recipe_executor as rex

    run = _RunningRun()
    monkeypatch.setattr(rex, "_running_executions", {"exec-f105-h": run})
    return SimpleNamespace(module=rex, run=run)


def test_a_cancel_from_a_route_thread_is_made_on_the_loops_thread(running_run):
    """``Task.cancel`` is not thread-safe: the Cancel route's thread hands it to the loop."""
    loop_thread, stopped = _from_a_route_thread(
        lambda: running_run.module.request_execution_cancel("exec-f105-h"), lambda: True)
    assert stopped is True and running_run.run.cancelled_in == [loop_thread]


def test_a_cancel_on_the_loop_and_with_no_loop_is_made_where_it_is_asked(running_run):
    async def on_the_loop():
        return running_run.module.request_execution_cancel("exec-f105-h"), threading.get_ident()

    stopped, loop_thread = anyio.run(on_the_loop)
    assert stopped is True and running_run.run.cancelled_in == [loop_thread]
    assert running_run.module.request_execution_cancel("exec-f105-h") is False      # already done
    assert running_run.module.request_execution_cancel("not-on-this-worker") is False


def test_a_real_run_cancelled_from_a_route_thread_stops_at_once(monkeypatch):
    import api.recipe_executor as rex

    monkeypatch.setattr(rex, "_running_executions", {})

    async def route():
        run = asyncio.get_running_loop().create_task(asyncio.sleep(SETTLE_SECONDS * 10))
        rex._running_executions["exec-real"] = run
        await asyncio.sleep(0)
        stopped = await anyio.to_thread.run_sync(rex.request_execution_cancel, "exec-real")
        with anyio.fail_after(SETTLE_SECONDS):
            with pytest.raises(asyncio.CancelledError):
                await run
        return stopped

    assert anyio.run(route) is True
