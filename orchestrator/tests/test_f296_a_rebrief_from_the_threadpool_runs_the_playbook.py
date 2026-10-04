"""F296 (night 8) — a re-brief on a playbook's card runs the playbook, from the route's threadpool.

#0249 at 01:19Z: "Update ticket and re-queue" on a playbook's card answered HTTP
500 "Internal server error", yet the new brief was on the card. Then the card
failed "Stalled: no progress for 120s (status was 'pending')". The re-brief route
is a plain ``def``, so FastAPI runs it in its threadpool, where no event loop
runs, and ``launch_recipe_task``'s ``asyncio.create_task`` raised "no running
event loop" after the brief was committed. The rerun row stayed pending for good.

These tests start runs the way the routes do. The board's own re-brief test
(F243) replaces the launch, so it never reached ``create_task``.
"""
from __future__ import annotations

import asyncio
import functools
import threading
import uuid

import anyio

from tests import test_f243_a_redo_runs_on_the_same_card as f243

engine = f243.engine
workspace = f243.workspace

NEW_BRIEF = "Monday stock: list every line under five, with the supplier beside it."
SETTLE_SECONDS = 2.0


def _launch_kwargs():
    return {"recipe_execution_id": "exec-f296", "recipe_id": 7, "workspace_id": uuid.uuid4(), "input_data": {}}


def test_a_run_started_from_a_route_thread_is_created_on_the_event_loop(monkeypatch):
    """FastAPI's threadpool (anyio's worker thread) has no loop; the task is
    created on the loop's own thread, where ``create_task`` works."""
    import api.recipe_executor as rex
    from services.playbook_engine import get_playbook_engine

    seen = {}

    def _launch(**kwargs):
        asyncio.get_running_loop()  # what create_task needs; raises in a worker thread
        seen.update(kwargs, thread=threading.get_ident())

    monkeypatch.setattr(rex, "launch_recipe_task", _launch)
    kwargs = _launch_kwargs()

    async def _route():
        loop_thread = threading.get_ident()
        await anyio.to_thread.run_sync(functools.partial(get_playbook_engine().launch, **kwargs))
        return loop_thread

    loop_thread = anyio.run(_route)
    assert seen["thread"] == loop_thread
    assert {k: seen[k] for k in kwargs} == kwargs


def test_a_run_started_on_the_event_loop_starts_right_there(monkeypatch):
    import api.recipe_executor as rex
    from services.playbook_engine import get_playbook_engine

    threads = []
    monkeypatch.setattr(rex, "launch_recipe_task", lambda **kwargs: threads.append(threading.get_ident()))

    async def _handler():
        get_playbook_engine().launch(**_launch_kwargs())
        return threading.get_ident()

    assert threads == [anyio.run(_handler)]


def test_a_run_started_outside_any_loop_starts_where_it_is_asked(monkeypatch):
    """A script or a plain test: no loop and no route thread, so the launch runs
    in the caller's thread, as it did before F296."""
    import api.recipe_executor as rex
    from services.playbook_engine import get_playbook_engine

    threads = []
    monkeypatch.setattr(rex, "launch_recipe_task", lambda **kwargs: threads.append(threading.get_ident()))
    get_playbook_engine().launch(**_launch_kwargs())
    assert threads == [threading.get_ident()]


def test_a_rebrief_on_a_playbook_card_from_the_threadpool_runs_the_playbook(workspace, new_session, monkeypatch):
    """The whole path #0249 took: the route's threadpool, the redo on the same card,
    the real ``launch_recipe_task``. Only the playbook's own steps are stood in for."""
    import api.recipe_executor as rex
    from api.board_task_rebrief import RebriefBody, rebrief_task

    started = []

    async def _steps(**kwargs):
        started.append(kwargs["recipe_execution_id"])

    monkeypatch.setattr(rex, "execute_recipe_direct", _steps)
    pb = f243._finished_playbook(new_session, workspace)

    async def _route():
        answer = await anyio.to_thread.run_sync(functools.partial(
            rebrief_task, pb.card, RebriefBody(brief=NEW_BRIEF), ctx=f243._owner(workspace), db=new_session()))
        with anyio.fail_after(SETTLE_SECONDS):
            while not started:
                await asyncio.sleep(0.01)
        return answer

    answer = anyio.run(_route)
    card = f243._card(new_session, pb.card)
    assert answer["success"] is True
    assert card.source_id != pb.run, "the redo runs on the same card, under its new run"
    assert started == [card.source_id]
