"""The render queue's admission rule (owner, 2026-09-23): at most two renders at
once overall and one per workspace; everything else waits, first come first served.
"""

from __future__ import annotations

import asyncio

import pytest

from media_render.lanes import Lane


def run(coro):
    return asyncio.run(coro)


def test_two_run_overall_and_the_third_waits():
    async def scenario():
        lane = Lane(max_running=2, max_per_workspace=1)
        a, b, c = lane.submit("a", "ws-a"), lane.submit("b", "ws-b"), lane.submit("c", "ws-c")
        assert (a.done(), b.done(), c.done()) == (True, True, False)
        assert lane.waiting() == ("c",) and lane.position("c") == 1 and lane.running == 2
        lane.release("ws-a")
        assert c.done() and lane.running == 2 and lane.waiting() == ()

    run(scenario())


def test_one_workspace_never_runs_two_at_once_but_never_blocks_the_others():
    async def scenario():
        lane = Lane(max_running=2, max_per_workspace=1)
        a1, a2, b1 = lane.submit("a1", "ws-a"), lane.submit("a2", "ws-a"), lane.submit("b1", "ws-b")
        # a2 waits for a1; b1, behind it in line, is not held up by it.
        assert (a1.done(), a2.done(), b1.done()) == (True, False, True)
        lane.release("ws-b")
        assert not a2.done(), "a finished ws-b job must not start a second ws-a job"
        lane.release("ws-a")
        assert a2.done()

    run(scenario())


def test_waiting_jobs_start_in_arrival_order():
    async def scenario():
        lane = Lane(max_running=1, max_per_workspace=1)
        started = []
        futures = {key: lane.submit(key, f"ws-{key}") for key in ("a", "b", "c", "d")}
        for key, future in futures.items():
            future.add_done_callback(lambda _, key=key: started.append(key))
        for _ in range(4):
            await asyncio.sleep(0)
            lane.release(f"ws-{started[-1]}")
            await asyncio.sleep(0)
        assert started == ["a", "b", "c", "d"]

    run(scenario())


def test_a_withdrawn_job_never_takes_a_slot():
    async def scenario():
        lane = Lane(max_running=1, max_per_workspace=1)
        lane.submit("a", "ws-a")
        b, c = lane.submit("b", "ws-b"), lane.submit("c", "ws-c")
        b.cancel()
        lane.withdraw("b")
        lane.release("ws-a")
        assert c.done() and b.cancelled()
        assert lane.waiting() == () and lane.running == 1

    run(scenario())


def test_a_lane_needs_room():
    with pytest.raises(ValueError):
        Lane(max_running=0, max_per_workspace=1)
