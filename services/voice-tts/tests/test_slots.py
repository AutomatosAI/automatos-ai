"""The synthesis slots: a slot is held until its request has ended AND its synthesis thread has finished."""

from __future__ import annotations

import asyncio
import threading

from voice_tts.slots import Slots


async def _settle(condition, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "the condition never held"
        await asyncio.sleep(0.01)


def test_past_the_limit_admission_is_refused_at_once():
    async def go():
        slots = Slots(2)
        first, second = slots.try_acquire(), slots.try_acquire()
        assert first and second and slots.held == 2
        assert slots.try_acquire() is None
        first.close()
        assert slots.held == 1 and slots.try_acquire() is not None
        slots.shutdown()

    asyncio.run(go())


def test_a_cancelled_request_keeps_its_slot_until_its_synthesis_thread_finishes():
    async def go():
        slots = Slots(1)
        release, started = threading.Event(), threading.Event()

        def synthesis() -> bytes:
            started.set()
            assert release.wait(timeout=30)
            return b"pcm"

        slot = slots.try_acquire()
        waiting = asyncio.ensure_future(slot.run(synthesis))
        await _settle(started.is_set)
        waiting.cancel()  # the client went away; the handler's await is cancelled
        await asyncio.gather(waiting, return_exceptions=True)
        slot.close()  # the handler's finally
        assert slots.held == 1, "the synthesis thread still runs, so its slot is held"
        assert slots.try_acquire() is None, "a request arriving meanwhile is refused"
        release.set()
        await _settle(lambda: slots.held == 0)
        assert slot.released and slots.try_acquire() is not None
        slots.shutdown()

    asyncio.run(go())


def test_a_slot_whose_synthesis_finished_goes_back_when_the_request_ends():
    async def go():
        slots = Slots(1)
        slot = slots.try_acquire()
        assert await slot.run(lambda a, b: a + b, 2, 3) == 5
        await asyncio.sleep(0)
        assert slots.held == 1, "still held while the request goes on"
        slot.close()
        assert slots.held == 0
        slot.close()
        assert slots.held == 0, "closing twice gives it back once"
        slots.shutdown()

    asyncio.run(go())


def test_a_synthesis_that_raises_still_gives_the_slot_back():
    async def go():
        slots = Slots(1)
        slot = slots.try_acquire()

        def broken():
            raise RuntimeError("kokoro failed")

        try:
            await slot.run(broken)
        except RuntimeError:
            pass
        slot.close()
        await _settle(lambda: slots.held == 0)
        slots.shutdown()

    asyncio.run(go())
