"""A chat turn embeds each text once (voice latency D-3).

In one turn the same text reached ``EmbeddingManager.generate_embedding`` from memory L3 and
L2 (one gather), then the knowledge search and the owner's-documents leg: a provider call
each. With ``TURN_EMBED_ONCE`` on the turn runs inside ``core.llm.embed_once.turn_scope()``
and a repeat is answered from the turn's memo.

What is pinned here:

* outside a scope every call reaches the provider (every caller that is not a chat turn);
* inside a scope one text is one provider call, each caller gets a list of its own, and
  another text, model or dimension is another call;
* a caller sharing a call that failed makes one call of its own; the creator gets the error;
* a call in flight lives as long as someone waits for it: the last waiter's cancel cancels it;
* nothing is reused or kept once the scope has closed, and at most ``MAX_KEPT`` are kept;
* a kept vector is read from another loop in another thread; a call in flight is not;
* the switch is off by default, and off means no scope is opened.

Pure tests: the provider is a fake. No network, no database.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import pathlib
from types import SimpleNamespace as NS
from typing import List, Optional, Set

import pytest

import consumers.chatbot.turn_embeddings as te
from config import config
from core.llm import embed_once
from core.llm.embedding_manager import EmbeddingManager

TEXT = "when does the shop open on Sunday"
OTHER_TEXT = "when does the shop open on Sunday "  # one trailing space: another text
MODEL, DIMENSION = "embed-a", 4
EMBED_LOGGER = "core.llm.embed_once"
SURFACE = pathlib.Path(__file__).resolve().parents[1] / "reports" / "config-surface.json"


class _Provider:
    """Stands in for an embedding provider. A vector starts with the number of the call
    that made it, so two vectors are equal only when one call made both."""

    def __init__(self, model: str = MODEL, dimension: int = DIMENSION) -> None:
        self.config = NS(model=model, dimension=dimension)
        self.calls: List[str] = []
        self.cancelled = 0
        self.hold: Optional[asyncio.Event] = None  # set: a call waits here until released
        self.held: Optional[Set[int]] = None       # which calls wait on ``hold``; None = all
        self.failing: Set[int] = set()             # which calls raise

    async def generate_embedding(self, text: str) -> List[float]:
        self.calls.append(text)
        number = len(self.calls)
        try:
            if self.hold is not None and (self.held is None or number in self.held):
                await self.hold.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if number in self.failing:
            raise RuntimeError("the provider is down")
        return [float(number), float(len(text)), 0.0, 0.0]


def _manager(provider: Optional[_Provider] = None):
    """A real manager over a fake provider, with nothing loaded from settings."""
    provider = provider or _Provider()
    manager = EmbeddingManager.__new__(EmbeddingManager)
    manager.provider = provider
    manager._provider_loaded = True
    return provider, manager


async def _settle() -> None:
    """Let every task that is ready run, a few times over."""
    for _ in range(5):
        await asyncio.sleep(0)


# ── outside a scope: as before ─────────────────────────────────────────────


def test_the_switch_is_off_by_default_and_named_in_the_config_surface():
    # The class holds what the environment gave at import; tests patch the instance.
    assert type(config).TURN_EMBED_ONCE is False
    assert "TURN_EMBED_ONCE" in json.loads(SURFACE.read_text())["settings"]


def test_outside_a_scope_two_calls_are_two_provider_calls():
    provider, manager = _manager()

    async def go():
        together = await asyncio.gather(manager.generate_embedding(TEXT), manager.generate_embedding(TEXT))
        return together, await manager.generate_embedding(TEXT), embed_once.current()

    (first, second), third, memo = asyncio.run(go())

    assert provider.calls == [TEXT, TEXT, TEXT]
    assert memo is None
    assert {first[0], second[0], third[0]} == {1.0, 2.0, 3.0}


def test_the_model_key_names_the_provider_its_model_and_its_dimension():
    _, manager = _manager(_Provider(model="embed-b", dimension=8))

    assert manager._model_key() == "_Provider:embed-b:8"


def test_a_provider_with_no_config_still_has_a_key():
    _, manager = _manager()
    del manager.provider.config

    assert manager._model_key() == "_Provider:None:None"


# ── inside a scope: one call per text ──────────────────────────────────────


def test_inside_a_scope_two_calls_at_once_are_one_provider_call():
    provider, manager = _manager()

    async def go():
        with embed_once.turn_scope():
            first, second = await asyncio.gather(
                manager.generate_embedding(TEXT), manager.generate_embedding(TEXT))
            calls_after_two = len(provider.calls)
            third = await manager.generate_embedding(TEXT)
            return first, second, third, calls_after_two

    first, second, third, calls_after_two = asyncio.run(go())

    assert calls_after_two == 1 and provider.calls == [TEXT]
    assert first == second == third == [1.0, float(len(TEXT)), 0.0, 0.0]
    assert first is not second and third is not first and third is not second


def test_a_callers_list_is_its_own_to_change():
    provider, manager = _manager()

    async def go():
        with embed_once.turn_scope():
            first = await manager.generate_embedding(TEXT)
            first.append(99.0)
            return await manager.generate_embedding(TEXT)

    assert asyncio.run(go()) == [1.0, float(len(TEXT)), 0.0, 0.0]
    assert provider.calls == [TEXT]


def test_another_text_model_or_dimension_is_another_call():
    provider, manager = _manager()
    other_model, on_other_model = _manager(_Provider(model="embed-b"))
    other_dimension, on_other_dimension = _manager(_Provider(dimension=8))

    async def go():
        with embed_once.turn_scope():
            await manager.generate_embedding(TEXT)
            await manager.generate_embedding(OTHER_TEXT)  # never trimmed: a different string
            await on_other_model.generate_embedding(TEXT)
            await on_other_dimension.generate_embedding(TEXT)
            # and each of them is kept under its own key
            await manager.generate_embedding(TEXT)
            await on_other_model.generate_embedding(TEXT)
            await on_other_dimension.generate_embedding(TEXT)

    asyncio.run(go())

    assert provider.calls == [TEXT, OTHER_TEXT]
    assert other_model.calls == [TEXT]
    assert other_dimension.calls == [TEXT]


def test_two_managers_on_one_model_share_the_turns_embedding():
    provider, manager = _manager()
    second_provider, second_manager = _manager()

    async def go():
        with embed_once.turn_scope():
            return await manager.generate_embedding(TEXT), await second_manager.generate_embedding(TEXT)

    first, second = asyncio.run(go())

    assert first == second
    assert provider.calls == [TEXT] and second_provider.calls == []


def test_the_synchronous_and_the_batch_calls_are_left_alone():
    provider, manager = _manager()

    async def go():
        with embed_once.turn_scope():
            return await manager.generate_embeddings_batch([TEXT, TEXT])

    first, second = asyncio.run(go())

    assert provider.calls == [TEXT, TEXT]
    assert first != second


def test_the_scope_logs_its_counts_and_never_a_text(caplog):
    provider, manager = _manager()
    caplog.set_level(logging.INFO, logger=EMBED_LOGGER)

    async def go():
        with embed_once.turn_scope():
            await asyncio.gather(manager.generate_embedding(TEXT), manager.generate_embedding(TEXT))
            await manager.generate_embedding(TEXT)

    asyncio.run(go())

    lines = [r.getMessage() for r in caplog.records if r.name == EMBED_LOGGER]
    assert lines == ["[embed-once] made=1 reused=2"]
    assert TEXT not in caplog.text


def test_a_scope_that_embedded_nothing_logs_nothing(caplog):
    caplog.set_level(logging.INFO, logger=EMBED_LOGGER)

    with embed_once.turn_scope():
        pass

    assert [r for r in caplog.records if r.name == EMBED_LOGGER] == []


# ── a shared call that fails ───────────────────────────────────────────────


def test_a_sharer_whose_shared_call_failed_makes_one_call_of_its_own():
    provider, manager = _manager()
    provider.failing = {1}

    async def go():
        with embed_once.turn_scope():
            creator = asyncio.ensure_future(manager.generate_embedding(TEXT))
            sharer = asyncio.ensure_future(manager.generate_embedding(TEXT))
            outcomes = await asyncio.gather(creator, sharer, return_exceptions=True)
            calls_then = list(provider.calls)
            return outcomes, calls_then, await manager.generate_embedding(TEXT)

    (creators, sharers), calls_then, next_call = asyncio.run(go())

    assert isinstance(creators, RuntimeError)       # the creator fails, as it would have alone
    assert sharers == [2.0, float(len(TEXT)), 0.0, 0.0]
    assert calls_then == [TEXT, TEXT]               # what two independent callers cost before
    assert next_call[0] == 3.0                      # a failure is not kept: the provider is asked again
    assert provider.calls == [TEXT, TEXT, TEXT]


def test_a_sharer_whose_own_call_fails_too_gets_its_own_error():
    provider, manager = _manager()
    provider.failing = {1, 2}

    async def go():
        with embed_once.turn_scope():
            creator = asyncio.ensure_future(manager.generate_embedding(TEXT))
            sharer = asyncio.ensure_future(manager.generate_embedding(TEXT))
            return await asyncio.gather(creator, sharer, return_exceptions=True)

    creators, sharers = asyncio.run(go())

    assert isinstance(creators, RuntimeError) and isinstance(sharers, RuntimeError)
    assert creators is not sharers
    assert provider.calls == [TEXT, TEXT]


# ── cancellation: a call lives as long as someone waits for it ─────────────


def test_one_of_two_waiters_cancelled_leaves_the_call_running():
    provider, manager = _manager()

    async def go():
        provider.hold = asyncio.Event()
        with embed_once.turn_scope():
            stays = asyncio.ensure_future(manager.generate_embedding(TEXT))
            leaves = asyncio.ensure_future(asyncio.wait_for(manager.generate_embedding(TEXT), 0.01))
            with pytest.raises(asyncio.TimeoutError):
                await leaves
            await _settle()
            cancelled_then = provider.cancelled
            provider.hold.set()
            vector = await stays
            return cancelled_then, vector, await manager.generate_embedding(TEXT)

    cancelled_then, vector, kept = asyncio.run(go())

    assert cancelled_then == 0 and provider.cancelled == 0
    assert vector == kept == [1.0, float(len(TEXT)), 0.0, 0.0]
    assert provider.calls == [TEXT]


def test_the_only_waiters_cancel_cancels_the_provider_call():
    provider, manager = _manager()

    async def go():
        provider.hold = asyncio.Event()
        with embed_once.turn_scope():
            memo = embed_once.current()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(manager.generate_embedding(TEXT), 0.01)
            await _settle()
            seen = (provider.cancelled, dict(memo.done), dict(memo.flying))
            provider.hold.set()
            return seen, await manager.generate_embedding(TEXT)

    (cancelled, kept, flying), again = asyncio.run(go())

    assert cancelled == 1                 # the fake saw CancelledError: nothing was left running
    assert kept == {} and flying == {}    # and nothing was stored
    assert again[0] == 2.0 and provider.calls == [TEXT, TEXT]


def test_the_last_of_two_waiters_to_be_cancelled_cancels_the_call(caplog):
    """Both waiters go, then the loop closes: the call was cancelled by the last one, so the
    loop has no pending task to destroy and no exception nobody read."""
    provider, manager = _manager()
    caplog.set_level(logging.DEBUG)

    async def go():
        provider.hold = asyncio.Event()
        with embed_once.turn_scope():
            waiters = [asyncio.ensure_future(manager.generate_embedding(TEXT)) for _ in range(2)]
            await _settle()
            for waiter in waiters:
                waiter.cancel()
            return await asyncio.gather(*waiters, return_exceptions=True)

    outcomes = asyncio.run(go())
    gc.collect()

    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    assert provider.calls == [TEXT] and provider.cancelled == 1
    assert "Task was destroyed but it is pending" not in caplog.text
    assert "never retrieved" not in caplog.text


def test_a_failed_call_whose_sharer_left_leaves_nothing_unread(caplog):
    provider, manager = _manager()
    provider.failing = {1}
    caplog.set_level(logging.DEBUG)

    async def go():
        provider.hold = asyncio.Event()
        with embed_once.turn_scope():
            creator = asyncio.ensure_future(manager.generate_embedding(TEXT))
            sharer = asyncio.ensure_future(manager.generate_embedding(TEXT))
            await _settle()
            sharer.cancel()
            await _settle()
            provider.hold.set()
            return await asyncio.gather(creator, sharer, return_exceptions=True)

    creators, sharers = asyncio.run(go())
    gc.collect()

    assert isinstance(creators, RuntimeError) and isinstance(sharers, asyncio.CancelledError)
    assert provider.calls == [TEXT]
    assert "never retrieved" not in caplog.text


# ── after the scope: nothing is reused, nothing is kept ────────────────────


def test_work_that_outlives_the_scope_embeds_as_if_there_were_none():
    provider, manager = _manager()

    async def go():
        release = asyncio.Event()

        async def outlives():
            await release.wait()
            seen = embed_once.current()  # the turn's memo, through the copied context
            return seen, await manager.generate_embedding(TEXT), await manager.generate_embedding(TEXT)

        with embed_once.turn_scope():
            memo = embed_once.current()
            first = await manager.generate_embedding(TEXT)
            later = asyncio.ensure_future(outlives())
            await _settle()
        after_exit = embed_once.current()
        release.set()
        seen, second, third = await later
        return memo, after_exit, seen, first, second, third

    memo, after_exit, seen, first, second, third = asyncio.run(go())

    assert after_exit is None
    assert seen is memo and memo.closed
    assert memo.done == {} and memo.flying == {}
    assert (first[0], second[0], third[0]) == (1.0, 2.0, 3.0)   # a provider call per call
    assert provider.calls == [TEXT, TEXT, TEXT]


def test_a_call_in_flight_when_the_scope_closes_still_answers_and_is_not_kept():
    provider, manager = _manager()

    async def go():
        provider.hold = asyncio.Event()
        with embed_once.turn_scope():
            memo = embed_once.current()
            waiter = asyncio.ensure_future(manager.generate_embedding(TEXT))
            await _settle()
            in_flight = list(provider.calls)
        provider.hold.set()
        vector = await waiter
        await _settle()
        return memo, in_flight, vector

    memo, in_flight, vector = asyncio.run(go())

    assert in_flight == [TEXT]
    assert vector == [1.0, float(len(TEXT)), 0.0, 0.0]
    assert provider.cancelled == 0
    assert memo.closed and memo.done == {} and memo.flying == {}


def test_a_turn_keeps_at_most_the_cap():
    provider, manager = _manager()
    texts = [f"text {number}" for number in range(embed_once.MAX_KEPT + 1)]

    async def go():
        with embed_once.turn_scope():
            for text in texts:
                await manager.generate_embedding(text)
            kept = len(embed_once.current().done)
            await manager.generate_embedding(texts[0])    # kept: no call
            calls_then = len(provider.calls)
            await manager.generate_embedding(texts[-1])   # one over the cap: embedded again
            return kept, calls_then

    kept, calls_then = asyncio.run(go())

    assert kept == embed_once.MAX_KEPT == 32
    assert calls_then == embed_once.MAX_KEPT + 1
    assert provider.calls == texts + [texts[-1]]


# ── other loops and threads ────────────────────────────────────────────────


def test_a_kept_vector_is_read_on_another_loop_in_another_thread():
    provider, manager = _manager()

    def on_its_own_loop():
        return asyncio.run(manager.generate_embedding(TEXT))

    async def go():
        with embed_once.turn_scope():
            first = await manager.generate_embedding(TEXT)
            # asyncio.to_thread runs it in a copy of this context, as the turn's threads are run.
            return first, await asyncio.to_thread(on_its_own_loop)

    first, elsewhere = asyncio.run(go())

    assert elsewhere == first and elsewhere is not first
    assert provider.calls == [TEXT]


def test_a_call_in_flight_on_one_loop_is_not_awaited_from_another():
    provider, manager = _manager()

    def on_its_own_loop():
        return asyncio.run(manager.generate_embedding(TEXT))

    async def go():
        provider.hold, provider.held = asyncio.Event(), {1}  # only this loop's call waits
        with embed_once.turn_scope():
            here = asyncio.ensure_future(manager.generate_embedding(TEXT))
            await _settle()
            elsewhere = await asyncio.to_thread(on_its_own_loop)
            calls_then = list(provider.calls)
            provider.hold.set()
            return await here, elsewhere, calls_then

    here, elsewhere, calls_then = asyncio.run(go())

    assert calls_then == [TEXT, TEXT]         # the other loop made its own call
    assert (here[0], elsewhere[0]) == (1.0, 2.0)


# ── through the memory store ───────────────────────────────────────────────


class _Qdrant:
    """query_points that finds nothing and records the vector it was asked with."""

    def __init__(self) -> None:
        self.vectors: list = []

    async def query_points(self, collection_name, query, query_filter, limit, **_kwargs):
        self.vectors.append(query)
        return NS(points=[])


def test_two_memory_reads_of_one_text_embed_it_once():
    from modules.memory.durable_store import DurableMemoryStore

    provider, manager = _manager()
    client = _Qdrant()
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client, store._embedder = client, manager
    mine = DurableMemoryStore._namespace_filter("mem:workspace")

    async def go():
        with embed_once.turn_scope():
            return await asyncio.gather(store._query(TEXT, mine, 5), store._query(TEXT, mine, 5))

    assert asyncio.run(go()) == [[], []]
    assert provider.calls == [TEXT]
    assert len(client.vectors) == 2 and client.vectors[0] == client.vectors[1]
    assert client.vectors[0] is not client.vectors[1]


# ── the turn's wrapper ─────────────────────────────────────────────────────


async def _turn_stub(chat, *args, **kwargs):
    """A turn that says which memo it ran under, embedding one text twice on the way."""
    await chat.generate_embedding(TEXT)
    await chat.generate_embedding(TEXT)
    yield embed_once.current()


def _a_turn(monkeypatch, switch: bool):
    provider, manager = _manager()
    monkeypatch.setattr(config, "TURN_EMBED_ONCE", switch)

    async def go():
        frames = [frame async for frame in te.embeds_each_text_once(_turn_stub)(manager)]
        return frames, embed_once.current()

    frames, afterwards = asyncio.run(go())
    return provider, frames, afterwards


def test_with_the_switch_off_the_turn_opens_no_scope(monkeypatch):
    provider, frames, afterwards = _a_turn(monkeypatch, False)

    assert frames == [None] and afterwards is None
    assert provider.calls == [TEXT, TEXT]


def test_with_the_switch_on_the_turn_runs_in_a_scope_that_closes_with_it(monkeypatch):
    provider, (memo,), afterwards = _a_turn(monkeypatch, True)

    assert memo is not None and memo.closed
    assert afterwards is None
    assert provider.calls == [TEXT]


def test_the_wrapper_passes_the_turns_arguments_and_frames_through(monkeypatch):
    async def turn(chat, chat_id, message, **kwargs):
        yield (chat, chat_id, message, kwargs)
        yield "second"

    async def frames_of(switch):
        monkeypatch.setattr(config, "TURN_EMBED_ONCE", switch)
        return [frame async for frame in te.embeds_each_text_once(turn)("chat", "c1", "hello", spoken_mode=True)]

    expected = [("chat", "c1", "hello", {"spoken_mode": True}), "second"]
    assert asyncio.run(frames_of(False)) == expected
    assert asyncio.run(frames_of(True)) == expected


@pytest.mark.parametrize("switch", [False, True])
def test_a_turn_that_raises_raises_through_the_wrapper_and_leaves_no_scope(monkeypatch, switch):
    monkeypatch.setattr(config, "TURN_EMBED_ONCE", switch)

    async def turn(chat):
        yield "first"
        raise RuntimeError("the turn failed")

    async def go():
        frames = []
        with pytest.raises(RuntimeError, match="the turn failed"):
            async for frame in te.embeds_each_text_once(turn)(object()):
                frames.append(frame)
        return frames, embed_once.current()

    assert asyncio.run(go()) == (["first"], None)


@pytest.mark.parametrize("switch", [False, True])
def test_closing_the_wrapper_closes_the_turn_inside_it(monkeypatch, switch):
    monkeypatch.setattr(config, "TURN_EMBED_ONCE", switch)
    closed = []

    async def turn(chat):
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    async def go():
        frames = te.embeds_each_text_once(turn)(object())
        first = await frames.__anext__()
        await frames.aclose()
        return first, list(closed)

    assert asyncio.run(go()) == ("first", [True])


def test_a_turn_closed_from_another_task_does_not_raise(monkeypatch):
    _, manager = _manager()
    monkeypatch.setattr(config, "TURN_EMBED_ONCE", True)

    async def go():
        frames = te.embeds_each_text_once(_turn_stub)(manager)

        async def first_frame():
            return await frames.__anext__()

        memo = await asyncio.ensure_future(first_frame())  # the scope opens in that task's context
        await frames.aclose()                              # and closes in this one
        return memo, embed_once.current()

    memo, here = asyncio.run(go())

    assert memo is not None and memo.closed
    assert here is None


def test_a_turn_closed_from_another_task_leaves_that_tasks_own_scope_alone(monkeypatch):
    _, manager = _manager()
    monkeypatch.setattr(config, "TURN_EMBED_ONCE", True)

    async def go():
        with embed_once.turn_scope():
            outer = embed_once.current()
            frames = te.embeds_each_text_once(_turn_stub)(manager)

            async def first_frame():
                return await frames.__anext__()

            inner = await asyncio.ensure_future(first_frame())
            await frames.aclose()
            return outer, inner, embed_once.current(), outer.closed

    outer, inner, here, outer_closed_inside = asyncio.run(go())

    assert inner is not outer and inner.closed
    assert here is outer and not outer_closed_inside


def test_the_chat_turn_is_wrapped_directly_beneath_its_timing():
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.turn_timing import keeps_the_turns_timing

    async def _stub(chat, *args, **kwargs):
        yield None

    codes, fn = [], StreamingChatService._stream_response_with_agent_scoped
    while fn is not None:
        codes.append(fn.__code__)
        fn = getattr(fn, "__wrapped__", None)

    embeds = te.embeds_each_text_once(_stub).__code__
    assert embeds in codes
    assert codes.index(embeds) == codes.index(keeps_the_turns_timing(_stub).__code__) + 1
