"""PRD-273 US-273-02 — durable and field memory are read page by page, with no cap.

The GDPR export read at most 10,000 memories of each store; a move needs them all, a page at a
time. ``DurableMemoryStore.scroll_workspace`` and ``VectorFieldSharedContext.scroll_workspace``
follow Qdrant's scroll offsets to the end, without vectors, and both stores' ``export_workspace``
(the GDPR leg) now read through them and keep their cap. Qdrant is faked: a store is built
without its constructor (which dials Qdrant) and given a client that serves pages.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from modules.context.adapters.vector_field import VectorFieldSharedContext
from modules.memory.durable_store import DurableMemoryStore

POINTS = 600


class FakeQdrant:
    """Serves ``count`` points of the asked workspace, ``limit`` at a time, by offset."""

    def __init__(self, count: int) -> None:
        self.points = [SimpleNamespace(id=f"p{n}", payload={"content": f"memory {n}", "key": f"k{n}", "namespace": "ws"})
                       for n in range(count)]
        self.calls = []

    async def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.calls.append({"limit": limit, "offset": offset, "with_vectors": with_vectors})
        start = offset or 0
        page = self.points[start:start + limit]
        following = start + limit
        return page, (following if following < len(self.points) else None)


def _durable(client) -> DurableMemoryStore:
    store = object.__new__(DurableMemoryStore)
    store._enabled, store._client, store._collection, store._bootstrap_done = True, client, "durable_memory", True
    return store


def _field(client) -> VectorFieldSharedContext:
    store = object.__new__(VectorFieldSharedContext)
    store._client, store._bootstrap_done = client, True
    return store


async def _all(pages):
    return [item async for page in pages for item in page]


def test_durable_memory_scrolls_every_page_without_a_cap():
    client = FakeQdrant(POINTS)
    items = asyncio.run(_all(_durable(client).scroll_workspace("ws-1")))
    assert [point_id for point_id, _ in items] == [f"p{n}" for n in range(POINTS)]
    assert items[0][1]["content"] == "memory 0"
    assert all(call["with_vectors"] is False for call in client.calls) and len(client.calls) == 3


def test_field_memory_scrolls_every_page_without_a_cap():
    client = FakeQdrant(POINTS)
    items = asyncio.run(_all(_field(client).scroll_workspace("ws-1")))
    assert [item["key"] for item in items] == [f"k{n}" for n in range(POINTS)]


def test_the_gdpr_exports_keep_their_cap():
    durable = asyncio.run(_durable(FakeQdrant(POINTS)).export_workspace("ws-1", limit=300))
    assert len(durable) == 300 and durable[0]["memory"] == "memory 0"
    field = asyncio.run(_field(FakeQdrant(POINTS)).export_workspace("ws-1", limit=300))
    assert len(field) == 300 and set(field[0]) == {"key", "value", "strength", "agent_id", "mission_id", "task_id",
                                                   "created_at"}


def test_durable_memory_off_scrolls_nothing():
    store = _durable(FakeQdrant(POINTS))
    store._enabled = False
    assert asyncio.run(_all(store.scroll_workspace("ws-1"))) == []
