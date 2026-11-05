"""Knowledge → Memory says why memory is not working, not only that it is not.

Memory went on by default in the local edition: the compose stack runs Qdrant and
envs/api.defaults sets QDRANT_URL=http://qdrant:6333. A fresh install still has no
embeddings key, and without one the durable store's embedder falls back to the degraded
stand-in that refuses to embed, so nothing can be remembered. ``GET /api/v1/memory/health``
answered "degraded" with no memories and no reason; now it answers "unavailable" with the
reason to show: add the embeddings key in Settings → API Keys. An install with memory off
(QDRANT_URL empty) and a store that does not answer each get their own reason.

The route runs for real with its memory service, its listing and its database faked; the
durable store's check runs against a real DurableMemoryStore whose embedder is faked.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from api import memory_stats  # noqa: E402
from core.llm.clients.base import DeterministicEmbeddingProvider  # noqa: E402
from modules.memory import health  # noqa: E402
from modules.memory.durable_store import DurableMemoryStore  # noqa: E402

_CTX = SimpleNamespace(workspace_id=uuid.uuid4())
_QDRANT = "http://qdrant:6333"
_MEMORY = {"memory": "Wholesale minimum is 3 kg.", "created_at": "2026-11-01T10:00:00+00:00"}


class _Embedder:
    """The durable store's embedder: ``provider`` is what ``_ensure_provider`` loads."""

    def __init__(self, provider):
        self.provider = None
        self._loaded = provider
        self.loads = 0

    def _ensure_provider(self):
        self.loads += 1
        self.provider = self._loaded


def _service(ready: bool) -> MagicMock:
    service = MagicMock()
    service._durable.embeddings_ready = MagicMock(return_value=ready)
    return service


def _db() -> MagicMock:
    db = MagicMock()
    db.execute.side_effect = OperationalError("SELECT", {}, Exception("no table"))
    return db


def _health(monkeypatch, service, items=None, listing_fails=False, qdrant_url=_QDRANT) -> dict:
    monkeypatch.setattr(health.config, "QDRANT_URL", qdrant_url, raising=False)
    listing = AsyncMock(side_effect=RuntimeError("connection refused")) if listing_fails \
        else AsyncMock(return_value=[("global", item) for item in items or []])
    with patch.object(memory_stats, "_get_memory_service", return_value=service), \
         patch.object(memory_stats, "_get_agent_ids", return_value=[]), \
         patch.object(memory_stats, "_fetch_all_scoped_memories", listing):
        return asyncio.run(memory_stats.get_memory_health(ctx=_CTX, db=_db()))


# --------------------------------------------------------------------------- the route


def test_no_embeddings_key_is_unavailable_and_says_to_add_one(monkeypatch):
    """The fresh local install: Qdrant answers, nothing is stored, and no key embeds."""
    body = _health(monkeypatch, _service(ready=False))
    assert body["health_status"] == "unavailable"
    assert body["reason"] == health.NO_EMBEDDINGS
    assert "Settings → API Keys" in body["reason"] and "embeddings key" in body["reason"]
    assert body["search_effectiveness"] == {"total_searches": 0, "hits": 0, "hit_rate": 0}


def test_memory_off_says_the_store_is_not_configured(monkeypatch):
    body = _health(monkeypatch, None, qdrant_url="")
    assert (body["health_status"], body["reason"]) == ("unavailable", health.NO_STORE)


def test_a_store_that_does_not_answer_says_so(monkeypatch):
    body = _health(monkeypatch, _service(ready=True), listing_fails=True)
    assert (body["health_status"], body["reason"]) == ("unavailable", health.STORE_DOWN)
    assert body["mem0_available"] is False


def test_a_working_store_has_no_reason(monkeypatch):
    assert _health(monkeypatch, _service(ready=True))["health_status"] == "degraded"
    body = _health(monkeypatch, _service(ready=True), items=[_MEMORY])
    assert (body["health_status"], body["reason"], body["total_memories"]) == ("healthy", None, 1)


# --------------------------------------------------------------------------- the store's own embedder


def _store(provider) -> DurableMemoryStore:
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._embedder = _Embedder(provider)
    return store


def test_the_store_is_not_ready_on_the_degraded_stand_in():
    store = _store(DeterministicEmbeddingProvider(dimension=8))
    assert store.embeddings_ready() is False
    assert store._embedder.loads == 1, "it loads the provider the store's adds and searches use"


def test_the_store_is_ready_on_a_real_provider():
    assert _store(SimpleNamespace(is_degraded=False)).embeddings_ready() is True
    assert _store(SimpleNamespace()).embeddings_ready() is True


@pytest.mark.parametrize("reason", [health.NO_STORE, health.STORE_DOWN, health.NO_EMBEDDINGS])
def test_every_reason_is_plain_words_without_internals(reason):
    assert "Traceback" not in reason and "Exception" not in reason and reason.endswith(".")
