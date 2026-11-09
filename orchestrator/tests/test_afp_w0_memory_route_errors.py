"""PRD-274 Wave 0 (Automatos for products): /api/memory says when memory was not kept.

Before: a refused store (``{"success": False}``) answered 201 with an invented id,
and a raised store, or an install with no durable memory, "stored" the memory in a
per-process dict (lost on restart, invisible to other workers), also with a 201.
Now a refused or failed call is a 503 whose detail carries a code and a plain
message, never the store's own error text, and nothing is kept in process memory.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from api import widget_memory

WS = "00000000-0000-0000-0000-0000000000c1"
CTX = NS(workspace_id=WS)
SECRET = "qdrant at 10.0.0.7:6333 refused the write"


def _with_service(monkeypatch, service):
    monkeypatch.setattr(widget_memory, "_get_memory_service", lambda: service)


def _store(content="Acme renews its enterprise plan every March", **kwargs):
    body = widget_memory.MemoryCreate(content=content, **kwargs)
    return asyncio.run(widget_memory.store_memory(body=body, ctx=CTX))


def _refusal(call) -> dict:
    with pytest.raises(HTTPException) as caught:
        call()
    assert caught.value.status_code == 503
    return caught.value.detail


# ── store ───────────────────────────────────────────────────────────────────────


def test_a_stored_memory_comes_back_with_the_id_the_store_kept(monkeypatch):
    kept = str(uuid.uuid4())

    async def store_long_term(**_):
        return {"success": True, "id": kept}

    _with_service(monkeypatch, NS(store_long_term=store_long_term))
    assert _store().id == kept


def test_a_refused_store_is_a_503_not_a_201_with_an_invented_id(monkeypatch):
    async def store_long_term(**_):
        return {"success": False, "error": SECRET}

    _with_service(monkeypatch, NS(store_long_term=store_long_term))
    detail = _refusal(_store)
    assert detail["code"] == widget_memory.MEMORY_STORE_FAILED
    assert SECRET not in str(detail)                     # the store's reason stays in the log


def test_a_store_that_kept_nothing_without_an_id_is_refused(monkeypatch):
    async def store_long_term(**_):
        return {"success": True}

    _with_service(monkeypatch, NS(store_long_term=store_long_term))
    assert _refusal(_store)["code"] == widget_memory.MEMORY_STORE_FAILED


def test_disabled_durable_memory_says_memory_off(monkeypatch):
    async def store_long_term(**_):
        return {"success": False, "skipped": True, "error": "durable memory not configured (QDRANT_URL empty)"}

    _with_service(monkeypatch, NS(store_long_term=store_long_term))
    assert _refusal(_store)["code"] == widget_memory.MEMORY_OFF


def test_a_store_that_raises_is_a_503_and_is_logged(monkeypatch, caplog):
    async def store_long_term(**_):
        raise RuntimeError(SECRET)

    _with_service(monkeypatch, NS(store_long_term=store_long_term))
    detail = _refusal(_store)
    assert detail["code"] == widget_memory.MEMORY_STORE_FAILED and SECRET not in str(detail)
    assert any(r.exc_info and SECRET in str(r.exc_info[1]) for r in caplog.records)


# ── no durable memory on this install ─────────────────────────────────────────────


def test_every_route_says_memory_off_when_the_install_has_no_durable_memory(monkeypatch):
    _with_service(monkeypatch, None)
    calls = [
        _store,
        lambda: asyncio.run(widget_memory.list_memories(ctx=CTX, limit=10)),
        lambda: asyncio.run(widget_memory.search_memories(q="Acme", ctx=CTX, limit=5)),
        lambda: asyncio.run(widget_memory.delete_memory(memory_id="m1", ctx=CTX)),
    ]
    for call in calls:
        assert _refusal(call)["code"] == widget_memory.MEMORY_OFF


def test_nothing_is_kept_in_process_memory():
    assert not [name for name in vars(widget_memory) if "fallback" in name.lower()]


# ── reads that fail ──────────────────────────────────────────────────────────────


def test_a_failed_read_is_a_503_not_an_empty_answer(monkeypatch):
    async def boom(**_):
        raise RuntimeError(SECRET)

    _with_service(monkeypatch, NS(get_all_memories=boom, search_long_term=boom, delete_memory=boom))
    calls = [
        lambda: asyncio.run(widget_memory.list_memories(ctx=CTX, limit=10)),
        lambda: asyncio.run(widget_memory.search_memories(q="Acme", ctx=CTX, limit=5)),
        lambda: asyncio.run(widget_memory.delete_memory(memory_id="m1", ctx=CTX)),
    ]
    for call in calls:
        detail = _refusal(call)
        assert detail["code"] == widget_memory.MEMORY_UNAVAILABLE and SECRET not in str(detail)


def test_search_keeps_the_score_and_list_does_not(monkeypatch):
    row = {"id": "m1", "memory": "Acme renews its enterprise plan every March", "metadata": {"tags": ["client"]}, "score": 0.8}

    async def rows(**_):
        return [row]

    _with_service(monkeypatch, NS(get_all_memories=rows, search_long_term=rows))
    found = asyncio.run(widget_memory.search_memories(q="renewal", ctx=CTX, limit=5)).results[0]
    listed = asyncio.run(widget_memory.list_memories(ctx=CTX, limit=10)).memories[0]
    assert (found.score, found.tags, found.content) == (0.8, ["client"], row["memory"])
    assert (listed.score, listed.tags) == (None, ["client"])
