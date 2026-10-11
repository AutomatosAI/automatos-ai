"""POST /api/memory/consolidate removes the originals it merged.

The handler called ``service.delete_memory(mid)`` without the required
``workspace_id``: the merged memory was stored, then a TypeError gave a 500
and every original stayed — each retry added another merged copy. Agent and
daily memories live under their own namespace, so each original is deleted
under the namespace it was read from.
"""
from __future__ import annotations

import asyncio
import os
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import memory_stats  # noqa: E402
from modules.memory.unified_memory_service import UnifiedMemoryService  # noqa: E402

_WS = uuid.uuid4()
_WS_NS = f"mem:{_WS}"
_AGENT_NS = f"mem:{_WS}:agent:7"


def _fake_service() -> MagicMock:
    service = MagicMock()
    service.store_long_term = AsyncMock(return_value=True)
    service.delete_memories_scoped = AsyncMock(return_value=True)
    service.delete_memory = AsyncMock(return_value=True)
    return service


def _items():
    return [
        ("global", {"id": "a", "memory": "likes tea", "namespace": _WS_NS}),
        ("agent", {"id": "b", "memory": "prefers mornings", "namespace": _AGENT_NS}),
        ("agent", {"id": "c", "memory": "uses metric", "namespace": _AGENT_NS}),
        ("global", {"id": "d", "memory": "legacy item"}),
    ]


def _consolidate(service, memory_ids, strategy="merge"):
    body = memory_stats.ConsolidateRequest(memory_ids=memory_ids, strategy=strategy)
    ctx = SimpleNamespace(workspace_id=_WS)
    with patch.object(memory_stats, "_get_memory_service", return_value=service), \
         patch.object(memory_stats, "_get_agent_ids", return_value=[7]), \
         patch.object(memory_stats, "_fetch_all_scoped_memories", AsyncMock(return_value=_items())):
        return asyncio.run(memory_stats.consolidate_memories(body, ctx=ctx, db=MagicMock()))


def test_consolidate_stores_one_merged_memory_and_deletes_every_original():
    service = _fake_service()

    result = _consolidate(service, ["a", "b", "c", "d"])

    assert result["success"] is True
    assert result["deleted_count"] == 4
    service.store_long_term.assert_awaited_once()
    scoped = {call.args[1]: sorted(call.args[0]) for call in service.delete_memories_scoped.await_args_list}
    assert scoped == {_WS_NS: ["a"], _AGENT_NS: ["b", "c"]}
    for call in service.delete_memories_scoped.await_args_list:
        assert call.args[2] == str(_WS)
    service.delete_memory.assert_awaited_once_with("d", workspace_id=str(_WS))


def test_consolidate_counts_only_what_the_store_deleted():
    service = _fake_service()
    service.delete_memories_scoped = AsyncMock(side_effect=lambda ids, ns, ws: ns == _WS_NS)

    result = _consolidate(service, ["a", "b", "c"])

    assert result["deleted_count"] == 1


def test_summarise_runs_the_llm_call_off_the_event_loop_thread():
    service = _fake_service()
    seen = {}

    def _create(**kwargs):
        seen["thread"] = threading.current_thread()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="merged"))])

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    with patch("openai.OpenAI", return_value=client):
        result = _consolidate(service, ["a", "b"], strategy="summarise")

    assert result["success"] is True
    assert seen["thread"] is not threading.main_thread()
    assert service.store_long_term.await_args.kwargs["content"] == "merged"


def _real_service_with_store(store: AsyncMock) -> UnifiedMemoryService:
    service = UnifiedMemoryService.__new__(UnifiedMemoryService)
    service._durable = store
    return service


def test_delete_memories_scoped_refuses_another_workspaces_namespace():
    store = MagicMock()
    store.delete = AsyncMock(return_value=True)
    service = _real_service_with_store(store)

    other = f"mem:{uuid.uuid4()}:agent:7"
    lookalike = f"{_WS_NS}x:agent:7"

    assert asyncio.run(service.delete_memories_scoped(["a"], other, str(_WS))) is False
    assert asyncio.run(service.delete_memories_scoped(["a"], lookalike, str(_WS))) is False
    store.delete.assert_not_awaited()


def test_delete_memories_scoped_deletes_inside_the_workspace():
    store = MagicMock()
    store.delete = AsyncMock(return_value=True)
    service = _real_service_with_store(store)

    assert asyncio.run(service.delete_memories_scoped(["b", "c"], _AGENT_NS, str(_WS))) is True
    store.delete.assert_awaited_once_with(memory_ids=["b", "c"], user_id=_AGENT_NS, workspace_id=str(_WS))
