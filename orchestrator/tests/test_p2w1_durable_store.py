"""PRD-187 S1 — in-process durable memory store (the un-split, P2-06).

The dead mem0-fork-as-a-service is replaced by ``DurableMemoryStore``: a second
collection on the already-running Qdrant, behind the untouched
``UnifiedMemoryService`` L3 seam. These tests pin:

1. Write→read roundtrip shape — the upsert payload carries namespace /
   workspace_id / content / content_hash, and search returns the memory-item
   shape L3 consumers already read (``memory`` / ``score`` / ``created_at``).
2. Fail-closed tenancy — every write is workspace-stamped, whether the caller
   passed ``workspace_id`` explicitly or it must be parsed from the namespace.
3. Content-hash dedup — the same text in the same namespace stores once.
4. Delete ownership — ids belonging to another namespace are never deleted.
5. Relevance floor — sub-floor hits never leave the adapter (PRD-159 S3,
   rehoused from the retired client).
6. The retirement is total — no ``Mem0Client`` / ``MEM0_`` config / breaker /
   ``infer=`` smuggling anywhere in the live source tree.

All Qdrant/embedding IO is mocked at the boundary — no network, no Qdrant.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.memory.durable_store import (  # noqa: E402
    DurableMemoryStore,
    filter_by_relevance_floor,
    workspace_from_namespace,
)

WS = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Test double — a store wired to a fake Qdrant client + fake embedder
# ---------------------------------------------------------------------------

def _fake_qdrant(existing_by_hash=None, query_hits=None, retrieved=None):
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.count = AsyncMock(return_value=SimpleNamespace(count=0))
    client.retrieve = AsyncMock(return_value=retrieved or [])
    client.query_points = AsyncMock(
        return_value=SimpleNamespace(points=query_hits or [])
    )
    # first scroll call is the content-hash dedup lookup; default: no match
    client.scroll = AsyncMock(return_value=(existing_by_hash or [], None))
    return client


def _make_store(client) -> DurableMemoryStore:
    with patch("modules.memory.durable_store.AsyncQdrantClient", return_value=client), \
         patch("modules.memory.durable_store.EmbeddingManager") as embedder_cls:
        embedder_cls.return_value.generate_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        store = DurableMemoryStore()
    return store


# ---------------------------------------------------------------------------
# 1+2. Roundtrip shape + fail-closed workspace stamping
# ---------------------------------------------------------------------------

def test_durable_store_write_read_roundtrip():
    client = _fake_qdrant()
    store = _make_store(client)

    result = asyncio.run(store.add(
        messages=[{"role": "user", "content": "gerard prefers PRDs over ad-hoc builds"}],
        user_id=f"mem:{WS}",
        metadata={"category": "user_fact"},
        workspace_id=WS,
    ))
    assert result["success"] is True and result.get("id")

    point = client.upsert.call_args.kwargs["points"][0]
    assert point.payload["namespace"] == f"mem:{WS}"
    assert point.payload["workspace_id"] == WS
    assert point.payload["content"] == "gerard prefers PRDs over ad-hoc builds"
    assert point.payload["content_hash"]
    assert point.payload["created_at"]
    assert point.payload["metadata"] == {"category": "user_fact"}

    # Read side: a hit comes back in the memory-item shape consumers read.
    hit = SimpleNamespace(id="p1", score=0.92, payload=point.payload)
    client.query_points = AsyncMock(return_value=SimpleNamespace(points=[hit]))
    found = asyncio.run(store.search("what does gerard prefer", user_id=f"mem:{WS}"))
    assert found == [{
        "id": "p1",
        "memory": "gerard prefers PRDs over ad-hoc builds",
        "score": 0.92,
        "metadata": {"category": "user_fact"},
        "created_at": point.payload["created_at"],
        "namespace": f"mem:{WS}",
    }]


def test_durable_write_is_workspace_scoped_without_explicit_ws():
    # Scoped writers (recipe/agent namespaces) may omit workspace_id — the
    # adapter parses it from the namespace so tenancy stays fail-closed.
    client = _fake_qdrant()
    store = _make_store(client)
    asyncio.run(store.add(
        messages=[{"role": "user", "content": "step learning"}],
        user_id=f"mem:{WS}:recipe:7",
    ))
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["workspace_id"] == WS


def test_workspace_from_namespace_parse():
    assert workspace_from_namespace(f"mem:{WS}") == WS
    assert workspace_from_namespace(f"mem:{WS}:agent:42") == WS
    assert workspace_from_namespace(f"mem:{WS}:recipe:9:agent:1") == WS
    assert workspace_from_namespace("not-a-namespace") is None
    assert workspace_from_namespace("") is None


# ---------------------------------------------------------------------------
# 3. Content-hash dedup
# ---------------------------------------------------------------------------

def test_add_dedups_same_content_in_namespace():
    existing = SimpleNamespace(id="already-there", payload={})
    client = _fake_qdrant(existing_by_hash=[existing])
    store = _make_store(client)

    result = asyncio.run(store.add(
        messages=[{"role": "user", "content": "the same fact"}],
        user_id=f"mem:{WS}", workspace_id=WS,
    ))
    assert result == {"success": True, "id": "already-there", "deduped": True}
    client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Delete ownership check
# ---------------------------------------------------------------------------

def test_delete_refuses_foreign_namespace_ids():
    foreign = SimpleNamespace(id="m-x", payload={"namespace": f"mem:{WS}:agent:9"})
    client = _fake_qdrant(retrieved=[foreign])
    store = _make_store(client)

    ok = asyncio.run(store.delete(["m-x"], user_id=f"mem:{WS}"))
    assert ok is False
    client.delete.assert_not_called()


def test_delete_removes_owned_ids():
    owned = SimpleNamespace(id="m-1", payload={"namespace": f"mem:{WS}"})
    client = _fake_qdrant(retrieved=[owned])
    store = _make_store(client)

    ok = asyncio.run(store.delete(["m-1"], user_id=f"mem:{WS}"))
    assert ok is True
    assert client.delete.call_args.kwargs["points_selector"] == ["m-1"]


# ---------------------------------------------------------------------------
# 5. Relevance floor (PRD-159 S3, rehoused)
# ---------------------------------------------------------------------------

def test_search_applies_relevance_floor():
    strong = SimpleNamespace(id="s", score=0.9, payload={"namespace": f"mem:{WS}", "content": "strong"})
    weak = SimpleNamespace(id="w", score=0.01, payload={"namespace": f"mem:{WS}", "content": "weak"})
    client = _fake_qdrant(query_hits=[strong, weak])
    store = _make_store(client)

    found = asyncio.run(store.search("q", user_id=f"mem:{WS}", limit=5))
    assert [f["id"] for f in found] == ["s"]


def test_floor_helper_keeps_unscored_drops_subfloor():
    rows = [{"score": None}, {"score": 0.5}, {"score": 0.1}]
    assert filter_by_relevance_floor(rows, 0.3) == [{"score": None}, {"score": 0.5}]
    assert filter_by_relevance_floor(rows, 0) == rows


# ---------------------------------------------------------------------------
# 6. The retirement is total — no fork remnants in live source
# ---------------------------------------------------------------------------

_SOURCE_DIRS = ("modules", "services", "core", "api", "consumers", "evals")
_DEAD_TOKENS = ("Mem0Client", "MEM0_API_URL", "MEM0_HEALTH_PROBE", "mem0_client", "_mem0")


def _iter_source_files():
    for d in _SOURCE_DIRS:
        root = _ORCH / d
        if root.exists():
            yield from root.rglob("*.py")
    yield _ORCH / "main.py"
    yield _ORCH / "config.py"


def test_no_mem0_references_remain():
    offenders = []
    for path in _iter_source_files():
        text = path.read_text(errors="ignore")
        for token in _DEAD_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(_ORCH)}: {token}")
    assert not offenders, f"mem0 fork remnants survive the un-split: {offenders}"
    assert not (_ORCH / "modules" / "memory" / "integrations").exists(), (
        "modules/memory/integrations must be deleted with the fork"
    )


def test_no_infer_smuggling_in_memory_tree():
    # The infer= flag existed to bypass the fork's server-side extraction;
    # with the store gone the concept must not survive in the memory module.
    offenders = []
    for path in (_ORCH / "modules" / "memory").rglob("*.py"):
        if "infer=" in path.read_text(errors="ignore"):
            offenders.append(str(path.relative_to(_ORCH)))
    assert not offenders, f"infer= survives in: {offenders}"


def test_durable_config_knobs_exist():
    from config import config

    assert isinstance(config.DURABLE_MEMORY_COLLECTION, str) and config.DURABLE_MEMORY_COLLECTION
    assert int(config.DURABLE_MEMORY_PROBE_INTERVAL_SECONDS) > 0
    for gone in ("MEM0_API_URL", "MEM0_API_KEY", "MEM0_HEALTH_PROBE_ENABLED"):
        assert not hasattr(config, gone), f"config.{gone} must be retired with the fork"


def test_stats_delete_passes_workspace_scope():
    # PRD-187 S6: the Explorer delete endpoint calls the seam with its
    # REQUIRED workspace scope — the mem0-era call omitted it and raised
    # TypeError on every invocation (the endpoint never deleted anything).
    src = (_ORCH / "api" / "memory_stats.py").read_text()
    assert "service.delete_memory(\n            memory_id, workspace_id=str(ctx.workspace_id)\n        )" in src
