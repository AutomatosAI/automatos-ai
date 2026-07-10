"""PRD-187 S3 — semantic, always-on L2 recall (memory J3 / C.3, P2-06).

The old recall was structurally near-dead: an ILIKE full-substring match
invoked ONLY when a temporal regex fired — "what did we learn about the
Shopify sync?" matched zero rows by construction, and ``memory_access_log``
had 6 entries lifetime. These tests pin the new contract:

1. A non-temporal query returns the relevant row by MEANING (vector match on
   the L2 mirror, hydrated from live Postgres rows, resonance-ranked).
2. The router launches L2 recall on every non-live-data turn — the
   ``is_temporal`` gate is gone (temporal queries keep the window'd listing).
3. Recall results pass the PRD-185 S11 injection guard: sub-floor and
   noise-typed rows (playbook summaries, heartbeat digests) never reach the
   bundle.
4. Archived rows never recall (hydration filters them); recalled rows are
   touched so access_count climbs — the signal S4's promotion reads.
5. Every L2 write mirrors into the durable store's l2 namespace.

Mock the durable store + DB hydration at the boundary; no Qdrant, no DB.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import modules.memory.unified_memory_service as ums_mod  # noqa: E402
from modules.memory.context_router import ContextRouter  # noqa: E402
from modules.memory.unified_memory_service import UnifiedMemoryService  # noqa: E402

WS = "11111111-2222-3333-4444-555555555555"


def _service_with_semantic_l2(hits, hydrated_rows):
    svc = UnifiedMemoryService.__new__(UnifiedMemoryService)
    svc._durable = MagicMock()
    svc._durable.search = AsyncMock(return_value=hits)
    svc._durable.add = AsyncMock(return_value={"success": True, "id": "vec-1"})
    svc._hydrate_l2_rows_sync = lambda ids: [r for r in hydrated_rows if r["id"] in ids]
    svc.touch_short_term = AsyncMock(return_value=True)
    return svc


def _hit(l2_id: str, score: float, content_type: str = "task_learning"):
    return {
        "id": f"vec-{l2_id}",
        "memory": "mirrored text",
        "score": score,
        "metadata": {"l2_id": l2_id, "content_type": content_type},
        "created_at": "2026-07-09T00:00:00+00:00",
    }


def _row(l2_id: str, content: str, content_type: str = "task_learning"):
    return {
        "id": l2_id,
        "content": content,
        "content_type": content_type,
        "importance": 0.6,
        "decay_score": 1.0,
        "access_count": 0,
        "metadata": {},
        "created_at": "2026-07-09T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# 1. Semantic recall on a non-temporal query
# ---------------------------------------------------------------------------

def test_l2_recall_semantic_non_temporal_query():
    row = _row("r1", "The Shopify sync fails when the webhook secret rotates")
    svc = _service_with_semantic_l2([_hit("r1", 0.87)], [row])

    out = asyncio.run(svc.search_short_term_semantic(
        WS, "what did we learn about the Shopify sync?",
    ))

    assert len(out) == 1
    assert out[0]["id"] == "r1"
    assert "Shopify sync" in out[0]["content"]
    assert out[0]["score"] > 0
    # the vector leg was queried under the workspace's l2 namespace
    assert svc._durable.search.await_args.kwargs["user_id"] == f"mem:{WS}:l2"


def test_l2_semantic_archived_rows_never_recall():
    # vector hit exists, but hydration (live-rows-only) returns nothing
    svc = _service_with_semantic_l2([_hit("gone", 0.9)], [])
    out = asyncio.run(svc.search_short_term_semantic(WS, "anything"))
    assert out == []


def test_l2_semantic_recall_touches_rows():
    row = _row("r1", "fact")
    svc = _service_with_semantic_l2([_hit("r1", 0.8)], [row])
    asyncio.run(svc.search_short_term_semantic(WS, "fact query"))
    svc.touch_short_term.assert_awaited_once_with("r1")


# ---------------------------------------------------------------------------
# 2. The router runs L2 recall every turn (gate is gone)
# ---------------------------------------------------------------------------

def _fake_router_service(semantic_rows):
    fake = MagicMock()
    fake.get_session = AsyncMock(return_value=None)
    fake.search_long_term = AsyncMock(return_value=[])
    fake.search_short_term = AsyncMock(return_value=[])
    fake.search_short_term_semantic = AsyncMock(return_value=semantic_rows)
    fake.get_all_daily_logs = AsyncMock(return_value=[])
    return fake


def _retrieve(router, fake, query, monkeypatch):
    monkeypatch.setattr(ums_mod, "get_unified_memory_service", lambda: fake)
    return asyncio.run(router.retrieve_context(
        workspace_id=WS, agent_id=1, query=query,
    ))


def test_l2_recall_runs_every_turn(monkeypatch):
    fake = _fake_router_service([_row("r1", "Shopify sync learning")])
    bundle = _retrieve(
        ContextRouter(), fake,
        "what did we learn about the Shopify sync?", monkeypatch,
    )

    fake.search_short_term_semantic.assert_awaited_once()
    fake.search_short_term.assert_not_awaited()  # no temporal signal, no ILIKE path
    assert [m["id"] for m in bundle.temporal_results] == ["r1"]


def test_temporal_query_keeps_windowed_listing(monkeypatch):
    fake = _fake_router_service([])
    _retrieve(ContextRouter(), fake, "what did we discuss last week?", monkeypatch)

    fake.search_short_term.assert_awaited_once()
    fake.search_short_term_semantic.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. The S11 injection guard bites at the router
# ---------------------------------------------------------------------------

def test_l2_recall_respects_injection_floor(monkeypatch):
    rows = [
        {**_row("good", "a real learning"), "score": 0.8},
        {**_row("noise", "playbook exec chatter", content_type="playbook_summary"), "score": 0.9},
        {**_row("weak", "barely related"), "score": 0.05},
    ]
    fake = _fake_router_service(rows)
    bundle = _retrieve(ContextRouter(), fake, "tell me about our learnings", monkeypatch)

    assert [m["id"] for m in bundle.temporal_results] == ["good"], (
        "noise-typed and sub-floor rows must be dropped by the shared "
        "filter_injectable_memories chokepoint (PRD-185 S11), not injected"
    )


# ---------------------------------------------------------------------------
# 5. Every L2 write mirrors into the durable l2 namespace
# ---------------------------------------------------------------------------

def test_store_short_term_mirrors_to_durable():
    async def run():
        svc = UnifiedMemoryService.__new__(UnifiedMemoryService)
        svc._durable = MagicMock()
        svc._durable.add = AsyncMock(return_value={"success": True})
        svc._store_short_term_sync = lambda *a: "row-42"
        out = await svc.store_short_term(
            workspace_id=WS, content="a fresh learning",
            content_type="task_learning", importance=0.7,
        )
        await asyncio.sleep(0)  # let the fire-and-forget mirror run
        return out, svc

    row_id, svc = asyncio.run(run())
    assert row_id == "row-42"
    kwargs = svc._durable.add.await_args.kwargs
    assert kwargs["user_id"] == f"mem:{WS}:l2"
    assert kwargs["metadata"]["l2_id"] == "row-42"
    assert kwargs["metadata"]["content_type"] == "task_learning"
    assert kwargs["workspace_id"] == WS
