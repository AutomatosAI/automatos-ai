"""PRD-196 S6 (P2-15, governance J.10) — GDPR subject-tags → real subject-erase.

Closes the field-memory + durable-memory ``# GDPR-GAP`` markers: the data-subject
tag is written at store time and subject-level erasure becomes a real filter-
delete over ``workspace_id AND subject_id`` (fail-closed tenancy). The gap ledger
turns dynamic — SQL stays a documented gap, pre-tag rows an untagged-history
caveat.

Pure: ``AsyncQdrantClient`` is mocked at the boundary (the PRD-187 S1 shape); we
assert the payload the store writes and the FILTER the erase builds — no Qdrant,
no network.
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

from modules.context.adapters.vector_field import VectorFieldSharedContext  # noqa: E402
from modules.memory.durable_store import DurableMemoryStore  # noqa: E402

WS = "11111111-2222-3333-4444-555555555555"
SUBJECT = "user:42"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _fake_qdrant():
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.count = AsyncMock(return_value=SimpleNamespace(count=3))
    client.retrieve = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))  # dedup: no existing
    client.set_payload = AsyncMock()
    return client


def _make_durable(client) -> DurableMemoryStore:
    with patch("modules.memory.durable_store.AsyncQdrantClient", return_value=client), \
         patch("modules.memory.durable_store.EmbeddingManager") as emb:
        emb.return_value.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return DurableMemoryStore()


def _make_field(client) -> VectorFieldSharedContext:
    with patch("modules.context.adapters.vector_field.AsyncQdrantClient", return_value=client), \
         patch("modules.context.adapters.vector_field.EmbeddingManager") as emb:
        emb.return_value.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return VectorFieldSharedContext()


def _filter_keys(flt) -> dict:
    """Map a Qdrant Filter's must-conditions to {key: matched value}."""
    out = {}
    for cond in flt.must:
        out[cond.key] = cond.match.value
    return out


# ---------------------------------------------------------------------------
# Durable store — subject tag on write, index, subject-scoped erase
# ---------------------------------------------------------------------------

def test_durable_add_writes_subject_tag():
    client = _fake_qdrant()
    store = _make_durable(client)
    asyncio.run(store.add(
        messages=[{"role": "user", "content": "gerard prefers PRDs"}],
        user_id=f"mem:{WS}", workspace_id=WS, subject_id=SUBJECT,
    ))
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["subject_id"] == SUBJECT
    assert payload["workspace_id"] == WS


def test_durable_untagged_write_stores_null_subject():
    client = _fake_qdrant()
    store = _make_durable(client)
    asyncio.run(store.add(
        messages=[{"role": "user", "content": "heartbeat tick"}],
        user_id=f"mem:{WS}", workspace_id=WS,  # no subject_id
    ))
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["subject_id"] is None, "an untagged write stores a null tag, never a wrong one"


def test_durable_ensure_collection_indexes_subject_id():
    client = _fake_qdrant()
    store = _make_durable(client)
    asyncio.run(store.ensure_collection())
    indexed = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
    assert "subject_id" in indexed


def test_durable_erase_subject_is_workspace_and_subject_scoped():
    client = _fake_qdrant()
    store = _make_durable(client)
    count = asyncio.run(store.erase_subject(WS, SUBJECT))
    assert count == 3  # from the fake count
    selector = client.delete.call_args.kwargs["points_selector"]
    keys = _filter_keys(selector.filter)
    assert keys == {"workspace_id": WS, "subject_id": SUBJECT}, "never workspace-wide — both bind"


# ---------------------------------------------------------------------------
# Field memory — subject tag on inject, subject-scoped erase
# ---------------------------------------------------------------------------

def test_field_inject_writes_subject_tag():
    client = _fake_qdrant()
    field = _make_field(client)
    asyncio.run(field.inject(
        context_id="f1", key="fact", value="gerard prefers PRDs", agent_id=1,
        provenance={"workspace_id": WS, "subject_id": SUBJECT},
    ))
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["subject_id"] == SUBJECT


def test_field_erase_subject_is_workspace_and_subject_scoped():
    client = _fake_qdrant()
    field = _make_field(client)
    count = asyncio.run(field.erase_subject(WS, SUBJECT))
    assert count == 3
    selector = client.delete.call_args.kwargs["points_selector"]
    keys = _filter_keys(selector.filter)
    assert keys == {"workspace_id": WS, "subject_id": SUBJECT}


# ---------------------------------------------------------------------------
# UnifiedMemoryService threads subject_id through the distill write seam
# ---------------------------------------------------------------------------

def test_store_two_tier_threads_subject_to_durable():
    from modules.memory import unified_memory_service as ums

    ums.UnifiedMemoryService.reset_instance()
    fake_durable = MagicMock()
    fake_durable.add = AsyncMock(return_value={"success": True, "id": "p1"})
    with patch("modules.memory.durable_store.DurableMemoryStore", return_value=fake_durable):
        svc = ums.UnifiedMemoryService()
    svc._durable = fake_durable

    asyncio.run(svc.store_two_tier(
        WS, [{"role": "user", "content": "gerard prefers PRDs"}],
        agent_id=None, tier="global", subject_id=SUBJECT,
    ))
    ums.UnifiedMemoryService.reset_instance()

    assert fake_durable.add.called
    assert fake_durable.add.call_args.kwargs.get("subject_id") == SUBJECT


# ---------------------------------------------------------------------------
# The gap ledger is dynamic + honest, and the closed markers are gone
# ---------------------------------------------------------------------------

def test_gap_ledger_now_honest(monkeypatch):
    from uuid import uuid4
    from services import gdpr_service

    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 4)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 2)
    monkeypatch.setattr(gdpr_service, "_erase_subject_sql", lambda db, ws, subject_id: {"deleted": 0})
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda *a, **k: None)

    result = gdpr_service.erase_data_subject(
        MagicMock(), workspace_id=uuid4(), subject_id="cust_9", requested_by="webhook:shopify"
    )
    # real deleted counts from the closed stores
    assert result["derived"]["field_memory_deleted"] == 4
    assert result["derived"]["durable_memory_deleted"] == 2
    # only SQL remains a gap; field/durable are closed
    stores = {g["store"] for g in result["gaps"]}
    assert stores == {"sql"}
    # untagged pre-tag history is reported honestly, never claimed erased
    assert set(result["untagged_history"]["stores"]) == {"field_memory", "durable_memory"}


def test_no_closed_gdpr_gap_markers_remain():
    field_src = (_ORCH / "modules" / "context" / "adapters" / "vector_field.py").read_text()
    durable_src = (_ORCH / "modules" / "memory" / "unified_memory_service.py").read_text()
    sql_src = (_ORCH / "services" / "gdpr_service.py").read_text()
    assert "GDPR-GAP" not in field_src, "field-memory GDPR-GAP marker must be gone (S6 closed it)"
    assert "GDPR-GAP" not in durable_src, "durable-memory GDPR-GAP marker must be gone (S6 closed it)"
    assert "GDPR-GAP" in sql_src, "the SQL GDPR-GAP marker stays — still a documented gap"
