"""PRD-178 S2 (F062) — the retrieval-trace inspector must NOT mutate the field.

The trace inspector (POST /missions/{id}/field/query, PRD-166 S4) ran the
*writing* ``query`` path, which triggers Hebbian ``_reinforce_batch`` — every
inspection bumped access_count / last_accessed / strength on the very patterns
it was observing, corrupting the signal it is meant to report.

The fix threads ``record_access=False`` through query so the inspector reads
without reinforcing. These tests use the same import-isolation harness as
tests/test_vector_field.py and assert, at the Qdrant boundary, that a
read-only query issues ZERO ``set_payload`` writes while a normal query still
reinforces (so we haven't disabled Hebbian learning on the live path).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

# Reuse the battle-tested isolated importer + fixtures from the PRD-108 suite.
from tests.test_vector_field import (  # noqa: E402
    VectorFieldSharedContext,
    _make_scored_hit,
    _mock_query_response,
)

pytestmark = pytest.mark.asyncio

FAKE_EMBEDDING = [0.1] * 2048


def _adapter_with_client(client: MagicMock) -> VectorFieldSharedContext:
    """Build an adapter whose Qdrant client and embedder are fully mocked."""
    adapter = VectorFieldSharedContext.__new__(VectorFieldSharedContext)
    adapter._client = client
    adapter._embedder = MagicMock()
    adapter._embedder.generate_embedding = AsyncMock(return_value=FAKE_EMBEDDING)
    adapter._decay_rate = 0.1
    adapter._reinforce_bonus = 0.05
    adapter._reinforce_cap = 2.0
    adapter._archival_threshold = 0.05
    adapter._boundary_permeability = 1.0
    adapter._dimension = 2048
    adapter._half_life_access_scale = 0.5
    adapter._bootstrap_done = True
    return adapter


def _client_with_hits() -> MagicMock:
    client = MagicMock()
    client.query_points = AsyncMock(
        return_value=_mock_query_response([
            _make_scored_hit("p1", cosine=0.9, strength=1.0, access_count=1),
            _make_scored_hit("p2", cosine=0.8, strength=1.0, access_count=1),
        ])
    )
    client.retrieve = AsyncMock(return_value=[
        _make_scored_hit("p1", cosine=0.9, strength=1.0, access_count=1),
        _make_scored_hit("p2", cosine=0.8, strength=1.0, access_count=1),
    ])
    client.set_payload = AsyncMock()
    return client


async def test_field_trace_readonly():
    """A read-only query returns the same ranked hits but issues NO writes:
    set_payload (the Hebbian reinforcement write) is never called."""
    client = _client_with_hits()
    adapter = _adapter_with_client(client)

    results = await adapter.query(
        context_id="field-x", query="what did we learn", agent_id=0,
        record_access=False,
    )

    assert results, "trace query still returns the patterns that fired"
    assert client.set_payload.await_count == 0, (
        "read-only trace must not reinforce (mutate) the observed field"
    )


async def test_normal_query_still_reinforces():
    """The default query path (record_access=True) still reinforces — the
    read-only flag must not silently disable Hebbian learning on live reads."""
    client = _client_with_hits()
    adapter = _adapter_with_client(client)

    results = await adapter.query(
        context_id="field-x", query="what did we learn", agent_id=0,
    )

    assert results
    assert client.set_payload.await_count >= 1, (
        "live query reinforces accessed patterns (Hebbian learning intact)"
    )
