"""
Unit tests for VectorFieldSharedContext — PRD-108
==================================================

Strategy:
  - AsyncQdrantClient and EmbeddingManager are patched at import-time so no
    real network calls are made.
  - config values are fixed to their documented defaults (decay_rate=0.1,
    reinforce_bonus=0.05, reinforce_cap=2.0, archival_threshold=0.05,
    boundary_permeability=1.0, dimension=2048).
  - Each test class owns its own adapter instance via a fixture.
"""

from __future__ import annotations

import hashlib
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the orchestrator root importable
# ---------------------------------------------------------------------------
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

# ---------------------------------------------------------------------------
# Import VectorFieldSharedContext under stubbed heavy deps, then restore.
# Monkey-patching the instance (not the constructor) avoids re-import races.
# ---------------------------------------------------------------------------
def _import_vector_field_isolated():
    """Load VectorFieldSharedContext with qdrant_client / core.llm / core.ports
    stubbed, then restore sys.modules.

    The adapter captures the real SharedContextPort base at class-definition
    time, so the stubs can be torn down once the import completes. Restoring
    sys.modules stops the MagicMock ``core.ports.context`` (and the qdrant /
    core.llm stubs) from leaking into sibling modules' collection. (PRD-142
    W2-S2b.)
    """
    _keys = (
        "qdrant_client", "qdrant_client.models", "config",
        "core.llm", "core.llm.embedding_manager", "core.ports.context",
    )
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        _qdrant_stub = MagicMock()
        _qdrant_stub.AsyncQdrantClient = MagicMock
        # Expose every qdrant_client.models symbol the adapter uses
        _models_stub = MagicMock()
        for _sym in ("Distance", "FieldCondition", "Filter", "MatchValue",
                     "PayloadSchemaType", "PointStruct", "VectorParams"):
            setattr(_models_stub, _sym, MagicMock())
        _qdrant_stub.models = _models_stub
        sys.modules.setdefault("qdrant_client", _qdrant_stub)
        sys.modules.setdefault("qdrant_client.models", _models_stub)

        _fake_config_for_import = MagicMock()
        _fake_config_for_import.QDRANT_URL = "http://localhost:6333"
        _fake_config_for_import.QDRANT_API_KEY = ""
        _fake_config_for_import.FIELD_DECAY_RATE = 0.1
        _fake_config_for_import.FIELD_REINFORCE_BONUS = 0.05
        _fake_config_for_import.FIELD_REINFORCE_CAP = 2.0
        _fake_config_for_import.FIELD_ARCHIVAL_THRESHOLD = 0.05
        _fake_config_for_import.FIELD_BOUNDARY_PERMEABILITY = 1.0
        _fake_config_for_import.FIELD_EMBEDDING_DIM = 2048

        # Import the real SharedContextPort ABC BEFORE stubbing core.*
        # (vector_field.py subclasses it — MagicMock would break object.__new__)
        from core.ports.context import SharedContextPort

        # Stub heavy transitive imports so we don't need the full dep tree
        _config_mod = MagicMock()
        _config_mod.config = _fake_config_for_import
        sys.modules.setdefault("config", _config_mod)

        # Stub core.llm chain so EmbeddingManager import doesn't pull the world
        _core_stub = MagicMock()
        sys.modules.setdefault("core.llm", _core_stub)
        sys.modules.setdefault("core.llm.embedding_manager", _core_stub)

        # Re-register a ports module exposing the real ABC so the subclass resolves
        _ports_ctx_mod = MagicMock()
        _ports_ctx_mod.SharedContextPort = SharedContextPort
        sys.modules["core.ports.context"] = _ports_ctx_mod

        with (
            patch("modules.context.adapters.vector_field.AsyncQdrantClient", return_value=MagicMock()),
            patch("modules.context.adapters.vector_field.EmbeddingManager", return_value=MagicMock()),
            patch("modules.context.adapters.vector_field.config", _fake_config_for_import),
        ):
            from modules.context.adapters.vector_field import VectorFieldSharedContext
        return VectorFieldSharedContext
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                sys.modules.pop(_k, None)
            else:
                sys.modules[_k] = _v


VectorFieldSharedContext = _import_vector_field_isolated()

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 2048


def _make_point(
    point_id: str,
    key: str = "fact",
    value: str = "the sky is blue",
    strength: float = 1.0,
    access_count: int = 0,
    last_accessed: datetime | None = None,
    agent_id: int = 1,
) -> MagicMock:
    """Build a Qdrant-like ScoredPoint / Record mock."""
    now = last_accessed or datetime.now(timezone.utc)
    point = MagicMock()
    point.id = point_id
    point.payload = {
        "key": key,
        "value": value,
        "strength": strength,
        "access_count": access_count,
        "last_accessed": now.isoformat(),
        "created_at": now.isoformat(),
        "agent_id": agent_id,
        "content_hash": hashlib.sha256(f"{key}: {value}".encode()).hexdigest(),
    }
    return point


def _make_scored_hit(point_id: str, cosine: float, **payload_kwargs) -> MagicMock:
    """Build a ScoredPoint mock (has .score and .payload)."""
    hit = _make_point(point_id, **payload_kwargs)
    hit.score = cosine
    return hit


def _mock_query_response(points: list) -> MagicMock:
    """Wrap a list of ScoredPoint mocks in a QueryResponse-like object."""
    resp = MagicMock()
    resp.points = points
    return resp


@pytest.fixture
def mock_qdrant():
    """Patch AsyncQdrantClient for every test that uses it."""
    client = MagicMock()
    # Every method the adapter awaits must be an AsyncMock
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = AsyncMock()
    _empty_response = MagicMock()
    _empty_response.points = []
    client.query_points = AsyncMock(return_value=_empty_response)
    client.scroll = AsyncMock(return_value=([], None))
    client.retrieve = AsyncMock(return_value=[])
    client.set_payload = AsyncMock()
    # PRD-108: destroy is delete-by-filter on the shared collection, not
    # delete_collection (collections are no longer per-field).
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.generate_embedding.return_value = FAKE_EMBEDDING
    return embedder


@pytest.fixture
def adapter(mock_qdrant, mock_embedder):
    """
    A VectorFieldSharedContext instance with config values pinned to defaults
    and external I/O replaced by test doubles.

    We bypass __init__ entirely to avoid any constructor-level import races.
    The class is already imported at module collection time (see top of file).
    """
    inst = object.__new__(VectorFieldSharedContext)
    inst._client = mock_qdrant
    inst._embedder = mock_embedder
    inst._decay_rate = 0.1
    inst._reinforce_bonus = 0.05
    inst._reinforce_cap = 2.0
    inst._archival_threshold = 0.05
    inst._boundary_permeability = 1.0
    inst._dimension = 2048
    # PRD-108: the single shared collection is bootstrapped once via
    # ensure_shared_collection(). Behavioural tests pin it done so the
    # method-under-test runs without the bootstrap side-trip; the dedicated
    # bootstrap tests flip it back to False to exercise that path.
    inst._bootstrap_done = True
    return inst


# ===========================================================================
# _compute_decayed_strength — pure-math, no IO
# ===========================================================================


class TestComputeDecayedStrength:
    """S(t) = S₀ × e^(−λt) × access_boost, λ=0.1, boost capped at 2.0."""

    def test_zero_age_no_accesses(self, adapter):
        result = adapter._compute_decayed_strength(1.0, age_hours=0.0, access_count=0)
        # e^0 = 1, access_boost = 1 + 0*0.05 = 1.0
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_decay_after_one_hour(self, adapter):
        # S(1) = 1.0 × e^(-0.1 × 1) × 1.0
        expected = math.exp(-0.1 * 1.0)
        result = adapter._compute_decayed_strength(1.0, age_hours=1.0, access_count=0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_half_life_approximately_6_93_hours(self, adapter):
        # half-life = ln(2)/0.1 ≈ 6.931
        half_life = math.log(2) / 0.1
        result = adapter._compute_decayed_strength(1.0, age_hours=half_life, access_count=0)
        assert math.isclose(result, 0.5, rel_tol=1e-4)

    def test_access_boost_formula(self, adapter):
        # access_count=4 → boost = 1 + 4*0.05 = 1.2
        result = adapter._compute_decayed_strength(1.0, age_hours=0.0, access_count=4)
        assert math.isclose(result, 1.2, rel_tol=1e-9)

    def test_access_boost_capped_at_2(self, adapter):
        # access_count=100 → uncapped boost = 1 + 100*0.05 = 6.0 → capped at 2.0
        result = adapter._compute_decayed_strength(1.0, age_hours=0.0, access_count=100)
        assert math.isclose(result, 2.0, rel_tol=1e-9)

    def test_access_boost_cap_boundary_exactly_20_accesses(self, adapter):
        # access_count=20 → boost = 1 + 20*0.05 = 2.0 exactly → at cap, not over
        result = adapter._compute_decayed_strength(1.0, age_hours=0.0, access_count=20)
        assert math.isclose(result, 2.0, rel_tol=1e-9)

    def test_access_boost_one_below_cap(self, adapter):
        # access_count=19 → boost = 1 + 19*0.05 = 1.95 → under cap
        result = adapter._compute_decayed_strength(1.0, age_hours=0.0, access_count=19)
        assert math.isclose(result, 1.95, rel_tol=1e-9)

    def test_initial_strength_scales_output(self, adapter):
        s0 = 0.5
        result = adapter._compute_decayed_strength(s0, age_hours=0.0, access_count=0)
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    def test_decay_combined_with_access_boost(self, adapter):
        # S(2, count=10) = 1.0 × e^(-0.2) × 1.5
        expected = math.exp(-0.2) * 1.5
        result = adapter._compute_decayed_strength(1.0, age_hours=2.0, access_count=10)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_zero_initial_strength_always_zero(self, adapter):
        result = adapter._compute_decayed_strength(0.0, age_hours=5.0, access_count=50)
        assert result == 0.0

    def test_large_age_approaches_zero(self, adapter):
        # After 1000 hours the value should be negligible
        result = adapter._compute_decayed_strength(1.0, age_hours=1000.0, access_count=0)
        assert result < 1e-40


# ===========================================================================
# create_context
# ===========================================================================


class TestCreateContext:
    @pytest.mark.asyncio
    async def test_returns_string_uuid(self, adapter, mock_qdrant):
        field_id = await adapter.create_context([1, 2, 3])
        assert isinstance(field_id, str)
        assert len(field_id) == 36  # UUID4 canonical form

    @pytest.mark.asyncio
    async def test_bootstraps_single_shared_collection(self, adapter, mock_qdrant):
        # PRD-108: ONE shared collection (field_memory), not one per field.
        adapter._bootstrap_done = False
        mock_qdrant.collection_exists.return_value = False
        await adapter.create_context([1])
        mock_qdrant.create_collection.assert_awaited_once()
        call_kwargs = mock_qdrant.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "field_memory"

    @pytest.mark.asyncio
    async def test_creates_payload_indexes(self, adapter, mock_qdrant):
        adapter._bootstrap_done = False
        mock_qdrant.collection_exists.return_value = False
        await adapter.create_context([1])
        # Four indexes: field_id, content_hash, agent_id, created_at
        assert mock_qdrant.create_payload_index.await_count == 4

    @pytest.mark.asyncio
    async def test_bootstrap_skipped_when_collection_present(self, adapter, mock_qdrant):
        # Idempotent: an existing collection is not re-created.
        adapter._bootstrap_done = False
        mock_qdrant.collection_exists.return_value = True
        await adapter.create_context([1])
        mock_qdrant.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_initial_data_no_inject(self, adapter, mock_qdrant):
        await adapter.create_context([1], initial_data=None)
        mock_qdrant.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_initial_data_triggers_inject(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)  # nothing exists yet → no dedup
        await adapter.create_context([1], initial_data={"goal": "win"})
        mock_qdrant.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initial_data_multiple_keys(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)
        await adapter.create_context([1], initial_data={"a": "1", "b": "2", "c": "3"})
        assert mock_qdrant.upsert.await_count == 3

    @pytest.mark.asyncio
    async def test_unique_ids_per_call(self, adapter, mock_qdrant):
        id1 = await adapter.create_context([1])
        id2 = await adapter.create_context([1])
        assert id1 != id2


# ===========================================================================
# inject
# ===========================================================================


class TestInject:
    @pytest.mark.asyncio
    async def test_new_pattern_calls_upsert(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)  # no existing hash
        await adapter.inject("ctx-1", "key", "value", agent_id=7)
        mock_qdrant.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_payload_fields_present(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)

        # Capture the payload dict passed to PointStruct directly
        captured: dict = {}

        import modules.context.adapters.vector_field as _vf_mod

        _orig = _vf_mod.PointStruct

        def _capture_point(**kwargs):
            captured.update(kwargs.get("payload", {}))
            return _orig(**kwargs) if callable(_orig) else MagicMock()

        with patch.object(_vf_mod, "PointStruct", side_effect=_capture_point):
            await adapter.inject("ctx-1", "topic", "result", agent_id=3, strength=0.8)

        assert captured["key"] == "topic"
        assert captured["value"] == "result"
        assert captured["agent_id"] == 3
        # boundary_permeability=1.0 → effective_strength = 0.8 * 1.0 = 0.8
        assert math.isclose(captured["strength"], 0.8, rel_tol=1e-9)
        assert captured["access_count"] == 0
        assert "content_hash" in captured
        assert "created_at" in captured
        assert "last_accessed" in captured

    @pytest.mark.asyncio
    async def test_boundary_permeability_scales_strength(self, adapter, mock_qdrant, mock_embedder):
        """effective_strength = strength × boundary_permeability"""
        adapter._boundary_permeability = 0.5
        mock_qdrant.scroll.return_value = ([], None)

        captured: dict = {}

        import modules.context.adapters.vector_field as _vf_mod

        _orig = _vf_mod.PointStruct

        def _capture_point(**kwargs):
            captured.update(kwargs.get("payload", {}))
            return _orig(**kwargs) if callable(_orig) else MagicMock()

        with patch.object(_vf_mod, "PointStruct", side_effect=_capture_point):
            await adapter.inject("ctx-1", "k", "v", agent_id=1, strength=1.0)

        assert math.isclose(captured["strength"], 0.5, rel_tol=1e-9)

    @pytest.mark.asyncio
    async def test_dedup_reinforces_existing_not_upsert(self, adapter, mock_qdrant, mock_embedder):
        """Injecting the same content twice reinforces, does not insert a second point."""
        content = "key: value"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        existing = _make_point("existing-id", key="key", value="value")
        existing.payload["content_hash"] = content_hash
        mock_qdrant.scroll.return_value = ([existing], None)
        mock_qdrant.retrieve.return_value = [existing]

        await adapter.inject("ctx-1", "key", "value", agent_id=99)

        mock_qdrant.upsert.assert_not_awaited()
        mock_qdrant.set_payload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dedup_increments_access_count(self, adapter, mock_qdrant, mock_embedder):
        existing = _make_point("pt-1", access_count=3)
        mock_qdrant.scroll.return_value = ([existing], None)
        mock_qdrant.retrieve.return_value = [existing]

        await adapter.inject("ctx-1", existing.payload["key"], existing.payload["value"], agent_id=1)

        set_payload_call = mock_qdrant.set_payload.call_args
        assert set_payload_call.kwargs["payload"]["access_count"] == 4

    @pytest.mark.asyncio
    async def test_content_hash_is_sha256_of_key_colon_value(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)
        key, value = "alpha", "beta"
        expected_hash = hashlib.sha256(f"{key}: {value}".encode()).hexdigest()

        captured: dict = {}

        import modules.context.adapters.vector_field as _vf_mod

        _orig = _vf_mod.PointStruct

        def _capture_point(**kwargs):
            captured.update(kwargs.get("payload", {}))
            return _orig(**kwargs) if callable(_orig) else MagicMock()

        with patch.object(_vf_mod, "PointStruct", side_effect=_capture_point):
            await adapter.inject("ctx-1", key, value, agent_id=1)

        assert captured["content_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_embedding_called_with_key_colon_value(self, adapter, mock_qdrant, mock_embedder):
        mock_qdrant.scroll.return_value = ([], None)
        await adapter.inject("ctx-1", "my_key", "my_value", agent_id=1)
        mock_embedder.generate_embedding.assert_awaited_once_with("my_key: my_value")

    @pytest.mark.asyncio
    async def test_embedding_not_called_on_dedup(self, adapter, mock_qdrant, mock_embedder):
        existing = _make_point("pt-1")
        mock_qdrant.scroll.return_value = ([existing], None)
        mock_qdrant.retrieve.return_value = [existing]

        await adapter.inject("ctx-1", existing.payload["key"], existing.payload["value"], agent_id=1)

        mock_embedder.generate_embedding.assert_not_awaited()


# ===========================================================================
# query — resonance scoring, archival filtering, Hebbian reinforcement
# ===========================================================================


class TestQuery:
    @pytest.mark.asyncio
    async def test_resonance_is_cosine_squared_times_decayed_strength(
        self, adapter, mock_qdrant, mock_embedder
    ):
        cosine = 0.9
        strength = 1.0
        hit = _make_scored_hit("pt-1", cosine=cosine, strength=strength, access_count=0)
        mock_qdrant.query_points.return_value = _mock_query_response([hit])
        mock_qdrant.retrieve.return_value = [hit]

        results = await adapter.query("ctx-1", "query text", agent_id=1, top_k=5)

        assert len(results) == 1
        # age ≈ 0 → decayed_strength ≈ strength = 1.0
        expected_score = (cosine ** 2) * 1.0
        assert math.isclose(results[0]["score"], expected_score, rel_tol=1e-3)

    @pytest.mark.asyncio
    async def test_archival_threshold_filters_weak_patterns(
        self, adapter, mock_qdrant, mock_embedder
    ):
        # Pattern with very old last_accessed will have decayed_strength < 0.05
        old_time = datetime.now(timezone.utc) - timedelta(hours=500)
        weak_hit = _make_scored_hit(
            "pt-weak", cosine=0.99, strength=0.04, access_count=0, last_accessed=old_time
        )
        mock_qdrant.query_points.return_value = _mock_query_response([weak_hit])

        results = await adapter.query("ctx-1", "query", agent_id=1)

        assert results == []

    @pytest.mark.asyncio
    async def test_above_archival_threshold_included(
        self, adapter, mock_qdrant, mock_embedder
    ):
        # strength=1.0, age=0 → decayed_strength=1.0 > 0.05
        strong_hit = _make_scored_hit("pt-strong", cosine=0.8, strength=1.0, access_count=0)
        mock_qdrant.query_points.return_value = _mock_query_response([strong_hit])
        mock_qdrant.retrieve.return_value = [strong_hit]

        results = await adapter.query("ctx-1", "query", agent_id=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(
        self, adapter, mock_qdrant, mock_embedder
    ):
        hit_a = _make_scored_hit("pt-a", cosine=0.5, strength=1.0, access_count=0)
        hit_b = _make_scored_hit("pt-b", cosine=0.9, strength=1.0, access_count=0)
        hit_c = _make_scored_hit("pt-c", cosine=0.7, strength=1.0, access_count=0)
        mock_qdrant.query_points.return_value = _mock_query_response([hit_a, hit_b, hit_c])
        mock_qdrant.retrieve.return_value = [hit_a, hit_b, hit_c]

        results = await adapter.query("ctx-1", "query", agent_id=1)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_top_k_limits_returned_results(
        self, adapter, mock_qdrant, mock_embedder
    ):
        hits = [
            _make_scored_hit(f"pt-{i}", cosine=0.9 - i * 0.01, strength=1.0, access_count=0)
            for i in range(10)
        ]
        mock_qdrant.query_points.return_value = _mock_query_response(hits)
        mock_qdrant.retrieve.return_value = hits

        results = await adapter.query("ctx-1", "query", agent_id=1, top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_returns_required_fields(
        self, adapter, mock_qdrant, mock_embedder
    ):
        hit = _make_scored_hit("pt-1", cosine=0.8, strength=1.0, access_count=0)
        mock_qdrant.query_points.return_value = _mock_query_response([hit])
        mock_qdrant.retrieve.return_value = [hit]

        results = await adapter.query("ctx-1", "query", agent_id=1)

        r = results[0]
        assert "id" in r
        assert "key" in r
        assert "value" in r
        assert "score" in r
        assert "agent_id" in r
        assert "decayed_strength" in r
        assert "cosine_similarity" in r

    @pytest.mark.asyncio
    async def test_hebbian_reinforcement_called_for_returned_results(
        self, adapter, mock_qdrant, mock_embedder
    ):
        hit = _make_scored_hit("pt-1", cosine=0.8, strength=1.0, access_count=0)
        mock_qdrant.query_points.return_value = _mock_query_response([hit])
        mock_qdrant.retrieve.return_value = [hit]

        await adapter.query("ctx-1", "query", agent_id=1)

        mock_qdrant.retrieve.assert_awaited()
        mock_qdrant.set_payload.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_hebbian_when_no_results(
        self, adapter, mock_qdrant, mock_embedder
    ):
        mock_qdrant.query_points.return_value = _mock_query_response([])

        await adapter.query("ctx-1", "query", agent_id=1)

        mock_qdrant.set_payload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_query_points_overfetches_3x_top_k(
        self, adapter, mock_qdrant, mock_embedder
    ):
        mock_qdrant.query_points.return_value = _mock_query_response([])
        await adapter.query("ctx-1", "query", agent_id=1, top_k=7)

        qp_call = mock_qdrant.query_points.call_args
        assert qp_call.kwargs["limit"] == 21  # 7 × 3

    @pytest.mark.asyncio
    async def test_query_uses_embedding_of_query_string(
        self, adapter, mock_qdrant, mock_embedder
    ):
        mock_qdrant.query_points.return_value = _mock_query_response([])
        await adapter.query("ctx-1", "what is the plan?", agent_id=1)
        mock_embedder.generate_embedding.assert_awaited_once_with("what is the plan?")


# ===========================================================================
# Hebbian reinforcement — _reinforce_batch
# ===========================================================================


class TestHebbianReinforcement:
    @pytest.mark.asyncio
    async def test_single_pattern_no_co_access_bonus(
        self, adapter, mock_qdrant
    ):
        """Single pattern: boosted == initial_strength (no co-access)."""
        pt = _make_point("pt-1", strength=0.8, access_count=2)
        mock_qdrant.retrieve.return_value = [pt]

        await adapter._reinforce_batch(["pt-1"])

        payload = mock_qdrant.set_payload.call_args.kwargs["payload"]
        assert math.isclose(payload["strength"], 0.8, rel_tol=1e-9)
        assert payload["access_count"] == 3

    @pytest.mark.asyncio
    async def test_two_patterns_co_access_bonus(self, adapter, mock_qdrant):
        """Two patterns: each gets +2% × 1 co-pattern = 1.02 × initial_strength."""
        pt_a = _make_point("pt-a", strength=1.0, access_count=0)
        pt_b = _make_point("pt-b", strength=1.0, access_count=0)
        mock_qdrant.retrieve.return_value = [pt_a, pt_b]

        await adapter._reinforce_batch(["pt-a", "pt-b"])

        all_payloads = [call.kwargs["payload"] for call in mock_qdrant.set_payload.call_args_list]
        strengths = [p["strength"] for p in all_payloads]
        # Each: min(1.0 × (1 + 0.02×1), 1.0 × 2.0) = 1.02
        assert all(math.isclose(s, 1.02, rel_tol=1e-9) for s in strengths)

    @pytest.mark.asyncio
    async def test_co_access_bonus_capped_by_reinforce_cap(self, adapter, mock_qdrant):
        """Co-access bonus cannot push strength beyond initial_strength × reinforce_cap (2.0)."""
        # 100 co-patterns → uncapped would be 1 + 0.02×99 = 2.98, capped at 2.0
        pts = [_make_point(f"pt-{i}", strength=1.0, access_count=0) for i in range(100)]
        mock_qdrant.retrieve.return_value = pts

        await adapter._reinforce_batch([f"pt-{i}" for i in range(100)])

        all_payloads = [call.kwargs["payload"] for call in mock_qdrant.set_payload.call_args_list]
        for payload in all_payloads:
            assert payload["strength"] <= 2.0 + 1e-9  # never exceeds cap

    @pytest.mark.asyncio
    async def test_access_count_incremented_for_every_pattern(self, adapter, mock_qdrant):
        pt_a = _make_point("pt-a", access_count=5)
        pt_b = _make_point("pt-b", access_count=10)
        mock_qdrant.retrieve.return_value = [pt_a, pt_b]

        await adapter._reinforce_batch(["pt-a", "pt-b"])

        payloads = {
            call.kwargs["points"][0]: call.kwargs["payload"]["access_count"]
            for call in mock_qdrant.set_payload.call_args_list
        }
        assert payloads["pt-a"] == 6
        assert payloads["pt-b"] == 11

    @pytest.mark.asyncio
    async def test_no_op_when_empty_ids(self, adapter, mock_qdrant):
        """Empty ids list → retrieve not called, set_payload not called."""
        # simulate empty result
        mock_qdrant.retrieve.return_value = []
        await adapter._reinforce_batch([])
        mock_qdrant.set_payload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_payload_called_once_per_pattern(self, adapter, mock_qdrant):
        pts = [_make_point(f"pt-{i}", strength=1.0) for i in range(3)]
        mock_qdrant.retrieve.return_value = pts
        await adapter._reinforce_batch([f"pt-{i}" for i in range(3)])
        assert mock_qdrant.set_payload.await_count == 3


# ===========================================================================
# destroy_context
# ===========================================================================


class TestDestroyContext:
    @pytest.mark.asyncio
    async def test_deletes_by_filter_on_shared_collection(self, adapter, mock_qdrant):
        # PRD-108: destroy = delete-by-field_id-filter on the shared
        # collection, NOT a per-field delete_collection.
        await adapter.destroy_context("abc-123")
        mock_qdrant.delete.assert_awaited_once()
        assert mock_qdrant.delete.call_args.kwargs["collection_name"] == "field_memory"

    @pytest.mark.asyncio
    async def test_does_not_raise_on_qdrant_error(self, adapter, mock_qdrant):
        mock_qdrant.delete.side_effect = Exception("network error")
        # Must swallow the exception gracefully
        await adapter.destroy_context("abc-123")  # no raise

    @pytest.mark.asyncio
    async def test_delete_called_once(self, adapter, mock_qdrant):
        await adapter.destroy_context("some-id")
        assert mock_qdrant.delete.await_count == 1


# ===========================================================================
# measure_stability
# ===========================================================================


class TestMeasureStability:
    @pytest.mark.asyncio
    async def test_empty_field_returns_zero_stability(self, adapter, mock_qdrant):
        mock_qdrant.scroll.return_value = ([], None)
        result = await adapter.measure_stability("ctx-1")
        assert result == {"stability": 0.0, "pattern_count": 0, "avg_strength": 0.0}

    @pytest.mark.asyncio
    async def test_single_pattern_no_stddev(self, adapter, mock_qdrant):
        pt = _make_point("pt-1", strength=1.0, access_count=0)
        mock_qdrant.scroll.return_value = ([pt], None)

        result = await adapter.measure_stability("ctx-1")

        # One pattern: stddev=0, organization = max(0, 1 - 0/avg) = 1.0
        # stability = avg*0.6 + 1.0*0.4 ≈ decayed_strength*0.6 + 0.4
        assert result["pattern_count"] == 1
        assert 0.0 <= result["stability"] <= 1.5  # sanity bound

    @pytest.mark.asyncio
    async def test_stability_formula(self, adapter, mock_qdrant):
        """stability = avg_strength × 0.6 + organization × 0.4"""
        # Two patterns with equal (fresh) strength → stddev=0, organization=1
        pt_a = _make_point("pt-a", strength=0.8, access_count=0)
        pt_b = _make_point("pt-b", strength=0.8, access_count=0)
        mock_qdrant.scroll.return_value = ([pt_a, pt_b], None)

        result = await adapter.measure_stability("ctx-1")

        avg = result["avg_strength"]
        org = result["organization"]
        expected_stability = round(avg * 0.6 + org * 0.4, 4)
        assert math.isclose(result["stability"], expected_stability, rel_tol=1e-6)

    @pytest.mark.asyncio
    async def test_organization_decreases_with_high_variance(self, adapter, mock_qdrant):
        """Patterns with wildly different strengths yield lower organization."""
        pt_weak = _make_point("pt-weak", strength=0.01, access_count=0)
        pt_strong = _make_point("pt-strong", strength=1.0, access_count=0)
        mock_qdrant.scroll.return_value = ([pt_weak, pt_strong], None)

        result = await adapter.measure_stability("ctx-1")

        assert result["organization"] < 1.0

    @pytest.mark.asyncio
    async def test_returns_active_and_decayed_pattern_counts(self, adapter, mock_qdrant):
        strong = _make_point("pt-s", strength=1.0, access_count=0)
        old_time = datetime.now(timezone.utc) - timedelta(hours=500)
        weak = _make_point("pt-w", strength=0.04, access_count=0, last_accessed=old_time)
        mock_qdrant.scroll.return_value = ([strong, weak], None)

        result = await adapter.measure_stability("ctx-1")

        assert result["active_patterns"] >= 1
        assert result["decayed_patterns"] >= 1
        assert result["active_patterns"] + result["decayed_patterns"] == result["pattern_count"]

    @pytest.mark.asyncio
    async def test_scrolls_up_to_10000_points(self, adapter, mock_qdrant):
        mock_qdrant.scroll.return_value = ([], None)
        await adapter.measure_stability("ctx-1")
        scroll_call = mock_qdrant.scroll.call_args
        assert scroll_call.kwargs.get("limit") == 10000 or scroll_call.args[1] == 10000

    @pytest.mark.asyncio
    async def test_stability_bounded_between_0_and_max(self, adapter, mock_qdrant):
        """Stability should not exceed avg_strength × 0.6 + 1.0 × 0.4."""
        pts = [_make_point(f"pt-{i}", strength=1.0, access_count=0) for i in range(5)]
        mock_qdrant.scroll.return_value = (pts, None)

        result = await adapter.measure_stability("ctx-1")

        assert result["stability"] >= 0.0
        assert result["stability"] <= 2.0  # upper bound with max access_boost

    @pytest.mark.asyncio
    async def test_all_expected_keys_present(self, adapter, mock_qdrant):
        pt = _make_point("pt-1", strength=1.0, access_count=0)
        mock_qdrant.scroll.return_value = ([pt], None)

        result = await adapter.measure_stability("ctx-1")

        expected_keys = {
            "stability", "pattern_count", "avg_strength",
            "organization", "active_patterns", "decayed_patterns",
        }
        assert expected_keys.issubset(result.keys())


# ===========================================================================
# _find_by_hash
# ===========================================================================


class TestFindByHash:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, adapter, mock_qdrant):
        mock_qdrant.scroll.return_value = ([], None)
        result = await adapter._find_by_hash("ctx-1", "deadbeef")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_first_match(self, adapter, mock_qdrant):
        pt = _make_point("pt-1")
        mock_qdrant.scroll.return_value = ([pt], None)
        result = await adapter._find_by_hash("ctx-1", "any-hash")
        assert result is pt

    @pytest.mark.asyncio
    async def test_scroll_uses_correct_collection(self, adapter, mock_qdrant):
        mock_qdrant.scroll.return_value = ([], None)
        await adapter._find_by_hash("my-ctx", "hash123")
        scroll_call = mock_qdrant.scroll.call_args
        # PRD-108: lookups hit the shared collection; field isolation is via filter.
        assert scroll_call.kwargs.get("collection_name") == "field_memory"

    @pytest.mark.asyncio
    async def test_scroll_limits_to_1(self, adapter, mock_qdrant):
        mock_qdrant.scroll.return_value = ([], None)
        await adapter._find_by_hash("ctx-1", "hash123")
        scroll_call = mock_qdrant.scroll.call_args
        assert scroll_call.kwargs.get("limit") == 1
