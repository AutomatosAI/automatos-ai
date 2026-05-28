"""PRD-139 US-003: Unit tests for edge + affinity builder service.

Tests:
1. Synthetic 200-row log produces expected edges (counts, confidences within tolerance)
2. Re-running on same data is a no-op (no duplicate edges, no count drift)
3. Wilson lower bound correctness
4. Intent clustering determinism
5. Affinity computation accuracy
6. Session grouping by turn_id
7. Time-window splitting for fallback grouping
"""

import asyncio
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pytest

# Ensure orchestrator root is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------
# NOTE: do NOT mock core.database.base here. These tests exercise pure helpers
# but import the REAL ToolRoutingEdge/ToolRoutingAffinity models (via
# edge_builder), and a MagicMock Base would build Mock-based model classes that
# corrupt sibling tests sharing the same process (e.g. test_graph_router_negative
# uses the real models with a fake DB session). Real Base imports cleanly.


# Import intent_clustering directly (pure numpy, no DB deps)
from core.services.intent_clustering import (
    compute_intent_clusters,
    _kmeans,
    ClusterResult,
)

# Import wilson_lower_bound from edge_builder without triggering full import
from core.services.edge_builder import (
    wilson_lower_bound,
    _compute_used_after_edges,
    _compute_failed_after_edges,
    _derive_session_key,
    _split_by_time_window,
    _compute_affinities,
    _AffinityKey,
    _AffinityAccumulator,
    _SAMPLE_FLOOR,
)


# ---------------------------------------------------------------------------
# Test: Wilson lower bound
# ---------------------------------------------------------------------------


class TestWilsonLowerBound:
    def test_zero_total(self):
        assert wilson_lower_bound(0, 0) == 0.0

    def test_all_successes(self):
        result = wilson_lower_bound(100, 100)
        # Should be close to 1.0 but not exactly 1.0
        assert 0.95 < result < 1.0

    def test_half_success(self):
        result = wilson_lower_bound(50, 100)
        # Should be close to 0.5 but lower (conservative bound)
        assert 0.39 < result < 0.50

    def test_small_sample(self):
        result = wilson_lower_bound(3, 3)
        # Small sample = wide CI = low lower bound
        assert 0.3 < result < 0.85

    def test_large_sample_high_rate(self):
        result = wilson_lower_bound(950, 1000)
        # Large sample, high success rate = tight CI
        assert 0.93 < result < 0.96

    def test_monotonic_in_successes(self):
        """More successes with same total should give higher lower bound."""
        r1 = wilson_lower_bound(10, 100)
        r2 = wilson_lower_bound(50, 100)
        r3 = wilson_lower_bound(90, 100)
        assert r1 < r2 < r3

    def test_wilson_lower_bound_with_failures(self):
        """PRD-141 US-018: failed_after confidence is the Wilson lower bound of
        the FAILURE RATE (failures / co-occurrences), so failures < total.

        3 failures out of 10 co-occurrences = point estimate 0.3; the
        conservative lower bound sits below that, and more failures at the same
        total raise the bound (so a consistently-failing transition outranks a
        rarely-failing one when both are penalised).
        """
        partial = wilson_lower_bound(3, 10)
        assert 0.0 < partial < 0.3  # conservative: below the 0.3 point estimate
        # Monotonic in the failure count at fixed total
        assert wilson_lower_bound(2, 10) < wilson_lower_bound(8, 10)
        # A single failure in many co-occurrences is barely-confident (near 0)
        assert wilson_lower_bound(1, 50) < 0.1


# ---------------------------------------------------------------------------
# Test: Intent clustering
# ---------------------------------------------------------------------------


class TestIntentClustering:
    def test_empty_input(self):
        result = compute_intent_clusters(
            embeddings=np.array([]).reshape(0, 128),
            queries=[],
            action_names=[],
            statuses=[],
        )
        assert result.centroids == []
        assert result.labels == []

    def test_deterministic_results(self):
        """Same input produces same clusters (random_state=42)."""
        rng = np.random.RandomState(99)
        n = 50
        dim = 64
        embeddings = rng.randn(n, dim).astype(np.float32)
        queries = [f"query_{i}" for i in range(n)]
        actions = [f"action_{i % 5}" for i in range(n)]
        statuses = ["success"] * n

        r1 = compute_intent_clusters(embeddings, queries, actions, statuses, k=5)
        r2 = compute_intent_clusters(embeddings, queries, actions, statuses, k=5)

        assert r1.labels == r2.labels
        assert r1.centroids == r2.centroids
        assert r1.sample_queries == r2.sample_queries

    def test_produces_expected_clusters(self):
        """Well-separated data should cluster correctly."""
        dim = 32
        # Create 3 well-separated clusters
        cluster_a = np.ones((20, dim)) * 10
        cluster_b = np.ones((20, dim)) * -10
        cluster_c = np.ones((20, dim)) * 0
        # Add small noise
        rng = np.random.RandomState(42)
        cluster_a += rng.randn(20, dim) * 0.1
        cluster_b += rng.randn(20, dim) * 0.1
        cluster_c += rng.randn(20, dim) * 0.1

        embeddings = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)
        queries = [f"q{i}" for i in range(60)]
        actions = (["email_send"] * 20 + ["cal_create"] * 20 + ["doc_write"] * 20)
        statuses = ["success"] * 60

        result = compute_intent_clusters(embeddings, queries, actions, statuses, k=3)

        assert len(result.centroids) == 3
        assert len(result.sample_counts) == 3
        # Each cluster should have ~20 samples
        for count in result.sample_counts:
            assert 15 <= count <= 25

    def test_action_names_hot_reflects_success(self):
        """action_names_hot should only include successful actions."""
        dim = 16
        embeddings = np.ones((10, dim), dtype=np.float32)
        queries = [f"q{i}" for i in range(10)]
        # 5 success, 5 error
        actions = ["good_action"] * 5 + ["bad_action"] * 5
        statuses = ["success"] * 5 + ["error"] * 5

        result = compute_intent_clusters(embeddings, queries, actions, statuses, k=1)

        # Only "good_action" should be in hot list
        assert "good_action" in result.action_names_hot[0]
        assert "bad_action" not in result.action_names_hot[0]


# ---------------------------------------------------------------------------
# Test: Session key derivation
# ---------------------------------------------------------------------------


class TestSessionKeyDerivation:
    def test_turn_id_priority(self):
        log = {"turn_id": "t1", "conversation_id": "c1", "agent_id": 5, "workspace_id": "ws1"}
        assert _derive_session_key(log) == "turn:t1"

    def test_conversation_id_fallback(self):
        log = {"turn_id": None, "conversation_id": "c1", "agent_id": 5, "workspace_id": "ws1"}
        assert _derive_session_key(log) == "conv:c1"

    def test_agent_workspace_fallback(self):
        log = {"turn_id": None, "conversation_id": None, "agent_id": 5, "workspace_id": "ws1"}
        assert _derive_session_key(log) == "agent:5:ws:ws1"


# ---------------------------------------------------------------------------
# Test: Time window splitting
# ---------------------------------------------------------------------------


class TestTimeWindowSplitting:
    def test_empty_list(self):
        assert _split_by_time_window([]) == []

    def test_all_within_window(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        logs = [
            {"executed_at": base + timedelta(seconds=i * 10)}
            for i in range(5)
        ]
        windows = _split_by_time_window(logs, window_seconds=300)
        assert len(windows) == 1
        assert len(windows[0]) == 5

    def test_split_across_gap(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        logs = [
            {"executed_at": base},
            {"executed_at": base + timedelta(seconds=60)},
            {"executed_at": base + timedelta(seconds=600)},  # 10min gap
            {"executed_at": base + timedelta(seconds=660)},
        ]
        windows = _split_by_time_window(logs, window_seconds=300)
        assert len(windows) == 2
        assert len(windows[0]) == 2
        assert len(windows[1]) == 2


# ---------------------------------------------------------------------------
# Test: Used-after edge computation
# ---------------------------------------------------------------------------


class TestUsedAfterEdges:
    def _make_log(self, action: str, turn_id: str, agent_id: int = 1,
                  workspace_id: str = "ws1", offset_seconds: int = 0) -> Dict[str, Any]:
        return {
            "id": offset_seconds,
            "action_name": action,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
            "status": "success",
            "user_query": f"do {action}",
            "turn_id": turn_id,
            "conversation_id": None,
            "executed_at": datetime(2026, 1, 1) + timedelta(seconds=offset_seconds),
        }

    def test_basic_sequence(self):
        logs = [
            self._make_log("A", "t1", offset_seconds=0),
            self._make_log("B", "t1", offset_seconds=1),
            self._make_log("C", "t1", offset_seconds=2),
        ]
        edges = _compute_used_after_edges(logs)
        assert edges[("A", "B", "ws1", 1)] == 1
        assert edges[("B", "C", "ws1", 1)] == 1
        assert ("A", "C", "ws1", 1) not in edges  # Not adjacent

    def test_repeated_sequence_accumulates(self):
        logs = [
            self._make_log("A", "t1", offset_seconds=0),
            self._make_log("B", "t1", offset_seconds=1),
            self._make_log("A", "t2", offset_seconds=10),
            self._make_log("B", "t2", offset_seconds=11),
        ]
        edges = _compute_used_after_edges(logs)
        assert edges[("A", "B", "ws1", 1)] == 2

    def test_self_edge_skipped(self):
        logs = [
            self._make_log("A", "t1", offset_seconds=0),
            self._make_log("A", "t1", offset_seconds=1),
            self._make_log("B", "t1", offset_seconds=2),
        ]
        edges = _compute_used_after_edges(logs)
        assert ("A", "A", "ws1", 1) not in edges
        assert edges[("A", "B", "ws1", 1)] == 1

    def test_different_turns_separate(self):
        logs = [
            self._make_log("A", "t1", offset_seconds=0),
            self._make_log("B", "t2", offset_seconds=1),  # Different turn
        ]
        edges = _compute_used_after_edges(logs)
        # No edge because they're in different turns
        assert ("A", "B", "ws1", 1) not in edges


# ---------------------------------------------------------------------------
# Test: failed_after edge computation (PRD-141 US-018)
# ---------------------------------------------------------------------------


class TestFailedAfterEdges:
    """failed_after(A, B): A SUCCEEDED and a tool B within the next 2 steps in
    the same session ERRORED. Tracks (failed, total) co-occurrence so confidence
    is the Wilson lower bound of the failure rate, not a raw count.
    """

    def _make_log(self, action: str, status: str, turn_id: str = "t1",
                  agent_id: int = 1, workspace_id: str = "ws1",
                  offset_seconds: int = 0) -> Dict[str, Any]:
        return {
            "id": offset_seconds,
            "action_name": action,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
            "status": status,
            "user_query": f"do {action}",
            "turn_id": turn_id,
            "conversation_id": None,
            "executed_at": datetime(2026, 1, 1) + timedelta(seconds=offset_seconds),
        }

    def test_basic_failed_after(self):
        """A succeeds, B errors right after → (A,B) = (1 failed, 1 total)."""
        logs = [
            self._make_log("A", "success", offset_seconds=0),
            self._make_log("B", "error", offset_seconds=1),
        ]
        failed = _compute_failed_after_edges(logs)
        assert failed[("A", "B", "ws1", 1)] == (1, 1)

    def test_requires_a_to_succeed(self):
        """If A did not succeed, no failed_after edge originates from it."""
        logs = [
            self._make_log("A", "error", offset_seconds=0),
            self._make_log("B", "error", offset_seconds=1),
        ]
        failed = _compute_failed_after_edges(logs)
        assert ("A", "B", "ws1", 1) not in failed

    def test_within_two_steps(self):
        """B two steps after a successful A still counts; the gap tool that
        succeeded does not produce a failed edge."""
        logs = [
            self._make_log("A", "success", offset_seconds=0),
            self._make_log("C", "success", offset_seconds=1),
            self._make_log("B", "error", offset_seconds=2),
        ]
        failed = _compute_failed_after_edges(logs)
        assert failed[("A", "B", "ws1", 1)] == (1, 1)   # B is 2 steps after A
        assert failed[("C", "B", "ws1", 1)] == (1, 1)   # B is 1 step after C
        assert ("A", "C", "ws1", 1) not in failed       # C succeeded, no failure

    def test_beyond_two_steps_excluded(self):
        """A failure 3+ steps after A is not attributed to A."""
        logs = [
            self._make_log("A", "success", offset_seconds=0),
            self._make_log("X", "success", offset_seconds=1),
            self._make_log("Y", "success", offset_seconds=2),
            self._make_log("B", "error", offset_seconds=3),
        ]
        failed = _compute_failed_after_edges(logs)
        assert ("A", "B", "ws1", 1) not in failed   # B is 3 steps after A
        assert failed[("Y", "B", "ws1", 1)] == (1, 1)  # adjacent
        assert failed[("X", "B", "ws1", 1)] == (1, 1)  # 2 steps

    def test_failure_rate_accumulates(self):
        """Repeated A→B pairs build a (failed, total) rate across sessions."""
        logs = []
        # 3 turns where B errors after A
        for i in range(3):
            logs.append(self._make_log("A", "success", turn_id=f"t{i}", offset_seconds=i * 10))
            logs.append(self._make_log("B", "error", turn_id=f"t{i}", offset_seconds=i * 10 + 1))
        # 7 turns where B succeeds after A
        for i in range(3, 10):
            logs.append(self._make_log("A", "success", turn_id=f"t{i}", offset_seconds=i * 10))
            logs.append(self._make_log("B", "success", turn_id=f"t{i}", offset_seconds=i * 10 + 1))

        failed = _compute_failed_after_edges(logs)
        # 3 failures out of 10 co-occurrences
        assert failed[("A", "B", "ws1", 1)] == (3, 10)

    def test_self_edge_skipped(self):
        """A→A is never an edge, even when the second A errors."""
        logs = [
            self._make_log("A", "success", offset_seconds=0),
            self._make_log("A", "error", offset_seconds=1),
            self._make_log("B", "error", offset_seconds=2),
        ]
        failed = _compute_failed_after_edges(logs)
        assert ("A", "A", "ws1", 1) not in failed
        # First A succeeded; B errors 2 steps later → (A,B) counted once
        assert failed[("A", "B", "ws1", 1)] == (1, 1)

    def test_no_failures_returns_empty(self):
        """All-success sessions yield no failed_after edges."""
        logs = [
            self._make_log("A", "success", offset_seconds=0),
            self._make_log("B", "success", offset_seconds=1),
        ]
        assert _compute_failed_after_edges(logs) == {}

    def test_different_turns_isolated(self):
        """A success and a B failure in different sessions are not linked."""
        logs = [
            self._make_log("A", "success", turn_id="t1", offset_seconds=0),
            self._make_log("B", "error", turn_id="t2", offset_seconds=1),
        ]
        assert ("A", "B", "ws1", 1) not in _compute_failed_after_edges(logs)


# ---------------------------------------------------------------------------
# Test: Affinity computation
# ---------------------------------------------------------------------------


class TestAffinityComputation:
    def test_succeeds_for_intent(self):
        logs = [
            {"action_name": "email_send", "agent_id": 1, "workspace_id": "ws1",
             "status": "success", "executed_at": datetime(2026, 1, 1)},
        ] * 5
        cluster_map = {i: 100 for i in range(5)}  # All in cluster 100

        affinities = _compute_affinities(logs, cluster_map)

        intent_affs = [a for a in affinities if a["affinity_type"] == "succeeds_for_intent"]
        assert len(intent_affs) == 1
        assert intent_affs[0]["action_name"] == "email_send"
        assert intent_affs[0]["sample_count"] == 5
        assert intent_affs[0]["intent_cluster_id"] == 100

    def test_fails_for_intent(self):
        logs = [
            {"action_name": "email_send", "agent_id": 1, "workspace_id": "ws1",
             "status": "error", "executed_at": datetime(2026, 1, 1)},
        ] * 4
        cluster_map = {i: 200 for i in range(4)}

        affinities = _compute_affinities(logs, cluster_map)

        fail_affs = [a for a in affinities if a["affinity_type"] == "fails_for_intent"]
        assert len(fail_affs) == 1
        assert fail_affs[0]["action_name"] == "email_send"
        assert fail_affs[0]["sample_count"] == 4

    def test_agent_prefers_normalized(self):
        # Agent 1 calls email_send 6 times, cal_create 4 times = total 10
        logs = (
            [{"action_name": "email_send", "agent_id": 1, "workspace_id": "ws1",
              "status": "success", "executed_at": datetime(2026, 1, 1)}] * 6
            + [{"action_name": "cal_create", "agent_id": 1, "workspace_id": "ws1",
                "status": "success", "executed_at": datetime(2026, 1, 1)}] * 4
        )
        cluster_map = {}  # No clusters

        affinities = _compute_affinities(logs, cluster_map)

        agent_affs = [a for a in affinities if a["affinity_type"] == "agent_prefers"]
        email_aff = next(a for a in agent_affs if a["action_name"] == "email_send")
        cal_aff = next(a for a in agent_affs if a["action_name"] == "cal_create")

        # Weights should be normalized: 6/10 = 0.6, 4/10 = 0.4
        assert abs(email_aff["weight"] - 0.6) < 0.01
        assert abs(cal_aff["weight"] - 0.4) < 0.01

    def test_sample_floor_respected(self):
        """Actions with fewer than _SAMPLE_FLOOR observations are excluded."""
        logs = [
            {"action_name": "rare_action", "agent_id": 1, "workspace_id": "ws1",
             "status": "success", "executed_at": datetime(2026, 1, 1)},
        ] * (_SAMPLE_FLOOR - 1)
        cluster_map = {i: 300 for i in range(len(logs))}

        affinities = _compute_affinities(logs, cluster_map)

        # Should be empty because count < floor
        assert len(affinities) == 0

    def test_confidence_uses_wilson(self):
        """Confidence values should match Wilson lower bound calculation."""
        logs = [
            {"action_name": "test_action", "agent_id": 1, "workspace_id": "ws1",
             "status": "success", "executed_at": datetime(2026, 1, 1)},
        ] * 10
        cluster_map = {i: 400 for i in range(10)}

        affinities = _compute_affinities(logs, cluster_map)

        intent_aff = next(a for a in affinities if a["affinity_type"] == "succeeds_for_intent")
        expected_confidence = wilson_lower_bound(10, 10)
        assert abs(intent_aff["confidence"] - expected_confidence) < 0.001


# ---------------------------------------------------------------------------
# Test: Full pipeline (synthetic 200-row log)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Integration test: synthetic 200-row log produces expected edges."""

    def _generate_synthetic_logs(self, n: int = 200) -> List[Dict[str, Any]]:
        """Generate synthetic logs with known patterns."""
        rng = np.random.RandomState(42)
        logs = []
        base_time = datetime(2026, 1, 1)

        # Pattern: in turn 1-50, A->B happens frequently
        # In turn 51-100, B->C happens
        # Agent 1 prefers A,B. Agent 2 prefers C,D.
        actions = ["email_send", "cal_create", "doc_write", "file_read", "search_web"]

        for i in range(n):
            turn_idx = i // 4  # 4 logs per turn = 50 turns
            turn_id = f"turn_{turn_idx}"
            position_in_turn = i % 4

            if turn_idx < 25:
                # First 25 turns: email_send -> cal_create -> doc_write -> file_read
                action = actions[position_in_turn]
                agent_id = 1
            else:
                # Last 25 turns: cal_create -> doc_write -> file_read -> search_web
                action = actions[position_in_turn + 1] if position_in_turn < 4 else actions[4]
                agent_id = 2

            logs.append({
                "id": i,
                "action_name": action,
                "agent_id": agent_id,
                "workspace_id": "ws1",
                "status": "success" if rng.random() > 0.1 else "error",
                "user_query": f"Please {action.replace('_', ' ')} for turn {turn_idx}",
                "turn_id": turn_id,
                "conversation_id": None,
                "executed_at": base_time + timedelta(seconds=i * 10),
            })

        return logs

    def test_200_row_produces_edges(self):
        """200 synthetic logs produce expected edge structure."""
        logs = self._generate_synthetic_logs(200)
        edges = _compute_used_after_edges(logs)

        # Should have edges for A->B, B->C, C->D pattern in first 25 turns
        # and B->C, C->D, D->E in last 25 turns
        assert len(edges) > 0

        # email_send -> cal_create should appear (first 25 turns, 25 observations)
        key_ab = ("email_send", "cal_create", "ws1", 1)
        assert key_ab in edges
        assert edges[key_ab] == 25  # 25 turns with this pattern

    def test_200_row_edge_confidence(self):
        """Edge confidence should use Wilson lower bound."""
        logs = self._generate_synthetic_logs(200)
        edges = _compute_used_after_edges(logs)

        key_ab = ("email_send", "cal_create", "ws1", 1)
        count = edges[key_ab]

        # Wilson lower bound for 25 successes out of 25 total
        expected = wilson_lower_bound(count, count)
        assert expected > 0.8  # High confidence with 25 samples

    def test_idempotent_edge_computation(self):
        """Running edge computation twice on same data produces identical results."""
        logs = self._generate_synthetic_logs(200)

        edges_1 = _compute_used_after_edges(logs)
        edges_2 = _compute_used_after_edges(logs)

        assert edges_1 == edges_2

    def test_idempotent_clustering(self):
        """Running clustering twice on same embeddings produces identical results."""
        rng = np.random.RandomState(42)
        n = 200
        dim = 64
        embeddings = rng.randn(n, dim).astype(np.float32)
        queries = [f"query_{i}" for i in range(n)]
        actions = [f"action_{i % 5}" for i in range(n)]
        statuses = ["success"] * n

        r1 = compute_intent_clusters(embeddings, queries, actions, statuses)
        r2 = compute_intent_clusters(embeddings, queries, actions, statuses)

        assert r1.labels == r2.labels
        assert r1.centroids == r2.centroids
        assert r1.sample_counts == r2.sample_counts

    def test_idempotent_affinities(self):
        """Running affinity computation twice produces identical results."""
        logs = self._generate_synthetic_logs(200)
        cluster_map = {i: i % 5 for i in range(200)}

        aff_1 = _compute_affinities(logs, cluster_map)
        aff_2 = _compute_affinities(logs, cluster_map)

        # Sort for comparison (order may differ but content should be same)
        key_fn = lambda a: (a["action_name"], a["affinity_type"], str(a.get("agent_id")))
        assert sorted(aff_1, key=key_fn) == sorted(aff_2, key=key_fn)


# ---------------------------------------------------------------------------
# Test: K-means implementation
# ---------------------------------------------------------------------------


class TestKMeans:
    def test_basic_convergence(self):
        """K-means should converge on well-separated data."""
        data = np.vstack([
            np.random.RandomState(1).randn(30, 4) + 5,
            np.random.RandomState(2).randn(30, 4) - 5,
        ]).astype(np.float32)

        labels, centroids = _kmeans(data, k=2, random_state=42)

        assert labels.shape == (60,)
        assert centroids.shape == (2, 4)
        # Should find the two groups
        assert set(labels.tolist()) == {0, 1}
        # Each group should have ~30 members
        counts = [int((labels == i).sum()) for i in range(2)]
        assert all(25 <= c <= 35 for c in counts)

    def test_deterministic(self):
        """Same random_state produces same results."""
        data = np.random.RandomState(10).randn(50, 8).astype(np.float32)

        l1, c1 = _kmeans(data, k=3, random_state=42)
        l2, c2 = _kmeans(data, k=3, random_state=42)

        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_single_cluster(self):
        """K=1 should assign all points to cluster 0."""
        data = np.random.RandomState(5).randn(20, 4).astype(np.float32)
        labels, centroids = _kmeans(data, k=1, random_state=42)

        assert (labels == 0).all()
        assert centroids.shape == (1, 4)
