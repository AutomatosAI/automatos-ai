"""
Unit tests for PRD-139 US-006: seed_telemetry synthetic data generator.

Tests:
- Row generation counts (>= 2000 rows)
- Agent bias distribution per category
- Success/failure ratio (~80/20)
- Multi-action turn grouping produces used_after signal
- All rows tagged telemetry_source='synthetic'
- Idempotent seed (mock DB)
- Edge builder integration: synthetic data -> edges
"""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Import test targets
from scripts.eval.tool_routing.seed_telemetry import (
    AGENT_CATEGORY_BIAS,
    AGENT_OPS,
    AGENT_SCOUT,
    AGENT_SENTINEL,
    MULTI_ACTION_PAIRS,
    PAIR_PROBABILITY,
    REPETITIONS_PER_QUERY,
    SUCCESS_RATE,
    SYNTHETIC_AGENTS,
    SYNTHETIC_WORKSPACE_ID,
    TELEMETRY_SOURCE,
    _action_to_app_name,
    _select_agent,
    generate_synthetic_rows,
    seed_telemetry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_set() -> List[Dict[str, Any]]:
    """Load the real eval_set.jsonl for testing."""
    eval_path = Path(__file__).parent.parent / "scripts" / "eval" / "tool_routing" / "eval_set.jsonl"
    rows = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture
def synthetic_rows(eval_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate synthetic rows once for multiple tests."""
    return generate_synthetic_rows(eval_set)


# ---------------------------------------------------------------------------
# Generation count tests
# ---------------------------------------------------------------------------


class TestRowGeneration:
    """Test that row generation meets requirements."""

    def test_generates_minimum_2000_rows(self, synthetic_rows: List[Dict[str, Any]]):
        """AC: Generates at least 2000 rows."""
        assert len(synthetic_rows) >= 2000

    def test_all_eval_queries_represented(
        self, eval_set: List[Dict[str, Any]], synthetic_rows: List[Dict[str, Any]]
    ):
        """Every eval query appears in the synthetic data."""
        query_ids_expected = {e["query_id"] for e in eval_set}
        # Extract query_id from the turn_id pattern (query_id is the user_query match)
        queries_in_rows = {r["user_query"] for r in synthetic_rows}
        queries_expected = {e["query"] for e in eval_set}
        assert queries_expected == queries_in_rows

    def test_rows_per_query_roughly_correct(
        self, eval_set: List[Dict[str, Any]], synthetic_rows: List[Dict[str, Any]]
    ):
        """Each query has approximately REPETITIONS_PER_QUERY base rows (plus pairings)."""
        query_counts = Counter(r["user_query"] for r in synthetic_rows)
        for query_text in query_counts:
            count = query_counts[query_text]
            # Base repetitions + some pairing => at least REPETITIONS_PER_QUERY
            assert count >= REPETITIONS_PER_QUERY, (
                f"Query '{query_text[:40]}...' has only {count} rows, "
                f"expected >= {REPETITIONS_PER_QUERY}"
            )


# ---------------------------------------------------------------------------
# Agent bias tests
# ---------------------------------------------------------------------------


def _synthetic_agent(row: Dict[str, Any]) -> int:
    """Return the *synthetic* agent id (9001/9002/9003) for a generated row.

    ``_build_row`` resolves the synthetic id to a real production agent id for
    the ``agent_id`` column (see ``_resolve_agent_id``), but preserves the
    original synthetic id under ``router_decision['synthetic_agent_id']``. Bias
    assertions are written against the synthetic ids, so read it from there.
    """
    return row["router_decision"]["synthetic_agent_id"]


class TestAgentBias:
    """Test agent selection bias per category."""

    def test_three_synthetic_agents_used(self, synthetic_rows: List[Dict[str, Any]]):
        """All 3 synthetic agents appear in the data."""
        agents_seen = {_synthetic_agent(r) for r in synthetic_rows}
        assert agents_seen == set(SYNTHETIC_AGENTS)

    def test_sentinel_dominates_reports(self, synthetic_rows: List[Dict[str, Any]]):
        """Agent 9001 (Sentinel) should handle most report-related actions."""
        report_actions = {"platform_submit_report", "platform_get_latest_report"}
        report_rows = [r for r in synthetic_rows if r["action_name"] in report_actions]
        if not report_rows:
            pytest.skip("No report action rows generated")
        agent_counts = Counter(_synthetic_agent(r) for r in report_rows)
        # Sentinel should have the highest count for reports
        assert agent_counts[AGENT_SENTINEL] > agent_counts.get(AGENT_SCOUT, 0), (
            f"Sentinel ({agent_counts[AGENT_SENTINEL]}) should dominate reports, "
            f"but Scout has {agent_counts.get(AGENT_SCOUT, 0)}"
        )

    def test_scout_dominates_workspace(self, synthetic_rows: List[Dict[str, Any]]):
        """Agent 9002 (Scout) should handle most workspace actions."""
        ws_rows = [r for r in synthetic_rows if r["app_name"] == "WORKSPACE"]
        if not ws_rows:
            pytest.skip("No workspace action rows generated")
        agent_counts = Counter(_synthetic_agent(r) for r in ws_rows)
        assert agent_counts[AGENT_SCOUT] > agent_counts.get(AGENT_SENTINEL, 0), (
            f"Scout ({agent_counts[AGENT_SCOUT]}) should dominate workspace, "
            f"but Sentinel has {agent_counts.get(AGENT_SENTINEL, 0)}"
        )

    def test_ops_favors_agents_and_missions(self, synthetic_rows: List[Dict[str, Any]]):
        """Agent 9003 (Ops) should be well-represented in agents/missions."""
        ops_actions = {
            "platform_list_agents", "platform_get_agent", "platform_create_agent",
            "platform_list_missions", "platform_get_mission", "platform_create_mission",
        }
        target_rows = [r for r in synthetic_rows if r["action_name"] in ops_actions]
        if not target_rows:
            pytest.skip("No agents/missions rows generated")
        agent_counts = Counter(_synthetic_agent(r) for r in target_rows)
        # Ops should have significant presence (not necessarily highest for every action)
        total = len(target_rows)
        ops_share = agent_counts.get(AGENT_OPS, 0) / total
        assert ops_share > 0.25, (
            f"Ops should have >25% share for agents+missions, got {ops_share:.1%}"
        )


# ---------------------------------------------------------------------------
# Status distribution tests
# ---------------------------------------------------------------------------


class TestStatusDistribution:
    """Test success/failure ratio."""

    def test_success_rate_near_80_percent(self, synthetic_rows: List[Dict[str, Any]]):
        """~80% success, ~20% failure."""
        status_counts = Counter(r["status"] for r in synthetic_rows)
        total = len(synthetic_rows)
        success_rate = status_counts["success"] / total
        # Allow 5% tolerance
        assert 0.75 <= success_rate <= 0.85, (
            f"Success rate {success_rate:.1%} outside expected 75-85% range"
        )

    def test_both_statuses_present(self, synthetic_rows: List[Dict[str, Any]]):
        """Both 'success' and 'error' statuses appear."""
        statuses = {r["status"] for r in synthetic_rows}
        assert "success" in statuses
        assert "error" in statuses


# ---------------------------------------------------------------------------
# Telemetry source tagging
# ---------------------------------------------------------------------------


class TestTelemetrySourceTag:
    """All rows must be tagged for production exclusion."""

    def test_all_rows_tagged_synthetic(self, synthetic_rows: List[Dict[str, Any]]):
        """AC: All synthetic rows tagged with telemetry_source='synthetic'."""
        for row in synthetic_rows:
            assert row["telemetry_source"] == TELEMETRY_SOURCE, (
                f"Row for {row['action_name']} missing synthetic tag"
            )


# ---------------------------------------------------------------------------
# Multi-action turn tests (used_after signal)
# ---------------------------------------------------------------------------


class TestMultiActionTurns:
    """Test that multi-action turns create used_after edge signal."""

    def test_multi_action_turns_exist(self, synthetic_rows: List[Dict[str, Any]]):
        """Some turns should have multiple actions (for used_after edges)."""
        turn_ids = [r["router_decision"]["turn_id"] for r in synthetic_rows]
        turn_counts = Counter(turn_ids)
        multi_turns = [tid for tid, count in turn_counts.items() if count > 1]
        assert len(multi_turns) > 0, "No multi-action turns found"
        # With 47 queries * 42 reps * 0.45 pairing prob, expect significant count
        assert len(multi_turns) >= 50, (
            f"Only {len(multi_turns)} multi-action turns, expected >= 50"
        )

    def test_known_pairs_appear(self, synthetic_rows: List[Dict[str, Any]]):
        """Known multi-action pairs (e.g. read_file -> write_file) appear in same turns."""
        # Group rows by turn_id
        turns: Dict[str, List[str]] = defaultdict(list)
        for row in synthetic_rows:
            tid = row["router_decision"]["turn_id"]
            turns[tid].append(row["action_name"])

        # Check that at least some turns have a known pair
        found_pairs = set()
        for tid, actions in turns.items():
            if len(actions) >= 2:
                for primary, follow_ups in MULTI_ACTION_PAIRS.items():
                    if primary in actions:
                        for follow_up in follow_ups:
                            if follow_up in actions:
                                found_pairs.add((primary, follow_up))

        assert len(found_pairs) >= 3, (
            f"Expected >= 3 distinct action pairs, found {len(found_pairs)}: {found_pairs}"
        )

    def test_follow_up_timestamp_after_primary(self, synthetic_rows: List[Dict[str, Any]]):
        """In multi-action turns, follow-up timestamps are after primary."""
        turns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in synthetic_rows:
            tid = row["router_decision"]["turn_id"]
            turns[tid].append(row)

        for tid, turn_rows in turns.items():
            if len(turn_rows) >= 2:
                sorted_rows = sorted(turn_rows, key=lambda r: r["executed_at"])
                for i in range(len(sorted_rows) - 1):
                    assert sorted_rows[i]["executed_at"] <= sorted_rows[i + 1]["executed_at"]


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Test field validity and realistic-ness."""

    def test_app_name_matches_action_prefix(self, synthetic_rows: List[Dict[str, Any]]):
        """App name should be derived consistently from action prefix."""
        for row in synthetic_rows:
            expected = _action_to_app_name(row["action_name"])
            assert row["app_name"] == expected, (
                f"app_name mismatch: {row['app_name']} vs expected {expected} "
                f"for action {row['action_name']}"
            )

    def test_execution_time_positive(self, synthetic_rows: List[Dict[str, Any]]):
        """All execution times must be positive."""
        for row in synthetic_rows:
            assert row["execution_time_ms"] >= 10, (
                f"execution_time_ms={row['execution_time_ms']} too low"
            )

    def test_timestamps_within_window(self, synthetic_rows: List[Dict[str, Any]]):
        """All timestamps should be within the 30-day window."""
        now = datetime.utcnow()
        earliest = now - timedelta(days=31)
        for row in synthetic_rows:
            assert earliest <= row["executed_at"] <= now, (
                f"Timestamp {row['executed_at']} outside expected window"
            )

    def test_workspace_id_set(self, synthetic_rows: List[Dict[str, Any]]):
        """All rows have the synthetic workspace ID."""
        for row in synthetic_rows:
            assert row["workspace_id"] == SYNTHETIC_WORKSPACE_ID

    def test_router_decision_has_turn_id(self, synthetic_rows: List[Dict[str, Any]]):
        """Router decision JSONB contains turn_id for edge building."""
        for row in synthetic_rows:
            assert "turn_id" in row["router_decision"]
            assert len(row["router_decision"]["turn_id"]) > 0


# ---------------------------------------------------------------------------
# Idempotency test
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Test that seed_telemetry is idempotent."""

    def test_idempotent_clears_prior_synthetic(
        self, eval_set: List[Dict[str, Any]]
    ):
        """seed_telemetry deletes prior synthetic rows before inserting."""
        rows = generate_synthetic_rows(eval_set)

        # Mock the DB session
        mock_db = MagicMock(spec=["query", "add", "flush", "commit"])

        # Mock the query chain for DELETE
        mock_query_chain = MagicMock()
        mock_query_chain.filter.return_value.delete.return_value = 500  # "deleted" 500 prior rows
        mock_db.query.return_value = mock_query_chain

        try:
            inserted = seed_telemetry(mock_db, rows)
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(
                f"seed_telemetry import chain unavailable (core.models stubbed): {exc}"
            )

        # Verify DELETE was called (query -> filter -> delete)
        assert mock_db.query.called
        assert mock_query_chain.filter.called
        # Verify all rows were added
        assert mock_db.add.call_count == len(rows)
        assert inserted == len(rows)


# ---------------------------------------------------------------------------
# Edge builder integration test
# ---------------------------------------------------------------------------


class TestEdgePipelineIntegration:
    """Test that synthetic data, when passed through edge computation, produces edges.

    These tests import core.services.edge_builder which triggers the full
    database/config init chain.  When other test files stub sys.modules["config"]
    (e.g. test_platform_actions_section.py), the import will fail.  We skip
    gracefully in that case since the integration is proven when run in isolation.
    """

    def test_synthetic_data_produces_edges(self, synthetic_rows: List[Dict[str, Any]]):
        """Synthetic data should produce used_after edge signals.

        This tests the edge computation logic without a real DB by simulating
        what _compute_used_after_edges would see.
        """
        try:
            from core.services.edge_builder import _compute_used_after_edges
        except (ImportError, AttributeError) as exc:
            pytest.skip(f"edge_builder import chain unavailable: {exc}")

        # Convert synthetic rows to the format _compute_used_after_edges expects
        log_dicts = []
        for row in synthetic_rows:
            log_dicts.append({
                "id": hash(row["router_decision"]["turn_id"] + row["action_name"]),
                "agent_id": row["agent_id"],
                "workspace_id": str(row["workspace_id"]),
                "action_name": row["action_name"],
                "app_name": row["app_name"],
                "status": row["status"],
                "user_query": row["user_query"],
                "executed_at": row["executed_at"],
                "turn_id": row["router_decision"]["turn_id"],
                "conversation_id": None,
            })

        # Sort by executed_at (edge builder expects this)
        log_dicts.sort(key=lambda x: x["executed_at"])

        edge_data = _compute_used_after_edges(log_dicts)

        assert len(edge_data) > 0, "No edges computed from synthetic data"

        # Check that edges include known pairings
        edge_action_pairs = {(k[0], k[1]) for k in edge_data.keys()}
        assert len(edge_action_pairs) >= 3, (
            f"Expected >= 3 unique edge pairs, got {len(edge_action_pairs)}"
        )

    def test_all_affinity_types_represented(self, synthetic_rows: List[Dict[str, Any]]):
        """AC: All v1 affinity types should be computable from synthetic data.

        Checks that the data structure supports all 3 affinity types:
        - succeeds_for_intent (needs query + success)
        - fails_for_intent (needs query + failure)
        - agent_prefers (needs agent_id + action diversity)
        """
        has_queries = any(r.get("user_query") for r in synthetic_rows)
        has_successes = any(r["status"] == "success" for r in synthetic_rows)
        has_failures = any(r["status"] == "error" for r in synthetic_rows)
        has_agents = len({r["agent_id"] for r in synthetic_rows}) >= 2

        assert has_queries, "Need queries for intent affinities"
        assert has_successes, "Need successes for succeeds_for_intent"
        assert has_failures, "Need failures for fails_for_intent"
        assert has_agents, "Need multiple agents for agent_prefers"

    def test_edge_counts_above_sample_floor(self, synthetic_rows: List[Dict[str, Any]]):
        """Edges should meet the sample_floor threshold for the edge builder."""
        try:
            from core.services.edge_builder import _SAMPLE_FLOOR, _compute_used_after_edges
        except (ImportError, AttributeError) as exc:
            pytest.skip(f"edge_builder import chain unavailable: {exc}")

        log_dicts = []
        for row in synthetic_rows:
            log_dicts.append({
                "id": hash(row["router_decision"]["turn_id"] + row["action_name"]),
                "agent_id": row["agent_id"],
                "workspace_id": str(row["workspace_id"]),
                "action_name": row["action_name"],
                "app_name": row["app_name"],
                "status": row["status"],
                "user_query": row["user_query"],
                "executed_at": row["executed_at"],
                "turn_id": row["router_decision"]["turn_id"],
                "conversation_id": None,
            })

        log_dicts.sort(key=lambda x: x["executed_at"])
        edge_data = _compute_used_after_edges(log_dicts)

        # Count edges that meet the sample floor
        edges_above_floor = {k: v for k, v in edge_data.items() if v >= _SAMPLE_FLOOR}
        assert len(edges_above_floor) > 0, (
            f"No edges above sample_floor={_SAMPLE_FLOOR}. "
            f"Max edge count: {max(edge_data.values()) if edge_data else 0}"
        )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Test that generation is deterministic (same seed = same output)."""

    def test_deterministic_output(self, eval_set: List[Dict[str, Any]]):
        """Two runs with same seed and pinned base_time produce identical rows."""
        pinned_base = datetime(2026, 4, 1, 0, 0, 0)
        rows_a = generate_synthetic_rows(eval_set, base_time=pinned_base)
        rows_b = generate_synthetic_rows(eval_set, base_time=pinned_base)

        assert len(rows_a) == len(rows_b)
        for a, b in zip(rows_a, rows_b):
            assert a["agent_id"] == b["agent_id"]
            assert a["action_name"] == b["action_name"]
            assert a["status"] == b["status"]
            assert a["executed_at"] == b["executed_at"]
            assert a["router_decision"] == b["router_decision"]
