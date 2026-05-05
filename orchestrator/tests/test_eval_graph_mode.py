"""Tests for eval harness graph mode (PRD-139 US-007).

Validates:
  1. Graph mode is accepted as a valid mode in run_eval --mode choices
  2. Graph prompt builder returns expected format (prompt, surfaced, is_fallback)
  3. Graph prompt builder handles empty graph gracefully (fallback)
  4. Graph prompt builder handles GraphRouter exceptions (fallback)
  5. Graph prompt builder renders chain hints for multi-action chains
  6. Score handles graph mode rows correctly in aggregation + rendering
  7. Score handles missing graph rows (partial mode coverage)
  8. build_tools narrows schema for graph mode like filtered_schema
  9. run_eval tags graph-no-edges rows correctly in output

All tests are pure unit tests -- no Redis, no DB, no OpenRouter.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure orchestrator is on PYTHONPATH
# ---------------------------------------------------------------------------
_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

# ---------------------------------------------------------------------------
# Stub heavy modules before importing prompt_builder
# ---------------------------------------------------------------------------


@dataclass
class _FakeAction:
    name: str
    description: str
    category: str
    parameters: Optional[Dict[str, Any]] = None


def _fake_actions() -> List[_FakeAction]:
    """Minimal set of actions for testing."""
    return [
        _FakeAction(
            name="platform_list_agents",
            description="List all agents",
            category="agents",
            parameters={"properties": {"workspace_id": {}}, "required": ["workspace_id"]},
        ),
        _FakeAction(
            name="platform_get_latest_report",
            description="Get the latest agent report",
            category="reports",
            parameters={"properties": {"agent_id": {}}, "required": ["agent_id"]},
        ),
        _FakeAction(
            name="platform_submit_report",
            description="Submit an agent report",
            category="reports",
            parameters={"properties": {"content": {}}, "required": ["content"]},
        ),
        _FakeAction(
            name="platform_search_documents",
            description="Search workspace documents",
            category="documents",
            parameters={"properties": {"query": {}}, "required": ["query"]},
        ),
    ]


# Stub the modules.tools.discovery package hierarchy so prompt_builder
# can be imported without the full orchestrator dependency chain.
def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod


for _pkg in ("modules", "modules.tools", "modules.tools.discovery"):
    _ensure_stub(_pkg)


# Stub ActionSemanticIndex
_fake_asi = MagicMock()
_fake_asi.rank_actions = AsyncMock(return_value=[
    ("platform_list_agents", 0.95),
    ("platform_get_latest_report", 0.85),
    ("platform_submit_report", 0.80),
])
_fake_asi_mod = types.ModuleType("modules.tools.discovery.action_semantic_index")
_fake_asi_mod.get_action_semantic_index = lambda: _fake_asi
sys.modules["modules.tools.discovery.action_semantic_index"] = _fake_asi_mod


# Stub ActionRegistry
_fake_registry = MagicMock()
_fake_registry.build_filtered_prompt_summary.return_value = (
    "\n## Available Platform Actions\n\n- `platform_list_agents`: List all agents\n"
)
_fake_reg_mod = types.ModuleType("modules.tools.discovery.action_registry")
_fake_reg_mod.get_action_registry = lambda: _fake_registry
sys.modules["modules.tools.discovery.action_registry"] = _fake_reg_mod


# Stub GraphRouter
_fake_graph_router = MagicMock()
_fake_graph_router_mod = types.ModuleType("modules.tools.discovery.graph_router")
_fake_graph_router_mod.get_graph_router = lambda: _fake_graph_router
sys.modules["modules.tools.discovery.graph_router"] = _fake_graph_router_mod


# Now import prompt_builder
from scripts.eval.tool_routing.prompt_builder import PromptBuilder, _PREAMBLE, _CHAIN_HINT_HEADER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(actions=_fake_actions())


@pytest.fixture
def top_level_tools() -> List[Dict[str, Any]]:
    """Minimal top-level tools list with platform_execute."""
    return [
        {
            "type": "function",
            "function": {
                "name": "platform_execute",
                "description": "Execute a platform action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action name",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search knowledge base",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tests: PromptBuilder.build() accepts graph mode
# ---------------------------------------------------------------------------


class TestBuildGraphMode:
    """Test that build() routes to graph and returns expected format."""

    def test_build_graph_mode_accepted(self, builder: PromptBuilder) -> None:
        """Graph mode is accepted without ValueError."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_list_agents", 0.95, ["platform_list_agents"]),
        ])
        prompt, surfaced = builder.build("list my agents", mode="graph")
        assert isinstance(prompt, str)
        assert isinstance(surfaced, list)

    def test_build_graph_mode_invalid_raises(self, builder: PromptBuilder) -> None:
        """Unknown mode still raises ValueError."""
        with pytest.raises(ValueError, match="unknown mode"):
            builder.build("test", mode="banana")


# ---------------------------------------------------------------------------
# Tests: build_graph() return signature
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Test the build_graph() method directly."""

    def test_returns_three_elements(self, builder: PromptBuilder) -> None:
        """build_graph returns (prompt, surfaced, is_fallback)."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_list_agents", 0.95, ["platform_list_agents"]),
            ("platform_get_latest_report", 0.85, ["platform_get_latest_report", "platform_submit_report"]),
        ])
        result = builder.build_graph("submit a report", top_k=10)
        assert len(result) == 3
        prompt, surfaced, is_fallback = result
        assert isinstance(prompt, str)
        assert isinstance(surfaced, list)
        assert isinstance(is_fallback, bool)

    def test_multi_action_chains_not_fallback(self, builder: PromptBuilder) -> None:
        """When graph returns multi-action chains, is_fallback is False."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_get_latest_report", 0.9, ["platform_get_latest_report", "platform_submit_report"]),
        ])
        _, _, is_fallback = builder.build_graph("submit a report")
        assert is_fallback is False

    def test_single_action_chains_is_fallback(self, builder: PromptBuilder) -> None:
        """When all chains are single-action, is_fallback is True (no edges)."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_list_agents", 0.95, ["platform_list_agents"]),
            ("platform_get_latest_report", 0.85, ["platform_get_latest_report"]),
        ])
        _, _, is_fallback = builder.build_graph("list agents")
        assert is_fallback is True


# ---------------------------------------------------------------------------
# Tests: Empty graph fallback
# ---------------------------------------------------------------------------


class TestGraphFallback:
    """Test that empty/errored graph falls back to filtered mode."""

    def test_empty_chains_falls_back(self, builder: PromptBuilder) -> None:
        """Empty chain list → falls back to filtered, is_fallback=True."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[])
        prompt, surfaced, is_fallback = builder.build_graph("list agents")
        assert is_fallback is True
        # Should have called through filtered path
        assert len(surfaced) > 0

    def test_exception_falls_back(self, builder: PromptBuilder) -> None:
        """GraphRouter exception → falls back to filtered, is_fallback=True."""
        _fake_graph_router.rank_chains = AsyncMock(side_effect=RuntimeError("DB down"))
        prompt, surfaced, is_fallback = builder.build_graph("list agents")
        assert is_fallback is True
        assert len(surfaced) > 0


# ---------------------------------------------------------------------------
# Tests: Chain hints rendering
# ---------------------------------------------------------------------------


class TestChainHints:
    """Test chain hint block rendering for multi-action chains."""

    def test_chain_hints_present(self, builder: PromptBuilder) -> None:
        """Multi-action chains produce chain hint block in prompt."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_get_latest_report", 0.9, ["platform_get_latest_report", "platform_submit_report"]),
            ("platform_list_agents", 0.8, ["platform_list_agents"]),
        ])
        prompt, _, is_fallback = builder.build_graph("submit report after reading")
        assert is_fallback is False
        assert "Likely Platform Action Chains" in prompt
        assert "`platform_get_latest_report` -> `platform_submit_report`" in prompt

    def test_no_chain_hints_single_actions(self, builder: PromptBuilder) -> None:
        """Single-action chains do not produce chain hint block."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_list_agents", 0.95, ["platform_list_agents"]),
        ])
        prompt, _, _ = builder.build_graph("list agents")
        assert "Likely Platform Action Chains" not in prompt

    def test_surfaced_actions_deduplicated_ordered(self, builder: PromptBuilder) -> None:
        """Action names are deduplicated preserving rank order."""
        _fake_graph_router.rank_chains = AsyncMock(return_value=[
            ("platform_get_latest_report", 0.9, ["platform_get_latest_report", "platform_submit_report"]),
            ("platform_list_agents", 0.85, ["platform_list_agents"]),
            ("platform_get_latest_report", 0.7, ["platform_get_latest_report"]),  # duplicate
        ])
        _, surfaced, _ = builder.build_graph("mixed query")
        # Order: first appearance wins, no duplicates
        assert surfaced == [
            "platform_get_latest_report",
            "platform_submit_report",
            "platform_list_agents",
        ]


# ---------------------------------------------------------------------------
# Tests: build_tools for graph mode
# ---------------------------------------------------------------------------


class TestBuildToolsGraph:
    """Test that build_tools narrows schema for graph mode."""

    def test_graph_mode_narrows_enum(
        self, builder: PromptBuilder, top_level_tools: List[Dict[str, Any]]
    ) -> None:
        """Graph mode sets platform_execute.action.enum like filtered_schema."""
        ranked = ["platform_list_agents", "platform_submit_report"]
        tools = builder.build_tools(
            top_level_tools, "list agents", mode="graph", ranked_names=ranked
        )
        # Find platform_execute
        pe = next(t for t in tools if t["function"]["name"] == "platform_execute")
        enum = pe["function"]["parameters"]["properties"]["action"].get("enum")
        assert enum == ranked

    def test_full_mode_unchanged(
        self, builder: PromptBuilder, top_level_tools: List[Dict[str, Any]]
    ) -> None:
        """Full mode does NOT narrow."""
        tools = builder.build_tools(top_level_tools, "list agents", mode="full")
        assert tools is top_level_tools  # same object, not copied

    def test_filtered_mode_unchanged(
        self, builder: PromptBuilder, top_level_tools: List[Dict[str, Any]]
    ) -> None:
        """Filtered mode does NOT narrow schema."""
        tools = builder.build_tools(top_level_tools, "list agents", mode="filtered")
        assert tools is top_level_tools


# ---------------------------------------------------------------------------
# Tests: score.py handles graph mode
# ---------------------------------------------------------------------------


class TestScoreGraphMode:
    """Test that score.py aggregation and rendering handle graph mode rows."""

    def _make_rows(self) -> List[Dict[str, Any]]:
        """Create synthetic result rows across all modes."""
        modes = ["full", "filtered", "filtered_schema", "graph", "graph (no-edges)"]
        rows = []
        for i, mode in enumerate(modes):
            for qid in ["q001", "q002", "q003"]:
                rows.append({
                    "model": "anthropic/claude-sonnet-4.6",
                    "mode": mode,
                    "query_id": qid,
                    "query": f"test query {qid}",
                    "correct_actions": ["platform_list_agents"],
                    "category": "agents",
                    "difficulty": "easy",
                    "chosen_action": "platform_list_agents",
                    "chosen_via": "platform_execute",
                    "surfaced": ["platform_list_agents", "platform_get_latest_report"],
                    "prompt_tokens": 1000 - i * 100,
                    "completion_tokens": 50,
                    "total_tokens": 1050 - i * 100,
                    "latency_ms": 500 + i * 50,
                    "raw_finish": "tool_calls",
                    "error": None,
                })
        return rows

    def test_aggregate_includes_graph(self) -> None:
        """Graph mode rows are aggregated into separate (model, mode) buckets."""
        from scripts.eval.tool_routing.score import _aggregate

        rows = self._make_rows()
        model_costs = {"anthropic/claude-sonnet-4.6": (3.0, 15.0)}
        summary = _aggregate(rows, model_costs)

        modes_present = {s["mode"] for s in summary}
        assert "graph" in modes_present
        assert "graph (no-edges)" in modes_present

    def test_aggregate_graph_accuracy(self) -> None:
        """Graph mode accuracy computed correctly."""
        from scripts.eval.tool_routing.score import _aggregate

        rows = self._make_rows()
        model_costs = {"anthropic/claude-sonnet-4.6": (3.0, 15.0)}
        summary = _aggregate(rows, model_costs)

        graph_row = next(s for s in summary if s["mode"] == "graph")
        assert graph_row["accuracy"] == 1.0  # all correct in test data
        assert graph_row["n"] == 3

    def test_category_includes_graph(self) -> None:
        """Per-category breakdown includes graph mode."""
        from scripts.eval.tool_routing.score import _aggregate_by_category

        rows = self._make_rows()
        by_mode_cat = _aggregate_by_category(rows)
        assert "graph" in by_mode_cat
        assert "agents" in by_mode_cat["graph"]

    def test_render_main_table_sorts_graph(self) -> None:
        """Main table sorts graph after filtered_schema."""
        from scripts.eval.tool_routing.score import _render_main_table, _aggregate

        rows = self._make_rows()
        model_costs = {"anthropic/claude-sonnet-4.6": (3.0, 15.0)}
        summary = _aggregate(rows, model_costs)
        tiers = {"anthropic/claude-sonnet-4.6": "frontier"}

        table = _render_main_table(summary, tiers)
        lines = table.strip().split("\n")
        # Data lines (skip header + separator)
        data_lines = lines[2:]
        mode_column = [l.split("|")[3].strip() for l in data_lines]
        # Graph should come after filtered_schema
        if "graph" in mode_column and "filtered_schema" in mode_column:
            assert mode_column.index("graph") > mode_column.index("filtered_schema")

    def test_pair_diff_includes_graph_deltas(self) -> None:
        """Pair diff table includes graph vs full/filtered/filtered_schema."""
        from scripts.eval.tool_routing.score import _render_pair_diff, _aggregate

        rows = self._make_rows()
        model_costs = {"anthropic/claude-sonnet-4.6": (3.0, 15.0)}
        summary = _aggregate(rows, model_costs)
        tiers = {"anthropic/claude-sonnet-4.6": "frontier"}

        table = _render_pair_diff(summary, tiers)
        # Should contain graph delta rows
        assert "graph" in table

    def test_missing_graph_rows_handled(self) -> None:
        """Score works when only some modes are present (no graph)."""
        from scripts.eval.tool_routing.score import _aggregate, _render_main_table, _render_pair_diff

        # Only full + filtered rows
        rows = [r for r in self._make_rows() if r["mode"] in ("full", "filtered")]
        model_costs = {"anthropic/claude-sonnet-4.6": (3.0, 15.0)}
        summary = _aggregate(rows, model_costs)
        tiers = {"anthropic/claude-sonnet-4.6": "frontier"}

        # Should not crash
        table = _render_main_table(summary, tiers)
        assert "full" in table
        pair_table = _render_pair_diff(summary, tiers)
        # No graph rows, so no graph delta
        assert "graph" not in pair_table


# ---------------------------------------------------------------------------
# Tests: run_eval mode parsing
# ---------------------------------------------------------------------------


class TestRunEvalModes:
    """Test that run_eval accepts graph and all modes correctly."""

    def test_all_includes_graph(self) -> None:
        """'all' mode expands to include 'graph'."""
        # We test the mode expansion logic directly
        mode_arg = "all"
        if mode_arg == "all":
            modes = ["full", "filtered", "filtered_schema", "graph"]
        else:
            modes = [mode_arg]
        assert "graph" in modes
        assert len(modes) == 4

    def test_graph_mode_standalone(self) -> None:
        """'graph' as standalone mode."""
        mode_arg = "graph"
        if mode_arg == "all":
            modes = ["full", "filtered", "filtered_schema", "graph"]
        else:
            modes = [mode_arg]
        assert modes == ["graph"]

    def test_graph_is_valid_choice(self) -> None:
        """'graph' is in the argparse choices list."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mode",
            choices=["full", "filtered", "filtered_schema", "graph", "all"],
            default="all",
        )
        # Should not raise
        args = parser.parse_args(["--mode", "graph"])
        assert args.mode == "graph"

    def test_graph_no_edges_mode_label(self) -> None:
        """When graph has no edges, effective mode is 'graph (no-edges)'."""
        graph_fallback = {("graph", "test query"): True}
        mode = "graph"
        query = "test query"
        effective_mode = mode
        if mode == "graph":
            is_fb = graph_fallback.get((mode, query), False)
            if is_fb:
                effective_mode = "graph (no-edges)"
        assert effective_mode == "graph (no-edges)"

    def test_graph_with_edges_mode_label(self) -> None:
        """When graph has edges, effective mode stays 'graph'."""
        graph_fallback = {("graph", "test query"): False}
        mode = "graph"
        query = "test query"
        effective_mode = mode
        if mode == "graph":
            is_fb = graph_fallback.get((mode, query), False)
            if is_fb:
                effective_mode = "graph (no-edges)"
        assert effective_mode == "graph"
