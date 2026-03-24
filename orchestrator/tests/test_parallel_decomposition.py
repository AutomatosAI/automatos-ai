"""
Wiring tests for US-005: Parallel decomposition in planner.
=============================================================

Proves:
1. _parse_plan extracts complexity and parallel_group from task dicts
2. _validate_plan rejects parallel_group tasks with cross-dependencies
3. Token estimate uses COMPLEXITY_TOKEN_BUDGET (not flat 2000 * task_count)
4. decompose() with complex goal produces parallel_group tasks and synthesis
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.coordination.planner import (
    DecompositionResult,
    MissionPlanner,
    PlannedDependency,
    PlannedTask,
    _estimate_token_budget,
    _parse_plan,
    _validate_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(role: str) -> MagicMock:
    agent = MagicMock()
    agent.id = 1
    agent.name = role
    agent.status = "active"
    agent.skills = []
    agent.tags = []
    return agent


# ---------------------------------------------------------------------------
# Unit: _parse_plan extracts complexity and parallel_group
# ---------------------------------------------------------------------------

class TestParsePlanParallelFields:
    def test_extracts_complexity_and_parallel_group(self):
        raw = {
            "tasks": [
                {
                    "temp_id": "task_1",
                    "title": "Research AI",
                    "description": "Research AI topics",
                    "agent_role": "researcher",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "complexity": "complex",
                    "parallel_group": "research",
                    "dependencies": [],
                },
                {
                    "temp_id": "task_2",
                    "title": "Research ML",
                    "description": "Research ML topics",
                    "agent_role": "researcher",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "complexity": "simple",
                    "parallel_group": "research",
                    "dependencies": [],
                },
            ]
        }
        errors = []
        tasks, deps = _parse_plan(raw, errors)
        assert not errors
        assert len(tasks) == 2
        assert tasks[0].complexity == "complex"
        assert tasks[0].parallel_group == "research"
        assert tasks[1].complexity == "simple"
        assert tasks[1].parallel_group == "research"

    def test_defaults_complexity_to_moderate(self):
        raw = {
            "tasks": [
                {
                    "temp_id": "task_1",
                    "title": "Do something",
                    "description": "desc",
                    "agent_role": "writer",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "dependencies": [],
                },
            ]
        }
        errors = []
        tasks, _ = _parse_plan(raw, errors)
        assert not errors
        assert tasks[0].complexity == "moderate"
        assert tasks[0].parallel_group is None

    def test_invalid_complexity_defaults_to_moderate(self):
        raw = {
            "tasks": [
                {
                    "temp_id": "task_1",
                    "title": "Do something",
                    "description": "desc",
                    "agent_role": "writer",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "complexity": "ultra_hard",
                    "dependencies": [],
                },
            ]
        }
        errors = []
        tasks, _ = _parse_plan(raw, errors)
        assert not errors
        assert tasks[0].complexity == "moderate"


# ---------------------------------------------------------------------------
# Unit: _validate_plan rejects parallel_group cross-dependencies
# ---------------------------------------------------------------------------

class TestValidatePlanParallelGroup:
    def test_rejects_cross_dependent_parallel_group(self):
        """Tasks in same parallel_group with a dependency between them → error."""
        tasks = [
            PlannedTask(
                temp_id="task_1", title="Research A", description="...",
                agent_role="researcher", sequence_number=1,
                task_type="llm_generation", verification_criteria=[],
                required_tools=[], dependencies=[],
                complexity="moderate", parallel_group="research",
            ),
            PlannedTask(
                temp_id="task_2", title="Research B", description="...",
                agent_role="researcher", sequence_number=1,
                task_type="llm_generation", verification_criteria=[],
                required_tools=[], dependencies=["task_1"],
                complexity="moderate", parallel_group="research",
            ),
            PlannedTask(
                temp_id="task_3", title="Synthesize", description="...",
                agent_role="writer", sequence_number=2,
                task_type="synthesis", verification_criteria=[],
                required_tools=[], dependencies=["task_1", "task_2"],
                complexity="moderate",
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_2"),
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]
        agents = [_make_agent("researcher"), _make_agent("writer")]
        errors = _validate_plan(tasks, deps, agents)
        assert any("parallel_group" in e and "cross-dependency" in e for e in errors)

    def test_accepts_valid_parallel_group(self):
        """Tasks in same parallel_group with NO dependency between them → OK."""
        tasks = [
            PlannedTask(
                temp_id="task_1", title="Research A", description="...",
                agent_role="researcher", sequence_number=1,
                task_type="llm_generation", verification_criteria=[],
                required_tools=[], dependencies=[],
                complexity="moderate", parallel_group="research",
            ),
            PlannedTask(
                temp_id="task_2", title="Research B", description="...",
                agent_role="researcher", sequence_number=1,
                task_type="llm_generation", verification_criteria=[],
                required_tools=[], dependencies=[],
                complexity="moderate", parallel_group="research",
            ),
            PlannedTask(
                temp_id="task_3", title="Synthesize", description="...",
                agent_role="writer", sequence_number=2,
                task_type="synthesis", verification_criteria=[],
                required_tools=[], dependencies=["task_1", "task_2"],
                complexity="moderate",
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]
        agents = [_make_agent("researcher"), _make_agent("writer")]
        errors = _validate_plan(tasks, deps, agents)
        assert not any("parallel_group" in e for e in errors)


# ---------------------------------------------------------------------------
# Unit: Token estimate uses COMPLEXITY_TOKEN_BUDGET
# ---------------------------------------------------------------------------

class TestTokenEstimate:
    def test_not_flat_estimate(self):
        """Token estimate is NOT flat 2000 * task_count — uses complexity."""
        tasks = [
            PlannedTask(
                temp_id="t1", title="A", description="", agent_role="r",
                sequence_number=1, task_type="llm_generation",
                verification_criteria=[], required_tools=[], dependencies=[],
                complexity="simple",
            ),
            PlannedTask(
                temp_id="t2", title="B", description="", agent_role="r",
                sequence_number=2, task_type="llm_generation",
                verification_criteria=[], required_tools=[], dependencies=["t1"],
                complexity="complex",
            ),
            PlannedTask(
                temp_id="t3", title="C", description="", agent_role="r",
                sequence_number=3, task_type="synthesis",
                verification_criteria=[], required_tools=[], dependencies=["t2"],
                complexity="moderate",
            ),
        ]
        estimate = _estimate_token_budget(tasks)
        flat_estimate = 2000 * len(tasks)
        # simple(1000) + complex(8000) + moderate(4000) = 13000
        assert estimate != flat_estimate
        assert estimate == 1000 + 8000 + 4000


# ---------------------------------------------------------------------------
# Wiring: decompose() with complex goal produces parallel groups + synthesis
# ---------------------------------------------------------------------------

class TestDecomposeParallelPlan:
    @pytest.mark.asyncio
    async def test_complex_goal_produces_parallel_and_synthesis(self):
        """
        WIRING TEST: Decompose a complex research paper goal — result contains
        at least one parallel_group with 2+ tasks and at least one synthesis task.
        """
        goal = (
            "Write a 4000 word research paper with 6 sections covering "
            "AI coordination, prior art, experiments, and implications"
        )

        # LLM returns a plan with parallel groups and synthesis
        fake_plan = {
            "tasks": [
                {
                    "temp_id": "task_1",
                    "title": "Research AI coordination",
                    "description": "Research AI coordination concepts",
                    "agent_role": "researcher",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "complexity": "complex",
                    "parallel_group": "research",
                    "dependencies": [],
                },
                {
                    "temp_id": "task_2",
                    "title": "Research prior art",
                    "description": "Survey prior art literature",
                    "agent_role": "researcher",
                    "sequence_number": 1,
                    "task_type": "llm_generation",
                    "complexity": "complex",
                    "parallel_group": "research",
                    "dependencies": [],
                },
                {
                    "temp_id": "task_3",
                    "title": "Synthesize research findings",
                    "description": "Merge all research outputs",
                    "agent_role": "writer",
                    "sequence_number": 2,
                    "task_type": "synthesis",
                    "complexity": "moderate",
                    "dependencies": ["task_1", "task_2"],
                },
                {
                    "temp_id": "task_4",
                    "title": "Draft paper",
                    "description": "Write the full paper",
                    "agent_role": "writer",
                    "sequence_number": 3,
                    "task_type": "llm_generation",
                    "complexity": "complex",
                    "dependencies": ["task_3"],
                },
                {
                    "temp_id": "task_5",
                    "title": "Review paper",
                    "description": "Review and refine",
                    "agent_role": "researcher",
                    "sequence_number": 4,
                    "task_type": "review",
                    "complexity": "moderate",
                    "dependencies": ["task_4"],
                },
            ]
        }

        agents = [_make_agent("researcher"), _make_agent("writer")]

        with patch(
            "modules.coordination.planner.match_template", return_value=None
        ), patch(
            "modules.coordination.planner.create_llm_manager"
        ) as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.generate_response = AsyncMock(
                return_value=MagicMock(content=json.dumps(fake_plan))
            )
            mock_llm_factory.return_value = mock_llm

            result = await MissionPlanner.decompose(
                goal=goal,
                workspace_id=uuid4(),
                agents=agents,
            )

        assert isinstance(result, DecompositionResult)

        # At least one parallel_group with 2+ tasks
        group_counts: dict[str, int] = {}
        for t in result.tasks:
            if t.parallel_group:
                group_counts[t.parallel_group] = group_counts.get(t.parallel_group, 0) + 1
        assert any(count >= 2 for count in group_counts.values()), (
            f"Expected a parallel_group with 2+ tasks, got: {group_counts}"
        )

        # At least one synthesis task
        synthesis_tasks = [t for t in result.tasks if t.task_type == "synthesis"]
        assert len(synthesis_tasks) >= 1, "Expected at least one synthesis task"

        # Token estimate is complexity-based, not flat
        flat_estimate = 2000 * len(result.tasks)
        assert result.token_estimate != flat_estimate
