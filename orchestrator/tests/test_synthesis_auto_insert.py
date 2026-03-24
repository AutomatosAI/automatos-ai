"""
Wiring tests for US-008: Synthesis auto-insertion in planner.
==============================================================

Proves:
1. Plan with parallel group (2 tasks) and downstream non-synthesis task
   — auto-inserts synthesis between them
2. Plan with explicit synthesis task after parallel group — no auto-insertion
3. Auto-inserted synthesis task has correct fields (agent_role, task_type,
   complexity, dependencies)
4. Downstream task dependency repointed from group members to synth task
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from core.models.orchestration_enums import TaskType
from modules.coordination.planner import (
    PlannedDependency,
    PlannedTask,
    _ensure_synthesis_tasks,
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


def _task(
    temp_id: str,
    title: str = "Test task",
    agent_role: str = "researcher",
    seq: int = 1,
    task_type: str = "llm_generation",
    deps: list | None = None,
    complexity: str = "moderate",
    parallel_group: str | None = None,
) -> PlannedTask:
    return PlannedTask(
        temp_id=temp_id,
        title=title,
        description=f"Description for {title}",
        agent_role=agent_role,
        sequence_number=seq,
        task_type=task_type,
        verification_criteria=[],
        required_tools=[],
        dependencies=deps or [],
        complexity=complexity,
        parallel_group=parallel_group,
    )


# ---------------------------------------------------------------------------
# Test: Auto-insert synthesis when parallel group converges without one
# ---------------------------------------------------------------------------

class TestEnsureSynthesisAutoInsert:
    """Plan with parallel group 'research' (2 tasks) and downstream task_3
    depending on both but not synthesis — auto-inserts synthesis between."""

    def test_inserts_synthesis_between_parallel_group_and_downstream(self):
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Research ML", seq=1, parallel_group="research"),
            _task(
                "task_3", "Write report", agent_role="writer", seq=2,
                deps=["task_1", "task_2"],
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)

        # Should have 4 tasks now (original 3 + 1 auto-inserted synth)
        assert len(new_tasks) == 4

        synth_tasks = [t for t in new_tasks if t.task_type == TaskType.SYNTHESIS.value]
        assert len(synth_tasks) == 1

        synth = synth_tasks[0]
        assert synth.temp_id == "synth_research"
        assert synth.agent_role == "writer"
        assert synth.complexity == "synthesis"
        assert set(synth.dependencies) == {"task_1", "task_2"}
        assert synth.parallel_group is None

    def test_downstream_task_repointed_to_synth(self):
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Research ML", seq=1, parallel_group="research"),
            _task(
                "task_3", "Write report", agent_role="writer", seq=2,
                deps=["task_1", "task_2"],
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)

        # task_3 should now depend on synth_research, not task_1/task_2
        task_3 = next(t for t in new_tasks if t.temp_id == "task_3")
        assert task_3.dependencies == ["synth_research"]

        # Deps should show: task_1 → synth, task_2 → synth, synth → task_3
        dep_pairs = {(d.from_task_temp_id, d.to_task_temp_id) for d in new_deps}
        assert ("task_1", "synth_research") in dep_pairs
        assert ("task_2", "synth_research") in dep_pairs
        assert ("synth_research", "task_3") in dep_pairs
        # Old direct deps should be gone
        assert ("task_1", "task_3") not in dep_pairs
        assert ("task_2", "task_3") not in dep_pairs

    def test_synth_sequence_after_parallel_group(self):
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Research ML", seq=1, parallel_group="research"),
            _task(
                "task_3", "Write report", agent_role="writer", seq=2,
                deps=["task_1", "task_2"],
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]

        new_tasks, _ = _ensure_synthesis_tasks(tasks, deps)
        synth = next(t for t in new_tasks if t.temp_id == "synth_research")
        # Parallel group has seq=1, so synth should be seq=2
        assert synth.sequence_number == 2


# ---------------------------------------------------------------------------
# Test: No auto-insertion when explicit synthesis already exists
# ---------------------------------------------------------------------------

class TestNoInsertionWhenSynthesisExists:
    """Plan with explicit synthesis task after parallel group — no auto-insertion."""

    def test_skips_group_with_explicit_synthesis(self):
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Research ML", seq=1, parallel_group="research"),
            _task(
                "task_synth", "Merge research", agent_role="writer", seq=2,
                task_type=TaskType.SYNTHESIS.value,
                deps=["task_1", "task_2"],
            ),
            _task(
                "task_4", "Write report", agent_role="writer", seq=3,
                deps=["task_synth"],
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_synth"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_synth"),
            PlannedDependency(from_task_temp_id="task_synth", to_task_temp_id="task_4"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)

        # No new tasks should be added
        assert len(new_tasks) == 4
        assert len(new_deps) == 3

    def test_no_change_for_single_member_groups(self):
        """A parallel_group with only 1 member should not trigger insertion."""
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Write report", agent_role="writer", seq=2, deps=["task_1"]),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_2"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)
        assert len(new_tasks) == 2
        assert len(new_deps) == 1


# ---------------------------------------------------------------------------
# Test: No parallel groups at all — passthrough
# ---------------------------------------------------------------------------

class TestNoParallelGroups:
    def test_sequential_plan_unchanged(self):
        tasks = [
            _task("task_1", "Step 1", seq=1),
            _task("task_2", "Step 2", seq=2, deps=["task_1"]),
            _task("task_3", "Step 3", seq=3, deps=["task_2"]),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_2"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)
        assert len(new_tasks) == 3
        assert len(new_deps) == 2


# ---------------------------------------------------------------------------
# Test: Result passes validation
# ---------------------------------------------------------------------------

class TestAutoInsertedPlanPassesValidation:
    """Ensure auto-inserted plans don't break the validator."""

    def test_auto_inserted_plan_valid(self):
        tasks = [
            _task("task_1", "Research AI", seq=1, parallel_group="research"),
            _task("task_2", "Research ML", seq=1, parallel_group="research"),
            _task(
                "task_3", "Write report", agent_role="writer", seq=2,
                deps=["task_1", "task_2"],
            ),
        ]
        deps = [
            PlannedDependency(from_task_temp_id="task_1", to_task_temp_id="task_3"),
            PlannedDependency(from_task_temp_id="task_2", to_task_temp_id="task_3"),
        ]

        new_tasks, new_deps = _ensure_synthesis_tasks(tasks, deps)

        agents = [_make_agent("researcher"), _make_agent("writer")]
        errors = _validate_plan(new_tasks, new_deps, agents)
        assert errors == [], f"Validation errors: {errors}"
