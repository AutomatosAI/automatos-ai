"""
Wiring tests for US-007: Synthesis task executor.
==================================================

Proves:
1. _collect_upstream_outputs fetches dependency task outputs
2. _build_synthesis_prompt includes all upstream outputs
3. _auto_synthesis_verification_criteria generates required_sections + min_length
4. _prepare_task detects TaskType.SYNTHESIS and uses synthesis prompt
"""
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from core.models.orchestration_enums import TaskType
from services.coordinator_service import CoordinatorService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    *,
    task_id=None,
    title: str = "Test Task",
    description: str = "A test task",
    task_type: str = "llm_generation",
    output: str = "",
    sequence_number: int = 1,
    verification_criteria=None,
) -> MagicMock:
    task = MagicMock()
    task.id = task_id or uuid4()
    task.title = title
    task.description = description
    task.task_type = task_type
    task.output = output
    task.sequence_number = sequence_number
    task.verification_criteria = verification_criteria
    task.input_context = None
    return task


def _make_dep(task_id, depends_on_task_id) -> MagicMock:
    dep = MagicMock()
    dep.task_id = task_id
    dep.depends_on_task_id = depends_on_task_id
    return dep


def _mock_db_for_upstream(
    synthesis_task_id,
    upstream_tasks: List[MagicMock],
) -> MagicMock:
    """Create a mock DB session that returns upstream deps + tasks."""
    db = MagicMock()

    deps = [
        _make_dep(synthesis_task_id, ut.id) for ut in upstream_tasks
    ]

    # Chain: db.query(Dep).filter(...).all() → deps
    # Chain: db.query(Task).filter(...).order_by(...).all() → upstream_tasks
    dep_query = MagicMock()
    dep_query.filter.return_value = dep_query
    dep_query.all.return_value = deps

    task_query = MagicMock()
    task_query.filter.return_value = task_query
    task_query.order_by.return_value = task_query
    task_query.all.return_value = upstream_tasks

    call_count = [0]

    def side_effect(model):
        call_count[0] += 1
        # First query call is for deps, second for tasks
        if call_count[0] % 2 == 1:
            return dep_query
        return task_query

    db.query.side_effect = side_effect
    return db


# ---------------------------------------------------------------------------
# Test: _collect_upstream_outputs
# ---------------------------------------------------------------------------

class TestCollectUpstreamOutputs:
    def test_returns_upstream_task_outputs(self):
        task_a = _make_task(
            title="Research AI", output="AI findings here", sequence_number=1,
        )
        task_b = _make_task(
            title="Research ML", output="ML findings here", sequence_number=2,
        )
        synthesis = _make_task(
            title="Synthesise", task_type=TaskType.SYNTHESIS.value,
        )

        db = _mock_db_for_upstream(synthesis.id, [task_a, task_b])
        result = CoordinatorService._collect_upstream_outputs(db, synthesis)

        assert len(result) == 2
        assert result[0]["title"] == "Research AI"
        assert result[0]["output"] == "AI findings here"
        assert result[1]["title"] == "Research ML"
        assert result[1]["output"] == "ML findings here"

    def test_returns_empty_when_no_deps(self):
        synthesis = _make_task(title="Synthesise")

        db = MagicMock()
        dep_query = MagicMock()
        dep_query.filter.return_value = dep_query
        dep_query.all.return_value = []
        db.query.return_value = dep_query

        result = CoordinatorService._collect_upstream_outputs(db, synthesis)
        assert result == []


# ---------------------------------------------------------------------------
# Test: _build_synthesis_prompt contains upstream outputs
# ---------------------------------------------------------------------------

class TestBuildSynthesisPrompt:
    def test_contains_all_upstream_outputs(self):
        task = _make_task(
            title="Merge Research",
            description="Combine AI and ML research",
            task_type=TaskType.SYNTHESIS.value,
        )
        upstream = [
            {"title": "Research AI", "description": "", "output": "AI is transformative"},
            {"title": "Research ML", "description": "", "output": "ML enables automation"},
        ]

        prompt = CoordinatorService._build_synthesis_prompt(task, upstream)

        assert "Synthesis Task: Merge Research" in prompt
        assert "Combine AI and ML research" in prompt
        assert "Research AI" in prompt
        assert "AI is transformative" in prompt
        assert "Research ML" in prompt
        assert "ML enables automation" in prompt
        assert "unified voice" in prompt

    def test_handles_no_upstream(self):
        task = _make_task(
            title="Synthesise",
            task_type=TaskType.SYNTHESIS.value,
        )

        prompt = CoordinatorService._build_synthesis_prompt(task, [])
        assert "No upstream outputs available" in prompt


# ---------------------------------------------------------------------------
# Test: _auto_synthesis_verification_criteria
# ---------------------------------------------------------------------------

class TestAutoSynthesisVerificationCriteria:
    def test_generates_required_sections_from_titles(self):
        task = _make_task(task_type=TaskType.SYNTHESIS.value)
        upstream = [
            {"title": "Research AI", "output": "x" * 1000},
            {"title": "Research ML", "output": "y" * 1000},
        ]

        criteria = CoordinatorService._auto_synthesis_verification_criteria(
            task, upstream,
        )

        section_check = next(
            c for c in criteria if c["type"] == "required_sections"
        )
        assert "Research AI" in section_check["value"]
        assert "Research ML" in section_check["value"]

    def test_min_length_is_half_combined(self):
        task = _make_task(task_type=TaskType.SYNTHESIS.value)
        upstream = [
            {"title": "A", "output": "x" * 2000},
            {"title": "B", "output": "y" * 2000},
        ]

        criteria = CoordinatorService._auto_synthesis_verification_criteria(
            task, upstream,
        )

        length_check = next(c for c in criteria if c["type"] == "min_length")
        # 50% of 4000 = 2000
        assert length_check["value"] == 2000

    def test_min_length_floor_at_200(self):
        task = _make_task(task_type=TaskType.SYNTHESIS.value)
        upstream = [
            {"title": "A", "output": "short"},
        ]

        criteria = CoordinatorService._auto_synthesis_verification_criteria(
            task, upstream,
        )

        length_check = next(c for c in criteria if c["type"] == "min_length")
        assert length_check["value"] == 200


# ---------------------------------------------------------------------------
# Test: _prepare_task uses synthesis prompt for SYNTHESIS tasks
# ---------------------------------------------------------------------------

class TestPrepareTaskSynthesisWiring:
    @pytest.mark.asyncio
    async def test_synthesis_task_uses_synthesis_prompt(self):
        """Verify that _prepare_task builds synthesis prompt for SYNTHESIS tasks."""
        # Mock AgentFactory before it gets imported inside _prepare_task
        mock_agent_factory_cls = MagicMock()
        mock_factory_instance = MagicMock()
        mock_factory_instance.active_agents = {}

        async def mock_activate(agent_id, workspace_dir):
            runtime = MagicMock()
            runtime.llm_manager.config.max_tokens = 2000
            return runtime

        mock_factory_instance.activate_agent = mock_activate
        mock_agent_factory_cls.return_value = mock_factory_instance

        # Create a mock module for the lazy import
        mock_factory_module = MagicMock()
        mock_factory_module.AgentFactory = mock_agent_factory_cls

        service = CoordinatorService.__new__(CoordinatorService)
        service._field = None  # Skip lazy init

        task_a = _make_task(
            title="Research AI", output="AI findings", sequence_number=1,
        )
        task_b = _make_task(
            title="Research ML", output="ML findings", sequence_number=2,
        )

        synthesis_task = _make_task(
            title="Merge Research",
            description="Combine all research",
            task_type=TaskType.SYNTHESIS.value,
            verification_criteria=None,
        )
        synthesis_task.state = "assigned"

        run = MagicMock()
        run.id = uuid4()
        run.config = {}
        run.tokens_used = 0
        run.token_budget_estimate = 100000

        db = _mock_db_for_upstream(synthesis_task.id, [task_a, task_b])

        with patch(
            "services.coordinator_service.MissionDispatcher"
        ) as mock_dispatcher, patch.dict(
            "sys.modules",
            {"modules.agents.factory.agent_factory": mock_factory_module},
        ), patch.object(
            service, "_inject_task_output_into_field", return_value=None,
        ):
            mock_dispatcher.record_task_running.return_value = None
            mock_dispatcher.record_task_completion.return_value = None

            prep = await service._prepare_task(db, run, synthesis_task, agent_id=1)

        # Verify synthesis prompt was built (not standard build_task_prompt)
        prompt = prep["prompt"]
        assert "Synthesis Task: Merge Research" in prompt
        assert "AI findings" in prompt
        assert "ML findings" in prompt

        # Verify verification criteria were auto-set. Synthesis deliberately
        # emits only a min_length floor — no required_sections, because the
        # planner can't predict the agent's headings (see
        # _auto_synthesis_verification_criteria).
        assert synthesis_task.verification_criteria is not None
        length_check = next(
            c for c in synthesis_task.verification_criteria
            if c["type"] == "min_length"
        )
        # Two upstream outputs of 11 chars each -> 22; floored at 200.
        assert length_check["value"] == 200

    @pytest.mark.asyncio
    async def test_non_synthesis_task_uses_standard_prompt(self):
        """Verify that regular tasks still use build_task_prompt."""
        mock_agent_factory_cls = MagicMock()
        mock_factory_instance = MagicMock()
        mock_factory_instance.active_agents = {}

        async def mock_activate(agent_id, workspace_dir):
            runtime = MagicMock()
            runtime.llm_manager.config.max_tokens = 2000
            return runtime

        async def mock_execute(*, agent, prompt, max_retries, max_tool_iterations):
            return {"status": "success", "execution": {"tokens_used": 100}}

        mock_factory_instance.activate_agent = mock_activate
        mock_factory_instance.execute_with_prompt = mock_execute
        mock_agent_factory_cls.return_value = mock_factory_instance

        mock_factory_module = MagicMock()
        mock_factory_module.AgentFactory = mock_agent_factory_cls

        service = CoordinatorService.__new__(CoordinatorService)
        service._field = None

        task = _make_task(
            title="Research Topic",
            task_type="llm_generation",
        )
        task.state = "assigned"

        run = MagicMock()
        run.id = uuid4()
        run.config = {}
        run.tokens_used = 0
        run.token_budget_estimate = 100000

        db = MagicMock()
        dep_query = MagicMock()
        dep_query.filter.return_value = dep_query
        dep_query.all.return_value = []
        db.query.return_value = dep_query

        with patch(
            "services.coordinator_service.MissionDispatcher"
        ) as mock_dispatcher, patch.dict(
            "sys.modules",
            {"modules.agents.factory.agent_factory": mock_factory_module},
        ), patch.object(
            service, "_inject_task_output_into_field", return_value=None,
        ):
            mock_dispatcher.record_task_running.return_value = None
            mock_dispatcher.record_task_completion.return_value = None
            mock_dispatcher.build_task_prompt.return_value = "Standard prompt"

            await service._prepare_task(db, run, task, agent_id=1)

        # Verify standard build_task_prompt was called
        mock_dispatcher.build_task_prompt.assert_called_once_with(task)
