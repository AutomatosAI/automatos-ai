"""
PRD-82C Wiring Verification Test Suite (US-012)
=================================================

Dedicated integration tests proving ALL 82C features are wired end-to-end
through the coordinator tick, dispatcher, planner, and templates.

These tests trace through REAL code paths, mocking only external dependencies
(LLM calls, DB sessions). They prove the code is CALLED, not just defined.

Tests:
  1. Parallel dispatch — 2 independent tasks both dispatched when max_concurrent=2
  2. Sequential regression — max_concurrent=1 dispatches only 1 task
  3. Dependency blocking — task with unmet deps stays PENDING even with free slots
  4. Complexity detection — complex goal returns max_concurrent >= 2, simple → 1
  5. Parallel group validation — cross-dependent parallel tasks rejected
  6. Budget gate blocks — run at 100%+ budget pauses instead of dispatching
  7. Budget gate allows synthesis at critical — synthesis dispatches at 85%
  8. Synthesis prompt contains all upstream outputs
  9. Template parallel groups — content_pipeline renders with parallel research
 10. Token estimate uses complexity — NOT flat 2000 * task_count
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

from core.models.orchestration_enums import (
    BudgetStatus,
    ComplexityTier,
    EventType,
    RunState,
    TaskState,
    TaskType,
)
from modules.coordination.dispatcher import DispatchResult, MissionDispatcher
from modules.coordination.planner import (
    DecompositionResult,
    MissionPlanner,
    PlannedDependency,
    PlannedTask,
    _complexity_to_max_concurrent,
    _detect_complexity,
    _ensure_synthesis_tasks,
    _estimate_token_budget,
    _parse_plan,
    _validate_plan,
)
from modules.coordination.templates import (
    TEMPLATE_REGISTRY,
    match_template,
    render_template,
)
from services.coordinator_service import CoordinatorService


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_task(
    *,
    run_id=None,
    seq=1,
    state=TaskState.PENDING.value,
    agent_role="researcher",
    task_type="llm_generation",
    estimated_tokens=4000,
    task_id=None,
    title=None,
    description=None,
    output="",
    parallel_group=None,
    verification_criteria=None,
):
    """Create a mock OrchestrationTask with all fields used by 82C code."""
    task = MagicMock()
    task.id = task_id or uuid4()
    task.run_id = run_id or uuid4()
    task.sequence_number = seq
    task.state = state
    task.agent_role = agent_role
    task.task_type = task_type
    task.estimated_tokens = estimated_tokens
    task.input_context = {}
    task.version_id = 1
    task.failure_reason_code = None
    task.failure_detail = None
    task.assigned_agent_id = None
    task.title = title or f"Task {seq}"
    task.description = description or f"Description for task {seq}"
    task.output = output
    task.parallel_group = parallel_group
    task.verification_criteria = verification_criteria
    return task


def _mock_run(
    *,
    run_id=None,
    max_concurrent=2,
    token_budget_estimate=None,
    tokens_used=0,
):
    """Create a mock OrchestrationRun."""
    run = MagicMock()
    run.id = run_id or uuid4()
    run.max_concurrent = max_concurrent
    run.token_budget_estimate = token_budget_estimate
    run.tokens_used = tokens_used
    run.state = RunState.RUNNING.value
    run.config = {}
    run.workspace_id = uuid4()
    return run


def _mock_agent(*, agent_id=1, name="Agent One"):
    """Create a mock Agent for dispatch tests."""
    agent = MagicMock()
    agent.id = agent_id
    agent.name = name
    agent.status = "active"
    agent.skills = []
    agent.tags = []
    return agent


def _mock_match_result(agent_id, agent_name="Agent", score=0.9):
    mr = MagicMock()
    mr.agent_id = agent_id
    mr.agent_name = agent_name
    mr.total_score = score
    return mr


def _setup_dispatch_db(db, active_count=0):
    """Wire mock DB for dispatch_ready: count query + actionable query."""
    count_q = MagicMock()
    count_q.filter.return_value = count_q
    count_q.count.return_value = active_count

    actionable_q = MagicMock()
    actionable_q.filter.return_value = actionable_q
    actionable_q.order_by.return_value = actionable_q
    actionable_q.all.return_value = []

    db.query.side_effect = [count_q, actionable_q]
    db.execute.return_value = MagicMock(rowcount=1)
    db.expire = MagicMock()


def _planned_task(
    temp_id,
    *,
    title="Test",
    agent_role="researcher",
    seq=1,
    task_type="llm_generation",
    deps=None,
    complexity="moderate",
    parallel_group=None,
):
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


# ===========================================================================
# 1. Parallel dispatch — trace through dispatch_ready
# ===========================================================================


@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
class TestWiringParallelDispatch:
    """Prove dispatch_ready sends multiple tasks through the real dispatcher."""

    def test_two_independent_tasks_both_dispatch(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """E2E: max_concurrent=2, 2 independent tasks -> both dispatched in one call."""
        run = _mock_run(max_concurrent=2)
        task_a = _mock_task(run_id=run.id, seq=1)
        task_b = _mock_task(run_id=run.id, seq=2, agent_role="writer")

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [task_a, task_b]
        mock_matcher.match.side_effect = [
            _mock_match_result(1, "Agent One"),
            _mock_match_result(2, "Agent Two"),
        ]

        agents = [_mock_agent(agent_id=1), _mock_agent(agent_id=2, name="Agent Two")]
        results = MissionDispatcher.dispatch_ready(db, run, agents)

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 2
        assert {dispatched[0].task_id, dispatched[1].task_id} == {task_a.id, task_b.id}

        # Verify transition_task was called for both
        assert mock_transition.call_count == 2

    def test_sequential_regression_max_concurrent_1(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """E2E: max_concurrent=1 dispatches only 1 task even with 2 ready."""
        run = _mock_run(max_concurrent=1)
        task_a = _mock_task(run_id=run.id, seq=1)
        task_b = _mock_task(run_id=run.id, seq=2)

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [task_a, task_b]
        mock_matcher.match.return_value = _mock_match_result(1)

        results = MissionDispatcher.dispatch_ready(db, run, [_mock_agent()])
        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1

    def test_dependency_blocking_with_free_slots(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """E2E: task with unmet deps stays PENDING — DependencyResolver excludes it."""
        run = _mock_run(max_concurrent=3)
        task_a = _mock_task(run_id=run.id, seq=1)
        # task_b depends on task_a, so DependencyResolver won't return it

        db = MagicMock()
        _setup_dispatch_db(db)
        # Only task_a is dependency-ready
        mock_dep_resolver.get_ready_tasks.return_value = [task_a]
        mock_matcher.match.return_value = _mock_match_result(1)

        results = MissionDispatcher.dispatch_ready(db, run, [_mock_agent()])
        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1
        assert dispatched[0].task_id == task_a.id


# ===========================================================================
# 2. Complexity detection — trace through _detect_complexity + decompose
# ===========================================================================


class TestWiringComplexityDetection:
    """Prove complexity detection flows through to max_concurrent on result."""

    def test_complex_goal_returns_high_max_concurrent(self):
        tier = _detect_complexity(
            "Write a 4000 word research paper with 6 sections covering "
            "AI coordination, machine learning experiments, cloud deployment, "
            "and a dashboard. Include analysis, a presentation, and a report."
        )
        assert tier == ComplexityTier.COMPLEX
        assert _complexity_to_max_concurrent(tier) == 3

    def test_simple_goal_returns_1(self):
        tier = _detect_complexity("Summarize this document")
        assert tier == ComplexityTier.SIMPLE
        assert _complexity_to_max_concurrent(tier) == 1

    @pytest.mark.asyncio
    async def test_decompose_sets_max_concurrent_from_complexity(self):
        """E2E: decompose() calls _detect_complexity and sets result.max_concurrent."""
        goal = (
            "Write a 4000 word research paper with 6 sections covering "
            "AI coordination, prior art, experiments, and implications"
        )
        fake_plan = {
            "tasks": [
                {
                    "temp_id": f"task_{i}",
                    "title": f"Task {i}",
                    "description": f"Do task {i}",
                    "agent_role": "researcher",
                    "sequence_number": i,
                    "task_type": "research" if i < 4 else "review",
                    "verification_criteria": [],
                    "required_tools": [],
                    "depends_on": [f"task_{i-1}"] if i > 1 else [],
                }
                for i in range(1, 5)
            ]
        }
        agents = [_mock_agent(name="researcher"), _mock_agent(agent_id=2, name="writer")]

        with patch(
            "modules.coordination.planner.match_template", return_value=None
        ), patch("modules.coordination.planner.create_llm_manager") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.generate_response = AsyncMock(
                return_value=MagicMock(content=json.dumps(fake_plan))
            )
            mock_llm_factory.return_value = mock_llm

            result = await MissionPlanner.decompose(
                goal=goal, workspace_id=uuid4(), agents=agents,
            )

        assert isinstance(result, DecompositionResult)
        assert result.max_concurrent >= 2


# ===========================================================================
# 3. Parallel group validation — cross-deps rejected
# ===========================================================================


class TestWiringParallelGroupValidation:
    """Prove _validate_plan rejects cross-dependent parallel group tasks."""

    def test_cross_dependent_parallel_tasks_rejected(self):
        tasks = [
            _planned_task("t1", title="A", parallel_group="research", seq=1),
            _planned_task(
                "t2", title="B", parallel_group="research", seq=1,
                deps=["t1"],  # illegal cross-dep within same group
            ),
            _planned_task("t3", title="Synth", agent_role="writer", seq=2,
                          task_type="synthesis", deps=["t1", "t2"]),
        ]
        deps = [
            PlannedDependency("t1", "t2"),
            PlannedDependency("t1", "t3"),
            PlannedDependency("t2", "t3"),
        ]
        agents = [_mock_agent(name="researcher"), _mock_agent(agent_id=2, name="writer")]
        errors = _validate_plan(tasks, deps, agents)
        assert any("parallel_group" in e and "cross-dependency" in e for e in errors)

    def test_valid_parallel_group_accepted(self):
        tasks = [
            _planned_task("t1", title="A", parallel_group="research", seq=1),
            _planned_task("t2", title="B", parallel_group="research", seq=1),
            _planned_task("t3", title="Synth", agent_role="writer", seq=2,
                          task_type="synthesis", deps=["t1", "t2"]),
        ]
        deps = [
            PlannedDependency("t1", "t3"),
            PlannedDependency("t2", "t3"),
        ]
        agents = [_mock_agent(name="researcher"), _mock_agent(agent_id=2, name="writer")]
        errors = _validate_plan(tasks, deps, agents)
        assert not any("parallel_group" in e for e in errors)


# ===========================================================================
# 4. Budget gate — blocks at exceeded, allows synthesis at critical
# ===========================================================================


@patch("modules.coordination.dispatcher.transition_run")
@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
class TestWiringBudgetGate:
    """Prove budget gate is wired into dispatch_ready and blocks/pauses correctly."""

    def test_exceeded_budget_blocks_and_pauses_mission(
        self, mock_dep_resolver, mock_matcher, mock_transition_task,
        mock_emit, mock_board, mock_sync, mock_transition_run,
    ):
        """E2E: >100% budget -> task blocked, run paused via transition_run."""
        run = _mock_run(max_concurrent=2, token_budget_estimate=1000, tokens_used=1050)
        task = _mock_task(run_id=run.id, seq=1, estimated_tokens=8000)

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [task]

        results = MissionDispatcher.dispatch_ready(db, run, [_mock_agent()])

        # Task blocked
        blocked = [r for r in results if r.skipped_reason == "budget_exceeded"]
        assert len(blocked) == 1

        # Run paused
        mock_transition_run.assert_called_once()
        assert mock_transition_run.call_args.kwargs["new_state"] == RunState.PAUSED

        # Budget exceeded event emitted
        budget_events = [
            c for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == EventType.RUN_BUDGET_EXCEEDED
        ]
        assert len(budget_events) == 1

    def test_critical_budget_allows_synthesis_defers_heavy(
        self, mock_dep_resolver, mock_matcher, mock_transition_task,
        mock_emit, mock_board, mock_sync, mock_transition_run,
    ):
        """E2E: 85% budget -> synthesis dispatches, heavy task deferred."""
        run = _mock_run(max_concurrent=3, token_budget_estimate=10000, tokens_used=8500)

        heavy = _mock_task(run_id=run.id, seq=1, estimated_tokens=8000)
        synthesis = _mock_task(
            run_id=run.id, seq=2,
            task_type=TaskType.SYNTHESIS.value,
            estimated_tokens=6000,
            agent_role="writer",
        )

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [heavy, synthesis]
        mock_matcher.match.return_value = _mock_match_result(2, "Agent Two")

        results = MissionDispatcher.dispatch_ready(db, run, [_mock_agent(), _mock_agent(agent_id=2)])

        deferred = [r for r in results if r.skipped_reason == "budget_critical_deferred"]
        assert len(deferred) == 1 and deferred[0].task_id == heavy.id

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1 and dispatched[0].task_id == synthesis.id

        # Run NOT paused (only exceeded pauses)
        mock_transition_run.assert_not_called()


# ===========================================================================
# 5. Synthesis prompt — contains all upstream outputs
# ===========================================================================


class TestWiringSynthesisPrompt:
    """Prove synthesis prompt is built from upstream task outputs."""

    def test_synthesis_prompt_contains_upstream_outputs(self):
        """E2E: _build_synthesis_prompt includes all upstream titles and outputs."""
        task = _mock_task(
            title="Merge Research",
            description="Combine AI and ML research",
            task_type=TaskType.SYNTHESIS.value,
        )
        upstream = [
            {"title": "Research AI", "description": "", "output": "AI is transformative and impactful"},
            {"title": "Research ML", "description": "", "output": "ML enables automation at scale"},
        ]

        prompt = CoordinatorService._build_synthesis_prompt(task, upstream)

        assert "Synthesis Task: Merge Research" in prompt
        assert "Combine AI and ML research" in prompt
        assert "Research AI" in prompt
        assert "AI is transformative and impactful" in prompt
        assert "Research ML" in prompt
        assert "ML enables automation at scale" in prompt
        assert "unified voice" in prompt

    def test_collect_upstream_outputs_fetches_deps(self):
        """E2E: _collect_upstream_outputs traces dependency graph."""
        task_a = _mock_task(title="Research AI", output="AI findings", seq=1)
        task_b = _mock_task(title="Research ML", output="ML findings", seq=2)
        synthesis = _mock_task(title="Merge", task_type=TaskType.SYNTHESIS.value, seq=3)

        # Mock deps
        dep_a = MagicMock()
        dep_a.task_id = synthesis.id
        dep_a.depends_on_task_id = task_a.id
        dep_b = MagicMock()
        dep_b.task_id = synthesis.id
        dep_b.depends_on_task_id = task_b.id

        dep_query = MagicMock()
        dep_query.filter.return_value = dep_query
        dep_query.all.return_value = [dep_a, dep_b]

        task_query = MagicMock()
        task_query.filter.return_value = task_query
        task_query.order_by.return_value = task_query
        task_query.all.return_value = [task_a, task_b]

        call_count = [0]
        def side_effect(model):
            call_count[0] += 1
            return dep_query if call_count[0] % 2 == 1 else task_query

        db = MagicMock()
        db.query.side_effect = side_effect

        result = CoordinatorService._collect_upstream_outputs(db, synthesis)
        assert len(result) == 2
        assert result[0]["title"] == "Research AI"
        assert result[0]["output"] == "AI findings"
        assert result[1]["title"] == "Research ML"

    def test_auto_verification_criteria_generated(self):
        """E2E: synthesis tasks get auto-generated verification criteria."""
        task = _mock_task(task_type=TaskType.SYNTHESIS.value)
        upstream = [
            {"title": "Research AI", "output": "x" * 2000},
            {"title": "Research ML", "output": "y" * 2000},
        ]

        criteria = CoordinatorService._auto_synthesis_verification_criteria(task, upstream)

        section_check = next(c for c in criteria if c["type"] == "required_sections")
        assert "Research AI" in section_check["value"]
        assert "Research ML" in section_check["value"]

        length_check = next(c for c in criteria if c["type"] == "min_length")
        assert length_check["value"] == 2000  # 50% of 4000


# ===========================================================================
# 6. Template parallel groups — content_pipeline wired correctly
# ===========================================================================


class TestWiringTemplateParallelGroups:
    """Prove templates produce parallel groups and synthesis tasks."""

    def test_content_pipeline_has_parallel_research(self):
        """E2E: content_pipeline template renders with parallel research group."""
        template = next(t for t in TEMPLATE_REGISTRY if t.id == "content_pipeline")
        tasks = render_template(template, "Write a blog post about AI")

        research_tasks = [t for t in tasks if t.get("parallel_group") == "research"]
        assert len(research_tasks) == 2

        for rt in research_tasks:
            assert rt["dependencies"] == []

        synth = [t for t in tasks if t.get("task_type") == "synthesis"]
        assert len(synth) >= 1

        research_ids = {rt["temp_id"] for rt in research_tasks}
        for rid in research_ids:
            assert rid in synth[0]["dependencies"]

    def test_all_templates_produce_parallel_and_synthesis(self):
        """E2E: every template has parallel groups and synthesis tasks."""
        for template in TEMPLATE_REGISTRY:
            tasks = render_template(template, "Test goal")

            groups = {}
            for t in tasks:
                pg = t.get("parallel_group")
                if pg:
                    groups[pg] = groups.get(pg, 0) + 1
            assert any(c >= 2 for c in groups.values()), (
                f"Template '{template.id}' missing parallel group with 2+ tasks"
            )

            synth = [t for t in tasks if t.get("task_type") == "synthesis"]
            assert len(synth) >= 1, f"Template '{template.id}' missing synthesis task"

    def test_rendered_templates_pass_validation(self):
        """E2E: rendered templates pass _parse_plan + _validate_plan."""
        agents = [
            _mock_agent(name="researcher"),
            _mock_agent(agent_id=2, name="writer"),
            _mock_agent(agent_id=3, name="analyst"),
            _mock_agent(agent_id=4, name="search"),
            _mock_agent(agent_id=5, name="reviewer"),
        ]
        for template in TEMPLATE_REGISTRY:
            tasks = render_template(template, "Test validation goal")
            errors = []
            parsed_tasks, deps = _parse_plan({"tasks": tasks}, errors)
            assert not errors, f"Template '{template.id}' parse errors: {errors}"

            val_errors = _validate_plan(parsed_tasks, deps, agents)
            assert not val_errors, f"Template '{template.id}' validation errors: {val_errors}"


# ===========================================================================
# 7. Token estimate uses complexity — NOT flat 2000 * task_count
# ===========================================================================


class TestWiringTokenEstimate:
    """Prove token estimation uses COMPLEXITY_TOKEN_BUDGET, not flat rate."""

    def test_estimate_varies_by_complexity(self):
        """E2E: simple + complex + moderate != 2000 * 3."""
        tasks = [
            _planned_task("t1", complexity="simple", seq=1),
            _planned_task("t2", complexity="complex", seq=2, deps=["t1"]),
            _planned_task("t3", complexity="moderate", seq=3,
                          task_type="synthesis", deps=["t2"]),
        ]
        estimate = _estimate_token_budget(tasks)
        flat = 2000 * 3
        assert estimate != flat
        # simple(5000) + complex(35000) + moderate(15000) = 55000
        assert estimate == 55000

    def test_synthesis_complexity_uses_config(self):
        """E2E: 'synthesis' complexity tier maps to its own budget."""
        tasks = [
            _planned_task("t1", complexity="synthesis", seq=1,
                          task_type="synthesis"),
        ]
        estimate = _estimate_token_budget(tasks)
        assert estimate == 20000  # COMPLEXITY_TOKEN_BUDGET["synthesis"]


# ===========================================================================
# 8. Coordinator tick wiring — dispatch_ready called, not dispatch_next
# ===========================================================================


class TestWiringCoordinatorTick:
    """Prove _process_run uses dispatch_ready and gathers concurrent tasks."""

    @pytest.mark.asyncio
    async def test_process_run_calls_dispatch_ready(self):
        """E2E: coordinator tick calls dispatch_ready, executes both tasks."""
        svc = CoordinatorService.__new__(CoordinatorService)

        # Tick phases: _prepare_task (serial DB) -> _run_agent_io (concurrent)
        # -> _record_task_result (serial). _prepare_task threads the real task
        # through in its prep dict.
        async def _prep(db, run, task, agent_id):
            return {
                "task": task, "agent_id": agent_id, "agent_runtime": None,
                "prompt": f"prompt-{task.id}", "factory": MagicMock(),
                "attachment_ids": [], "mode_caps": {},
            }

        svc._prepare_task = AsyncMock(side_effect=_prep)
        svc._run_agent_io = AsyncMock(return_value={"status": "success"})
        svc._record_task_result = AsyncMock()
        svc._create_mission_field = AsyncMock(return_value="field-1")
        svc._get_field = MagicMock(return_value=None)

        run = _mock_run(max_concurrent=2)
        run.state = "running"

        task_1 = _mock_task(seq=1)
        task_2 = _mock_task(seq=2)
        dr1 = DispatchResult(dispatched=True, task_id=task_1.id, agent_id=1, agent_name="A1")
        dr2 = DispatchResult(dispatched=True, task_id=task_2.id, agent_id=2, agent_name="A2")

        _task_list = [task_1, task_2]
        db = MagicMock()

        def _build_query(model):
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
            q.first.side_effect = lambda: _task_list.pop(0) if _task_list else None
            return q

        db.query.side_effect = _build_query
        db.refresh = MagicMock()

        with (
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_ready",
                return_value=[dr1, dr2],
            ) as mock_ready,
            patch(
                "services.coordinator_service.MissionReconciler.reconcile",
                new_callable=AsyncMock,
            ) as mock_reconcile,
        ):
            await svc._process_run(db, run)

            mock_ready.assert_called_once()
            assert svc._run_agent_io.call_count == 2
            mock_reconcile.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_one_task_doesnt_crash_tick(self):
        """E2E: exception in one gather'd task doesn't prevent reconciliation."""
        svc = CoordinatorService.__new__(CoordinatorService)
        svc._create_mission_field = AsyncMock(return_value="field-1")
        svc._get_field = MagicMock(return_value=None)

        call_count = {"n": 0}
        task_1 = _mock_task(seq=1)
        task_2 = _mock_task(seq=2)

        async def _prep(db, run, task, agent_id):
            return {
                "task": task, "agent_id": agent_id, "agent_runtime": None,
                "prompt": f"prompt-{task.id}", "factory": MagicMock(),
                "attachment_ids": [], "mode_caps": {},
            }

        async def _io_side_effect(factory, agent_id, prompt, task,
                                  attachment_ids, *, mode_caps=None,
                                  agent_runtime=None):
            call_count["n"] += 1
            if task.id == task_1.id:
                raise RuntimeError("LLM timeout")

        # gather(return_exceptions=True) captures the raise; the tick records
        # an error result and still reconciles.
        svc._prepare_task = AsyncMock(side_effect=_prep)
        svc._run_agent_io = AsyncMock(side_effect=_io_side_effect)
        svc._record_task_result = AsyncMock()

        run = _mock_run(max_concurrent=2)
        run.state = "running"

        dr1 = DispatchResult(dispatched=True, task_id=task_1.id, agent_id=1, agent_name="A1")
        dr2 = DispatchResult(dispatched=True, task_id=task_2.id, agent_id=2, agent_name="A2")

        _task_list = [task_1, task_2]
        db = MagicMock()

        def _build_query(model):
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
            q.first.side_effect = lambda: _task_list.pop(0) if _task_list else None
            return q

        db.query.side_effect = _build_query
        db.refresh = MagicMock()

        with (
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_ready",
                return_value=[dr1, dr2],
            ),
            patch(
                "services.coordinator_service.MissionReconciler.reconcile",
                new_callable=AsyncMock,
            ) as mock_reconcile,
        ):
            await svc._process_run(db, run)
            assert call_count["n"] == 2
            mock_reconcile.assert_called_once()


# ===========================================================================
# 9. Synthesis auto-insertion wiring
# ===========================================================================


class TestWiringSynthesisAutoInsertion:
    """Prove _ensure_synthesis_tasks auto-inserts correctly."""

    def test_auto_inserts_when_no_explicit_synthesis(self):
        """E2E: parallel group with non-synthesis downstream -> auto-insert."""
        tasks = [
            _planned_task("t1", title="Research AI", seq=1, parallel_group="research"),
            _planned_task("t2", title="Research ML", seq=1, parallel_group="research"),
            _planned_task("t3", title="Write report", agent_role="writer", seq=2,
                          deps=["t1", "t2"]),
        ]
        deps = [
            PlannedDependency("t1", "t3"),
            PlannedDependency("t2", "t3"),
        ]

        new_tasks, _ = _ensure_synthesis_tasks(tasks, deps)

        assert len(new_tasks) == 4
        synth = [t for t in new_tasks if t.task_type == TaskType.SYNTHESIS.value]
        assert len(synth) == 1
        assert synth[0].temp_id == "synth_research"
        assert set(synth[0].dependencies) == {"t1", "t2"}

        # Downstream repointed
        task_3 = next(t for t in new_tasks if t.temp_id == "t3")
        assert task_3.dependencies == ["synth_research"]

    def test_no_insert_when_explicit_synthesis_exists(self):
        """E2E: explicit synthesis after parallel -> no auto-insertion."""
        tasks = [
            _planned_task("t1", title="Research AI", seq=1, parallel_group="research"),
            _planned_task("t2", title="Research ML", seq=1, parallel_group="research"),
            _planned_task("t_synth", title="Merge", agent_role="writer", seq=2,
                          task_type=TaskType.SYNTHESIS.value, deps=["t1", "t2"]),
            _planned_task("t4", title="Report", agent_role="writer", seq=3,
                          deps=["t_synth"]),
        ]
        deps = [
            PlannedDependency("t1", "t_synth"),
            PlannedDependency("t2", "t_synth"),
            PlannedDependency("t_synth", "t4"),
        ]

        new_tasks, _ = _ensure_synthesis_tasks(tasks, deps)
        assert len(new_tasks) == 4
        assert len(new_deps) == 3
