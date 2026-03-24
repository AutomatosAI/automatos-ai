"""
Wiring tests for US-002: Parallel dispatch in MissionDispatcher.
================================================================

Proves:
1. dispatch_ready() dispatches multiple tasks when max_concurrent > 1
2. dispatch_ready() respects max_concurrent=1 (regression)
3. dispatch_ready() respects dependency ordering
4. count_active_tasks() returns correct count
5. has_active_task() and dispatch_next() emit deprecation warnings
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from core.models.orchestration_enums import TaskState
from modules.coordination.dispatcher import MissionDispatcher, DispatchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    *,
    run_id,
    seq,
    state=TaskState.PENDING.value,
    agent_role="researcher",
    task_id=None,
):
    """Create a mock OrchestrationTask."""
    task = MagicMock()
    task.id = task_id or uuid4()
    task.run_id = run_id
    task.sequence_number = seq
    task.state = state
    task.agent_role = agent_role
    task.input_context = {}
    task.version_id = 1
    task.failure_reason_code = None
    task.failure_detail = None
    task.assigned_agent_id = None
    task.title = f"Task {seq}"
    task.description = f"Description for task {seq}"
    return task


def _make_run(*, run_id=None, max_concurrent=1):
    """Create a mock OrchestrationRun."""
    run = MagicMock()
    run.id = run_id or uuid4()
    run.max_concurrent = max_concurrent
    run.token_budget_estimate = None  # No budget = unlimited
    run.tokens_used = 0
    return run


def _make_agent(*, agent_id=1, name="Agent One"):
    """Create a mock Agent."""
    agent = MagicMock()
    agent.id = agent_id
    agent.name = name
    return agent


def _make_match_result(agent_id, agent_name="Agent", score=0.9):
    """Create a mock MatchResult."""
    mr = MagicMock()
    mr.agent_id = agent_id
    mr.agent_name = agent_name
    mr.total_score = score
    return mr


# ---------------------------------------------------------------------------
# count_active_tasks
# ---------------------------------------------------------------------------


class TestCountActiveTasks:
    def test_returns_zero_when_no_active(self):
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.count.return_value = 0
        db.query.return_value = q
        assert MissionDispatcher.count_active_tasks(db, uuid4()) == 0

    def test_returns_count_of_active(self):
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.count.return_value = 2
        db.query.return_value = q
        assert MissionDispatcher.count_active_tasks(db, uuid4()) == 2


# ---------------------------------------------------------------------------
# dispatch_ready — parallel dispatch
# ---------------------------------------------------------------------------

@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
class TestDispatchReady:

    def test_dispatches_two_tasks_when_max_concurrent_is_2(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """WIRING TEST: run with max_concurrent=2 and 2 independent PENDING tasks — both dispatch."""
        run = _make_run(max_concurrent=2)
        run_id = run.id
        agents = [_make_agent(agent_id=1), _make_agent(agent_id=2, name="Agent Two")]

        task_a = _make_task(run_id=run_id, seq=1)
        task_b = _make_task(run_id=run_id, seq=2)

        # Mock DB: no active tasks, no queued/retrying tasks
        db = MagicMock()
        count_q = MagicMock()
        count_q.filter.return_value = count_q
        count_q.count.return_value = 0  # no active tasks

        actionable_q = MagicMock()
        actionable_q.filter.return_value = actionable_q
        actionable_q.order_by.return_value = actionable_q
        actionable_q.all.return_value = []  # no queued/retrying

        # query() returns different mocks for count vs filter-all
        db.query.side_effect = [count_q, actionable_q]

        # DependencyResolver returns both tasks as ready
        mock_dep_resolver.get_ready_tasks.return_value = [task_a, task_b]

        # AgentMatcher returns match for each task
        mock_matcher.match.side_effect = [
            _make_match_result(1, "Agent One"),
            _make_match_result(2, "Agent Two"),
        ]

        # claim_task succeeds (mock the raw SQL)
        db.execute.return_value = MagicMock(rowcount=1)
        db.expire = MagicMock()

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 2, f"Expected 2 dispatched, got {len(dispatched)}: {results}"
        assert dispatched[0].task_id == task_a.id
        assert dispatched[1].task_id == task_b.id

    def test_dispatches_one_when_max_concurrent_is_1(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """WIRING TEST: run with max_concurrent=1 and 2 independent tasks — only 1 dispatches."""
        run = _make_run(max_concurrent=1)
        run_id = run.id
        agents = [_make_agent()]

        task_a = _make_task(run_id=run_id, seq=1)
        task_b = _make_task(run_id=run_id, seq=2)

        db = MagicMock()
        count_q = MagicMock()
        count_q.filter.return_value = count_q
        count_q.count.return_value = 0

        actionable_q = MagicMock()
        actionable_q.filter.return_value = actionable_q
        actionable_q.order_by.return_value = actionable_q
        actionable_q.all.return_value = []

        db.query.side_effect = [count_q, actionable_q]
        mock_dep_resolver.get_ready_tasks.return_value = [task_a, task_b]
        mock_matcher.match.return_value = _make_match_result(1)
        db.execute.return_value = MagicMock(rowcount=1)
        db.expire = MagicMock()

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1, f"Expected 1 dispatched, got {len(dispatched)}"

    def test_respects_dependency_ordering(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """WIRING TEST: task_a (no deps) and task_b (depends on task_a) — only task_a dispatches."""
        run = _make_run(max_concurrent=3)
        run_id = run.id
        agents = [_make_agent()]

        task_a = _make_task(run_id=run_id, seq=1)
        # task_b depends on task_a so it won't appear in get_ready_tasks

        db = MagicMock()
        count_q = MagicMock()
        count_q.filter.return_value = count_q
        count_q.count.return_value = 0

        actionable_q = MagicMock()
        actionable_q.filter.return_value = actionable_q
        actionable_q.order_by.return_value = actionable_q
        actionable_q.all.return_value = []

        db.query.side_effect = [count_q, actionable_q]

        # Only task_a is ready (task_b's dependency not met)
        mock_dep_resolver.get_ready_tasks.return_value = [task_a]
        mock_matcher.match.return_value = _make_match_result(1)
        db.execute.return_value = MagicMock(rowcount=1)
        db.expire = MagicMock()

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1
        assert dispatched[0].task_id == task_a.id

    def test_returns_max_concurrent_reached_when_slots_full(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """No slots available → returns max_concurrent_reached."""
        run = _make_run(max_concurrent=2)

        db = MagicMock()
        count_q = MagicMock()
        count_q.filter.return_value = count_q
        count_q.count.return_value = 2  # 2 active = max_concurrent

        db.query.return_value = count_q

        results = MissionDispatcher.dispatch_ready(db, run, [_make_agent()])

        assert len(results) == 1
        assert not results[0].dispatched
        assert results[0].skipped_reason == "max_concurrent_reached"

    def test_returns_no_ready_tasks_when_none_available(
        self, mock_dep_resolver, mock_matcher, mock_transition,
        mock_emit, mock_board, mock_sync,
    ):
        """No queued/retrying and no ready tasks → no_ready_tasks."""
        run = _make_run(max_concurrent=2)

        db = MagicMock()
        count_q = MagicMock()
        count_q.filter.return_value = count_q
        count_q.count.return_value = 0

        actionable_q = MagicMock()
        actionable_q.filter.return_value = actionable_q
        actionable_q.order_by.return_value = actionable_q
        actionable_q.all.return_value = []

        db.query.side_effect = [count_q, actionable_q]
        mock_dep_resolver.get_ready_tasks.return_value = []

        results = MissionDispatcher.dispatch_ready(db, run, [_make_agent()])

        assert len(results) == 1
        assert not results[0].dispatched
        assert results[0].skipped_reason == "no_ready_tasks"


# ---------------------------------------------------------------------------
# Deprecation warnings
# ---------------------------------------------------------------------------


class TestDeprecation:
    def test_has_active_task_emits_deprecation(self):
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.count.return_value = 0
        db.query.return_value = q

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MissionDispatcher.has_active_task(db, uuid4())
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "dispatch_ready" in str(deprecation_warnings[0].message)

    def test_dispatch_next_emits_deprecation(self):
        run = _make_run(max_concurrent=1)
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.count.return_value = 1  # active task → early return
        db.query.return_value = q

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MissionDispatcher.dispatch_next(db, run, [])
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "dispatch_ready" in str(deprecation_warnings[0].message)
