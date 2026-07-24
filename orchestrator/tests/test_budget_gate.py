"""
Wiring tests for US-009: Budget admission gate in MissionDispatcher.
====================================================================

Proves:
1. Run at >100% budget → heavy task blocked, mission paused
2. Run at 80-100% budget → synthesis task allowed, heavy task deferred
3. Run with no token_budget_estimate → all tasks dispatch normally
4. _get_budget_status returns correct thresholds
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from core.models.orchestration_enums import (
    BudgetStatus,
    EventType,
    RunState,
    TaskState,
    TaskType,
)
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
    task_type="llm_generation",
    estimated_tokens=4000,
    task_id=None,
):
    """Create a mock OrchestrationTask."""
    task = MagicMock()
    task.id = task_id or uuid4()
    task.run_id = run_id
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
    task.title = f"Task {seq}"
    task.description = f"Description for task {seq}"
    return task


def _make_run(*, run_id=None, max_concurrent=2, token_budget_estimate=None, tokens_used=0, config=None):
    """Create a mock OrchestrationRun.

    `config` mirrors the real OrchestrationRun.config JSONB column (DB default
    '{}'). Defaulting to an empty dict is required: the budget gate reads
    ``run.config.get("budget_pause_disabled")`` and a bare MagicMock would make
    that truthy, short-circuiting the gate to "allow".
    """
    run = MagicMock()
    run.id = run_id or uuid4()
    run.max_concurrent = max_concurrent
    run.token_budget_estimate = token_budget_estimate
    run.tokens_used = tokens_used
    run.state = RunState.RUNNING.value
    run.config = {} if config is None else config
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


def _setup_dispatch_db(db, active_count=0, actionable_tasks=None):
    """Wire mock DB for dispatch_ready: count query + actionable query."""
    count_q = MagicMock()
    count_q.filter.return_value = count_q
    count_q.count.return_value = active_count

    actionable_q = MagicMock()
    actionable_q.filter.return_value = actionable_q
    actionable_q.order_by.return_value = actionable_q
    actionable_q.all.return_value = actionable_tasks or []

    db.query.side_effect = [count_q, actionable_q]
    db.execute.return_value = MagicMock(rowcount=1)
    db.expire = MagicMock()


# ---------------------------------------------------------------------------
# _get_budget_status unit tests
# ---------------------------------------------------------------------------

class TestGetBudgetStatus:
    def test_no_budget_returns_healthy(self):
        run = _make_run(token_budget_estimate=None, tokens_used=9999)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.HEALTHY

    def test_zero_budget_returns_healthy(self):
        run = _make_run(token_budget_estimate=0, tokens_used=500)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.HEALTHY

    def test_under_50_pct_is_healthy(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=4000)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.HEALTHY

    def test_at_50_pct_is_warning(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=5000)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.WARNING

    def test_at_80_pct_is_critical(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=8000)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.CRITICAL

    def test_over_100_pct_is_exceeded(self):
        run = _make_run(token_budget_estimate=1000, tokens_used=1500)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.EXCEEDED


# ---------------------------------------------------------------------------
# _pre_dispatch_budget_check unit tests
# ---------------------------------------------------------------------------

class TestPreDispatchBudgetCheck:
    def test_no_budget_always_allows(self):
        run = _make_run(token_budget_estimate=None, tokens_used=9999)
        task = _make_task(run_id=run.id, seq=1, estimated_tokens=8000)
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "allow"

    def test_healthy_allows(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=2000)
        task = _make_task(run_id=run.id, seq=1)
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "allow"

    @patch("modules.coordination.dispatcher.emit_event")
    def test_warning_allows_with_event(self, mock_emit):
        run = _make_run(token_budget_estimate=10000, tokens_used=6000)
        task = _make_task(run_id=run.id, seq=1)
        db = MagicMock()
        result = MissionDispatcher._pre_dispatch_budget_check(db, run, task)
        assert result == "allow"
        mock_emit.assert_called_once()
        assert mock_emit.call_args.kwargs["event_type"] == EventType.RUN_BUDGET_WARNING

    def test_critical_defers_heavy_task(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=8500)
        task = _make_task(run_id=run.id, seq=1, task_type="llm_generation")
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "defer"

    def test_critical_allows_synthesis_task(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=8500)
        task = _make_task(run_id=run.id, seq=1, task_type=TaskType.SYNTHESIS.value)
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "allow"

    def test_critical_allows_review_task(self):
        run = _make_run(token_budget_estimate=10000, tokens_used=8500)
        task = _make_task(run_id=run.id, seq=1, task_type=TaskType.REVIEW.value)
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "allow"

    def test_exceeded_blocks(self):
        run = _make_run(token_budget_estimate=1000, tokens_used=1500)
        task = _make_task(run_id=run.id, seq=1)
        db = MagicMock()
        assert MissionDispatcher._pre_dispatch_budget_check(db, run, task) == "block"


# ---------------------------------------------------------------------------
# dispatch_ready integration with budget gate
# ---------------------------------------------------------------------------

@patch("modules.coordination.dispatcher.transition_run")
@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
class TestDispatchReadyBudgetGate:

    def test_exceeded_budget_blocks_and_pauses(
        self, mock_dep_resolver, mock_matcher, mock_transition_task,
        mock_emit, mock_board, mock_sync, mock_transition_run,
    ):
        """WIRING: Run at >100% budget — heavy task blocked, mission paused."""
        run = _make_run(
            max_concurrent=2,
            token_budget_estimate=1000,
            tokens_used=1050,
        )
        agents = [_make_agent()]
        task = _make_task(
            run_id=run.id, seq=1,
            estimated_tokens=8000,
            task_type="llm_generation",
        )

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [task]

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        # Task should be blocked
        blocked = [r for r in results if r.skipped_reason == "budget_exceeded"]
        assert len(blocked) == 1
        assert blocked[0].task_id == task.id

        # Run should be paused
        mock_transition_run.assert_called_once()
        call_kwargs = mock_transition_run.call_args.kwargs
        assert call_kwargs["new_state"] == RunState.PAUSED

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
        """WIRING: Run at 85% budget — synthesis task dispatches, heavy task deferred."""
        run = _make_run(
            max_concurrent=3,
            token_budget_estimate=10000,
            tokens_used=8500,
        )
        agents = [_make_agent(), _make_agent(agent_id=2, name="Agent Two")]

        heavy_task = _make_task(
            run_id=run.id, seq=1,
            task_type="llm_generation",
            estimated_tokens=8000,
        )
        synthesis_task = _make_task(
            run_id=run.id, seq=2,
            task_type=TaskType.SYNTHESIS.value,
            estimated_tokens=6000,
            agent_role="writer",
        )

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [heavy_task, synthesis_task]
        mock_matcher.match.return_value = _make_match_result(2, "Agent Two")

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        # Heavy task deferred
        deferred = [r for r in results if r.skipped_reason == "budget_critical_deferred"]
        assert len(deferred) == 1
        assert deferred[0].task_id == heavy_task.id

        # Synthesis task dispatched
        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 1
        assert dispatched[0].task_id == synthesis_task.id

        # Run NOT paused (only exceeded pauses)
        mock_transition_run.assert_not_called()

    def test_no_budget_dispatches_normally(
        self, mock_dep_resolver, mock_matcher, mock_transition_task,
        mock_emit, mock_board, mock_sync, mock_transition_run,
    ):
        """WIRING: Run with no token_budget_estimate — all tasks dispatch normally."""
        run = _make_run(
            max_concurrent=2,
            token_budget_estimate=None,
            tokens_used=0,
        )
        agents = [_make_agent(), _make_agent(agent_id=2, name="Agent Two")]

        task_a = _make_task(run_id=run.id, seq=1)
        task_b = _make_task(run_id=run.id, seq=2)

        db = MagicMock()
        _setup_dispatch_db(db)
        mock_dep_resolver.get_ready_tasks.return_value = [task_a, task_b]
        mock_matcher.match.side_effect = [
            _make_match_result(1, "Agent One"),
            _make_match_result(2, "Agent Two"),
        ]

        results = MissionDispatcher.dispatch_ready(db, run, agents)

        dispatched = [r for r in results if r.dispatched]
        assert len(dispatched) == 2

        # No budget events or pausing
        mock_transition_run.assert_not_called()
        budget_events = [
            c for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") in (
                EventType.RUN_BUDGET_EXCEEDED,
                EventType.RUN_BUDGET_WARNING,
            )
        ]
        assert len(budget_events) == 0


# ---------------------------------------------------------------------------
# PRD-163 S5: dollar-ceiling budget (replaces the token-estimate pause).
# An explicit config['cost_ceiling'] (USD) drives the gate; otherwise the token
# estimate is priced out at COORDINATOR_COST_PER_1K_TOKENS ($0.003/1k).
# ---------------------------------------------------------------------------

class TestDollarCeiling:
    def test_cost_ceiling_exceeded(self):
        # 200k tokens -> $0.60 > $0.30 ceiling -> EXCEEDED
        run = _make_run(config={"cost_ceiling": 0.30}, tokens_used=200_000)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.EXCEEDED

    def test_cost_ceiling_healthy_under_half(self):
        # 50k tokens -> $0.15 = 25% of a $0.60 ceiling -> HEALTHY
        run = _make_run(config={"cost_ceiling": 0.60}, tokens_used=50_000)
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.HEALTHY

    def test_dollar_ceiling_overrides_token_estimate(self):
        # The token estimate alone would read healthy, but the small $ ceiling wins.
        run = _make_run(
            token_budget_estimate=10_000_000, tokens_used=100_000,
            config={"cost_ceiling": 0.10},
        )
        # cost = $0.30 > $0.10 ceiling -> EXCEEDED
        assert MissionDispatcher._get_budget_status(run) == BudgetStatus.EXCEEDED
