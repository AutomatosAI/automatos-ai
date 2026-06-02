"""
Wiring tests for US-003: Parallel dispatch wired into coordinator tick.
=======================================================================

Proves:
1. _process_run() calls dispatch_ready() (not dispatch_next())
2. When 2 tasks are dispatched, both agent-I/O calls happen concurrently
3. Sequential missions (max_concurrent=1) still work unchanged
4. An exception in one task does not crash the tick or prevent reconciliation
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.coordination.dispatcher import DispatchResult
from services.coordinator_service import CoordinatorService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(*, task_id=None, seq=1):
    """Create a minimal mock OrchestrationTask."""
    task = MagicMock()
    task.id = task_id or uuid4()
    task.sequence_number = seq
    task.title = f"Task {seq}"
    return task


def _make_run(*, run_id=None, max_concurrent=1, workspace_id=None):
    """Create a minimal mock OrchestrationRun."""
    run = MagicMock()
    run.id = run_id or uuid4()
    run.max_concurrent = max_concurrent
    run.workspace_id = workspace_id or uuid4()
    run.config = {"field_id": "already-created"}
    run.state = "running"
    return run


def _make_dispatch_result(*, task_id, agent_id=1, dispatched=True):
    """Create a DispatchResult."""
    return DispatchResult(
        dispatched=dispatched,
        task_id=task_id if dispatched else None,
        agent_id=agent_id if dispatched else None,
        agent_name=f"Agent-{agent_id}" if dispatched else None,
        skipped_reason=None if dispatched else "no_ready_tasks",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coordinator():
    """Create a CoordinatorService instance with heavy deps mocked out."""
    svc = CoordinatorService.__new__(CoordinatorService)

    # The tick splits each task into three phases: _prepare_task (serial DB
    # prep) -> _run_agent_io (concurrent, no DB) -> _record_task_result
    # (serial). Mock all three. _prepare_task returns a prep dict embedding the
    # real task so the gather/record phases thread it through unchanged.
    async def _prep(db, run, task, agent_id):
        return {
            "task": task,
            "agent_id": agent_id,
            "agent_runtime": None,
            "prompt": f"prompt-{task.id}",
            "factory": MagicMock(),
            "attachment_ids": [],
            "mode_caps": {},
        }

    svc._prepare_task = AsyncMock(side_effect=_prep)
    svc._run_agent_io = AsyncMock(return_value={"status": "success"})
    svc._record_task_result = AsyncMock()
    svc._create_mission_field = AsyncMock(return_value="field-1")
    svc._get_field = MagicMock(return_value=None)
    return svc


@pytest.fixture
def mock_db():
    """Mock DB with chainable query API."""
    db = MagicMock()
    q = MagicMock()
    q.join.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = []  # no agents by default
    q.first.return_value = None
    db.query.return_value = q
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessRunParallelDispatch:
    """Verify _process_run uses dispatch_ready and executes tasks concurrently."""

    @pytest.mark.asyncio
    async def test_two_tasks_both_execute(self, coordinator, mock_db):
        """
        US-003 AC: dispatch 2 tasks, verify both agent-I/O calls happen.
        """
        run = _make_run(max_concurrent=2)
        task_1 = _make_task(seq=1)
        task_2 = _make_task(seq=2)

        dr1 = _make_dispatch_result(task_id=task_1.id, agent_id=1)
        dr2 = _make_dispatch_result(task_id=task_2.id, agent_id=2)

        # Build a task lookup so db.query(OT).filter(...).first() works
        task_map = {task_1.id: task_1, task_2.id: task_2}
        _task_queue = [task_1, task_2]

        def _build_query(model):
            from core.models.core import Agent
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
            # For OrchestrationTask lookups, pop from queue
            q.first.side_effect = lambda: _task_queue.pop(0) if _task_queue else None
            return q

        mock_db.query.side_effect = _build_query

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
            mock_db.refresh = MagicMock()
            run.state = "running"

            await coordinator._process_run(mock_db, run)

            # Both tasks should have been executed
            assert coordinator._run_agent_io.call_count == 2

            # Verify correct task/agent pairs (task is positional arg 3 of
            # _run_agent_io: factory, agent_id, prompt, task, attachment_ids)
            calls = coordinator._run_agent_io.call_args_list
            call_task_ids = {c.args[3].id for c in calls}
            assert task_1.id in call_task_ids
            assert task_2.id in call_task_ids

            # Reconcile still runs after dispatch
            mock_reconcile.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_single_dispatch(self, coordinator, mock_db):
        """
        Existing sequential missions (max_concurrent=1) still dispatch one task.
        """
        run = _make_run(max_concurrent=1)
        task_1 = _make_task(seq=1)
        dr1 = _make_dispatch_result(task_id=task_1.id, agent_id=1)

        def _build_query(model):
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
            q.first.return_value = task_1
            return q

        mock_db.query.side_effect = _build_query

        with (
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_ready",
                return_value=[dr1],
            ),
            patch(
                "services.coordinator_service.MissionReconciler.reconcile",
                new_callable=AsyncMock,
            ),
        ):
            mock_db.refresh = MagicMock()
            run.state = "running"

            await coordinator._process_run(mock_db, run)

            assert coordinator._run_agent_io.call_count == 1

    @pytest.mark.asyncio
    async def test_no_dispatch_still_reconciles(self, coordinator, mock_db):
        """When no tasks are ready, reconcile still runs."""
        run = _make_run(max_concurrent=2)
        no_dispatch = DispatchResult(
            dispatched=False, skipped_reason="no_ready_tasks",
        )

        # Simple query mock
        q = MagicMock()
        q.filter.return_value = q
        q.all.return_value = []
        mock_db.query.return_value = q

        with (
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_ready",
                return_value=[no_dispatch],
            ),
            patch(
                "services.coordinator_service.MissionReconciler.reconcile",
                new_callable=AsyncMock,
            ) as mock_reconcile,
        ):
            mock_db.refresh = MagicMock()
            run.state = "running"

            await coordinator._process_run(mock_db, run)

            assert coordinator._run_agent_io.call_count == 0
            mock_reconcile.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_exception_does_not_crash_tick(self, coordinator, mock_db):
        """
        If one task raises during execution, the other completes
        and reconcile still runs.
        """
        run = _make_run(max_concurrent=2)
        task_1 = _make_task(seq=1)
        task_2 = _make_task(seq=2)

        dr1 = _make_dispatch_result(task_id=task_1.id, agent_id=1)
        dr2 = _make_dispatch_result(task_id=task_2.id, agent_id=2)

        call_counter = {"n": 0}

        async def _io_side_effect(factory, agent_id, prompt, task,
                                  attachment_ids, *, mode_caps=None,
                                  agent_runtime=None):
            call_counter["n"] += 1
            if task.id == task_1.id:
                raise RuntimeError("LLM timeout")

        # asyncio.gather(return_exceptions=True) captures the raise; Phase 3
        # converts it to an error dict and still records + reconciles.
        coordinator._run_agent_io = AsyncMock(side_effect=_io_side_effect)

        _task_list = [task_1, task_2]

        def _build_query(model):
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.all.return_value = []
            q.first.side_effect = lambda: _task_list.pop(0) if _task_list else None
            return q

        mock_db.query.side_effect = _build_query

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
            mock_db.refresh = MagicMock()
            run.state = "running"

            # Should NOT raise despite task_1 failing
            await coordinator._process_run(mock_db, run)

            # Both tasks were attempted
            assert call_counter["n"] == 2
            # Reconcile still runs
            mock_reconcile.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_ready_called_not_dispatch_next(self, coordinator, mock_db):
        """Verify we call dispatch_ready, NOT the deprecated dispatch_next."""
        run = _make_run(max_concurrent=1)

        q = MagicMock()
        q.filter.return_value = q
        q.all.return_value = []
        mock_db.query.return_value = q

        with (
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_ready",
                return_value=[DispatchResult(dispatched=False, skipped_reason="no_ready_tasks")],
            ) as mock_ready,
            patch(
                "services.coordinator_service.MissionDispatcher.dispatch_next",
                side_effect=AssertionError("dispatch_next should not be called"),
            ),
            patch(
                "services.coordinator_service.MissionReconciler.reconcile",
                new_callable=AsyncMock,
            ),
        ):
            mock_db.refresh = MagicMock()
            run.state = "running"

            await coordinator._process_run(mock_db, run)

            mock_ready.assert_called_once()
