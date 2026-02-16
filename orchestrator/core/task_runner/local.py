"""
Local Task Runner
=================
PRD-56 Phase 1: In-process task execution via asyncio.

Wraps the current asyncio.create_task() pattern behind the TaskRunner
interface. Zero behavior change from the existing system — this is a
drop-in abstraction that enables future swap to QueuedTaskRunner or
KubernetesTaskRunner.

Task lifecycle:
  submit_task() → asyncio.create_task(execute_fn)
  get_status()  → check in-memory dict
  get_result()  → await asyncio.Event + return stored result
  cancel_task() → asyncio.Task.cancel()
  stream_updates() → yield from asyncio.Queue
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, Optional
from dataclasses import dataclass, field

from .base import TaskRunner
from .models import (
    AgentTask,
    TaskEvent,
    TaskEventType,
    TaskHandle,
    TaskResult,
    TaskStatusEnum,
)

logger = logging.getLogger(__name__)


@dataclass
class _TrackedTask:
    """Internal tracking state for an in-flight task."""
    task: AgentTask
    handle: TaskHandle
    asyncio_task: Optional[asyncio.Task] = None
    result: Optional[TaskResult] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    started_at: Optional[float] = None


class LocalTaskRunner(TaskRunner):
    """Execute agent tasks in-process using asyncio.

    This is the Phase 1 implementation. It behaves identically to the
    existing asyncio.create_task() pattern but routes through the
    TaskRunner interface for future-proofing.

    Usage:
        runner = LocalTaskRunner()

        # Register the execution function (wraps AgentFactory.execute_with_prompt)
        runner.register_executor(my_async_execute_fn)

        handle = await runner.submit_task(agent_task)
        result = await runner.get_result(handle)
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, _TrackedTask] = {}
        self._executor: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None

    @property
    def backend_name(self) -> str:
        return "local"

    def register_executor(
        self,
        executor: Callable[[AgentTask], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> None:
        """Register the async function that actually executes agent tasks.

        The executor receives an AgentTask and returns the result dict
        (same shape as AgentFactory.execute_with_prompt output).

        This is the bridge between the abstraction and your existing
        agent execution code.
        """
        self._executor = executor

    async def submit_task(self, task: AgentTask) -> TaskHandle:
        if self._executor is None:
            raise RuntimeError(
                "No executor registered. Call runner.register_executor() "
                "with an async function that executes agent tasks."
            )

        handle = TaskHandle(
            task_id=task.task_id,
            workspace_id=task.workspace_id,
            status=TaskStatusEnum.PENDING,
            runner_backend=self.backend_name,
        )

        tracked = _TrackedTask(task=task, handle=handle)
        self._tasks[task.task_id] = tracked

        # Fire off the asyncio task (same as existing pattern)
        tracked.asyncio_task = asyncio.create_task(
            self._run_task(tracked),
            name=f"task-{task.task_id[:8]}",
        )

        logger.info(
            "Task %s submitted (workspace=%s, agent=%d, type=%s, priority=%s)",
            task.task_id[:8],
            task.workspace_id,
            task.agent_id,
            task.task_type.value,
            task.priority.value,
        )

        return handle

    async def get_status(self, handle: TaskHandle) -> TaskStatusEnum:
        tracked = self._tasks.get(handle.task_id)
        if tracked is None:
            return TaskStatusEnum.FAILED
        return tracked.handle.status

    async def get_result(self, handle: TaskHandle, timeout: float = 300.0) -> TaskResult:
        tracked = self._tasks.get(handle.task_id)
        if tracked is None:
            return TaskResult(
                task_id=handle.task_id,
                status=TaskStatusEnum.FAILED,
                error="Task not found",
            )

        try:
            await asyncio.wait_for(tracked.done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Cancel the task on timeout
            await self.cancel_task(handle)
            return TaskResult(
                task_id=handle.task_id,
                status=TaskStatusEnum.TIMED_OUT,
                error=f"Task timed out after {timeout}s",
            )

        return tracked.result or TaskResult(
            task_id=handle.task_id,
            status=TaskStatusEnum.FAILED,
            error="Result unavailable",
        )

    async def cancel_task(self, handle: TaskHandle) -> bool:
        tracked = self._tasks.get(handle.task_id)
        if tracked is None:
            return False

        if tracked.asyncio_task and not tracked.asyncio_task.done():
            tracked.asyncio_task.cancel()
            tracked.handle.status = TaskStatusEnum.CANCELLED
            tracked.result = TaskResult(
                task_id=handle.task_id,
                status=TaskStatusEnum.CANCELLED,
                error="Task cancelled by user",
            )
            tracked.done_event.set()
            await self._emit_event(tracked, TaskEventType.STATUS_CHANGED, {
                "status": TaskStatusEnum.CANCELLED.value,
            })
            logger.info("Task %s cancelled", handle.task_id[:8])
            return True

        return False

    async def stream_updates(self, handle: TaskHandle) -> AsyncIterator[TaskEvent]:
        tracked = self._tasks.get(handle.task_id)
        if tracked is None:
            return

        while True:
            try:
                event = await asyncio.wait_for(tracked.event_queue.get(), timeout=1.0)
                yield event
                # Stop streaming on terminal states
                if (
                    event.event_type == TaskEventType.STATUS_CHANGED
                    and event.data.get("status") in (
                        TaskStatusEnum.COMPLETED.value,
                        TaskStatusEnum.FAILED.value,
                        TaskStatusEnum.CANCELLED.value,
                        TaskStatusEnum.TIMED_OUT.value,
                    )
                ):
                    return
            except asyncio.TimeoutError:
                # Check if task is done (event may have been missed)
                if tracked.done_event.is_set():
                    return
                continue

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        """Cancel all in-flight tasks on shutdown."""
        for tracked in self._tasks.values():
            if tracked.asyncio_task and not tracked.asyncio_task.done():
                tracked.asyncio_task.cancel()
        # Wait briefly for cancellations to propagate
        pending = [
            t.asyncio_task for t in self._tasks.values()
            if t.asyncio_task and not t.asyncio_task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        logger.info("LocalTaskRunner shutdown complete")

    # -- Internal --

    async def _run_task(self, tracked: _TrackedTask) -> None:
        """Execute the task and capture the result."""
        task = tracked.task
        start_time = time.monotonic()
        tracked.started_at = start_time

        # Transition: PENDING → RUNNING
        tracked.handle.status = TaskStatusEnum.RUNNING
        await self._emit_event(tracked, TaskEventType.STATUS_CHANGED, {
            "status": TaskStatusEnum.RUNNING.value,
        })

        try:
            raw_result = await asyncio.wait_for(
                self._executor(task),
                timeout=task.timeout_seconds,
            )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            tracked.result = TaskResult(
                task_id=task.task_id,
                status=TaskStatusEnum.COMPLETED,
                result=raw_result,
                tokens_used=raw_result.get("execution", {}).get("tokens_used", 0),
                execution_time_ms=elapsed_ms,
            )
            tracked.handle.status = TaskStatusEnum.COMPLETED

            await self._emit_event(tracked, TaskEventType.STATUS_CHANGED, {
                "status": TaskStatusEnum.COMPLETED.value,
                "execution_time_ms": elapsed_ms,
            })

            logger.info(
                "Task %s completed in %dms (tokens=%d)",
                task.task_id[:8],
                elapsed_ms,
                tracked.result.tokens_used,
            )

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            tracked.result = TaskResult(
                task_id=task.task_id,
                status=TaskStatusEnum.TIMED_OUT,
                error=f"Task exceeded {task.timeout_seconds}s timeout",
                execution_time_ms=elapsed_ms,
            )
            tracked.handle.status = TaskStatusEnum.TIMED_OUT
            await self._emit_event(tracked, TaskEventType.STATUS_CHANGED, {
                "status": TaskStatusEnum.TIMED_OUT.value,
            })
            logger.warning("Task %s timed out after %ds", task.task_id[:8], task.timeout_seconds)

        except asyncio.CancelledError:
            # Already handled in cancel_task()
            pass

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            tracked.result = TaskResult(
                task_id=task.task_id,
                status=TaskStatusEnum.FAILED,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
            tracked.handle.status = TaskStatusEnum.FAILED
            await self._emit_event(tracked, TaskEventType.ERROR, {
                "error": str(e),
            })
            await self._emit_event(tracked, TaskEventType.STATUS_CHANGED, {
                "status": TaskStatusEnum.FAILED.value,
            })
            logger.error("Task %s failed: %s", task.task_id[:8], e, exc_info=True)

        finally:
            tracked.done_event.set()

    async def _emit_event(
        self,
        tracked: _TrackedTask,
        event_type: TaskEventType,
        data: Dict[str, Any],
    ) -> None:
        """Push an event to the task's event queue."""
        event = TaskEvent(
            task_id=tracked.task.task_id,
            event_type=event_type,
            data=data,
        )
        try:
            tracked.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full for task %s, dropping event", tracked.task.task_id[:8])

    @property
    def active_task_count(self) -> int:
        """Number of tasks currently running."""
        return sum(
            1 for t in self._tasks.values()
            if t.handle.status == TaskStatusEnum.RUNNING
        )

    @property
    def total_task_count(self) -> int:
        """Total tracked tasks (all states)."""
        return len(self._tasks)

    def cleanup_completed(self, max_age_seconds: float = 3600) -> int:
        """Remove completed tasks older than max_age_seconds. Returns count removed."""
        now = time.monotonic()
        to_remove = []
        for task_id, tracked in self._tasks.items():
            if tracked.done_event.is_set() and tracked.started_at:
                age = now - tracked.started_at
                if age > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        return len(to_remove)
