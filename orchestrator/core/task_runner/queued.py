"""
Queued Task Runner
==================
PRD-56 Phase 2: Redis queue + workspace worker containers.

Tasks are serialized to Redis queues (priority-based), consumed by
workspace-worker ARQ processes, and results are returned via Redis
pub/sub + hash storage.

Architecture:
  API → QueuedTaskRunner.submit_task() → Redis queue
  Worker → ARQ consumer → execute → Redis result + pub/sub event
  API → QueuedTaskRunner.get_result() → Redis hash lookup
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

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

# Queue names by priority (highest first)
QUEUE_NAMES = {
    "critical": "workspace:tasks:critical",
    "high": "workspace:tasks:high",
    "normal": "workspace:tasks:normal",
    "low": "workspace:tasks:low",
}
DEAD_LETTER_QUEUE = "workspace:tasks:dead"

# Redis key patterns
TASK_STATUS_KEY = "workspace:task:{task_id}:status"
TASK_RESULT_KEY = "workspace:task:{task_id}:result"
TASK_EVENTS_CHANNEL = "workspace:task:{task_id}:events"
WS_ACTIVE_TASKS_KEY = "workspace:ws:{workspace_id}:active_tasks"
WS_TASK_COUNT_KEY = "workspace:ws:{workspace_id}:task_count"

# TTLs
RESULT_TTL_SECONDS = 3600       # 1 hour
STATUS_TTL_SECONDS = 7200       # 2 hours


class QueuedTaskRunner(TaskRunner):
    """Execute agent tasks via Redis queue + workspace worker containers.

    Phase 2 implementation. Tasks are enqueued to priority-based Redis
    lists. Workspace worker processes consume tasks, execute them in
    isolated workspace directories, and write results back to Redis.

    Usage:
        runner = QueuedTaskRunner(redis_url="redis://localhost:6379/0")
        handle = await runner.submit_task(agent_task)
        result = await runner.get_result(handle)
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        import os
        self._redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis = None
        self._pubsub_connections: Dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return "queued"

    async def _get_redis(self):
        """Lazy-init Redis connection."""
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return self._redis

    async def submit_task(self, task: AgentTask) -> TaskHandle:
        """Serialize task to Redis queue. Returns immediately."""
        redis = await self._get_redis()

        handle = TaskHandle(
            task_id=task.task_id,
            workspace_id=task.workspace_id,
            status=TaskStatusEnum.QUEUED,
            runner_backend=self.backend_name,
        )

        # Serialize task payload
        payload = {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "workspace_id": str(task.workspace_id),
            "agent_id": task.agent_id,
            "prompt": task.prompt,
            "system_prompt": task.system_prompt,
            "tools": task.tools,
            "context": task.context,
            "priority": task.priority.value,
            "timeout_seconds": task.timeout_seconds,
            "max_retries": task.max_retries,
            "resources": task.resources.model_dump(),
            "parent_execution_id": task.parent_execution_id,
            "correlation_id": task.correlation_id,
            "created_at": task.created_at.isoformat(),
        }

        # Write initial status
        status_key = TASK_STATUS_KEY.format(task_id=task.task_id)
        await redis.hset(status_key, mapping={
            "status": TaskStatusEnum.QUEUED.value,
            "workspace_id": str(task.workspace_id),
            "submitted_at": handle.submitted_at.isoformat(),
            "priority": task.priority.value,
            "task_type": task.task_type.value,
        })
        await redis.expire(status_key, STATUS_TTL_SECONDS)

        # Track active tasks for workspace concurrency limits
        ws_active_key = WS_ACTIVE_TASKS_KEY.format(workspace_id=task.workspace_id)
        await redis.sadd(ws_active_key, task.task_id)

        ws_count_key = WS_TASK_COUNT_KEY.format(workspace_id=task.workspace_id)
        await redis.incr(ws_count_key)
        await redis.expire(ws_count_key, STATUS_TTL_SECONDS)

        # Enqueue to priority queue (LPUSH for FIFO with RPOP)
        queue_name = QUEUE_NAMES.get(task.priority.value, QUEUE_NAMES["normal"])
        await redis.lpush(queue_name, json.dumps(payload))

        # Publish submission event
        events_channel = TASK_EVENTS_CHANNEL.format(task_id=task.task_id)
        await redis.publish(events_channel, json.dumps({
            "event_type": TaskEventType.STATUS_CHANGED.value,
            "task_id": task.task_id,
            "data": {"status": TaskStatusEnum.QUEUED.value},
        }))

        logger.info(
            "Task %s queued (workspace=%s, priority=%s, queue=%s)",
            task.task_id[:8], task.workspace_id, task.priority.value, queue_name,
        )

        return handle

    async def get_status(self, handle: TaskHandle) -> TaskStatusEnum:
        """Read current status from Redis hash."""
        redis = await self._get_redis()
        status_key = TASK_STATUS_KEY.format(task_id=handle.task_id)
        status_str = await redis.hget(status_key, "status")

        if status_str is None:
            return TaskStatusEnum.FAILED
        return TaskStatusEnum(status_str)

    async def get_result(self, handle: TaskHandle, timeout: float = 300.0) -> TaskResult:
        """Poll Redis for result until available or timeout."""
        redis = await self._get_redis()
        result_key = TASK_RESULT_KEY.format(task_id=handle.task_id)

        deadline = time.monotonic() + timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            result_json = await redis.get(result_key)
            if result_json:
                data = json.loads(result_json)
                return TaskResult(
                    task_id=handle.task_id,
                    status=TaskStatusEnum(data["status"]),
                    result=data.get("result"),
                    error=data.get("error"),
                    tokens_used=data.get("tokens_used", 0),
                    execution_time_ms=data.get("execution_time_ms", 0),
                    started_at=data.get("started_at"),
                    completed_at=data.get("completed_at"),
                )

            # Check if task failed without result
            status = await self.get_status(handle)
            if status in (TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED, TaskStatusEnum.TIMED_OUT):
                status_key = TASK_STATUS_KEY.format(task_id=handle.task_id)
                error = await redis.hget(status_key, "error")
                return TaskResult(
                    task_id=handle.task_id,
                    status=status,
                    error=error or f"Task {status.value}",
                )

            await asyncio.sleep(poll_interval)
            # Back off slightly
            poll_interval = min(poll_interval * 1.2, 2.0)

        return TaskResult(
            task_id=handle.task_id,
            status=TaskStatusEnum.TIMED_OUT,
            error=f"Result polling timed out after {timeout}s",
        )

    async def cancel_task(self, handle: TaskHandle) -> bool:
        """Mark task as cancelled in Redis. Worker checks before executing."""
        redis = await self._get_redis()
        status_key = TASK_STATUS_KEY.format(task_id=handle.task_id)
        current = await redis.hget(status_key, "status")

        if current in (TaskStatusEnum.COMPLETED.value, TaskStatusEnum.FAILED.value):
            return False

        await redis.hset(status_key, "status", TaskStatusEnum.CANCELLED.value)

        # Publish cancellation event (worker listens)
        events_channel = TASK_EVENTS_CHANNEL.format(task_id=handle.task_id)
        await redis.publish(events_channel, json.dumps({
            "event_type": TaskEventType.STATUS_CHANGED.value,
            "task_id": handle.task_id,
            "data": {"status": TaskStatusEnum.CANCELLED.value},
        }))

        # Cleanup workspace tracking
        ws_active_key = WS_ACTIVE_TASKS_KEY.format(workspace_id=handle.workspace_id)
        await redis.srem(ws_active_key, handle.task_id)

        logger.info("Task %s cancel requested", handle.task_id[:8])
        return True

    async def stream_updates(self, handle: TaskHandle) -> AsyncIterator[TaskEvent]:
        """Subscribe to Redis pub/sub channel for real-time events."""
        redis = await self._get_redis()
        events_channel = TASK_EVENTS_CHANNEL.format(task_id=handle.task_id)

        pubsub = redis.pubsub()
        await pubsub.subscribe(events_channel)

        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=2.0
                )
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    event = TaskEvent(
                        task_id=handle.task_id,
                        event_type=TaskEventType(data["event_type"]),
                        data=data.get("data", {}),
                    )
                    yield event

                    # Stop on terminal states
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

                # Check if task already completed (missed events)
                status = await self.get_status(handle)
                if status in (
                    TaskStatusEnum.COMPLETED,
                    TaskStatusEnum.FAILED,
                    TaskStatusEnum.CANCELLED,
                    TaskStatusEnum.TIMED_OUT,
                ):
                    return

        finally:
            await pubsub.unsubscribe(events_channel)
            await pubsub.close()

    async def health_check(self) -> bool:
        """Verify Redis is reachable."""
        try:
            redis = await self._get_redis()
            await redis.ping()
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        logger.info("QueuedTaskRunner shutdown complete")

    # -- Workspace concurrency helpers (used by API for quota checks) --

    async def get_workspace_active_count(self, workspace_id: str) -> int:
        """Number of active (queued + running) tasks for a workspace."""
        redis = await self._get_redis()
        ws_active_key = WS_ACTIVE_TASKS_KEY.format(workspace_id=workspace_id)
        return await redis.scard(ws_active_key)

    async def cleanup_workspace_tracking(self, workspace_id: str, task_id: str) -> None:
        """Remove a completed task from workspace active tracking. Called by worker."""
        redis = await self._get_redis()
        ws_active_key = WS_ACTIVE_TASKS_KEY.format(workspace_id=workspace_id)
        await redis.srem(ws_active_key, task_id)
