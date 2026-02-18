"""
Queued Task Runner
==================
PRD-56 Phase 2 / PRD-59 US-019: Redis-backed distributed task queue.

Submits agent tasks to a Redis queue and polls for results.
Worker containers (ARQ or custom) consume tasks from the queue,
execute them, and write results back to Redis.

Task lifecycle:
  submit_task()    → LPUSH to Redis queue; status → QUEUED
  get_status()     → HGET from Redis hash
  get_result()     → poll/BLPOP result key with timeout
  cancel_task()    → set cancel flag in Redis; worker checks flag
  stream_updates() → SUBSCRIBE to task-specific Redis channel

Redis key layout:
  task:{task_id}          → JSON hash of task payload
  task:{task_id}:status   → status string
  task:{task_id}:result   → JSON result when complete
  task:{task_id}:events   → pub/sub channel for streaming
  task:{task_id}:cancel   → "1" if cancellation requested
  queue:agent_tasks       → LIST used as FIFO queue
  queue:agent_tasks:{pri} → priority queues (high, normal, low)

Requires:
  pip install redis  (already in project)
  pip install arq    (optional — for managed workers)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

from .base import TaskRunner
from .models import (
    AgentTask,
    TaskEvent,
    TaskEventType,
    TaskHandle,
    TaskResult,
    TaskStatusEnum,
    TaskPriority,
)

logger = logging.getLogger(__name__)

# Redis key prefix
_PREFIX = "taskrunner"

# Priority queue mapping
_PRIORITY_QUEUES = {
    TaskPriority.CRITICAL: f"{_PREFIX}:queue:critical",
    TaskPriority.HIGH: f"{_PREFIX}:queue:high",
    TaskPriority.NORMAL: f"{_PREFIX}:queue:normal",
    TaskPriority.LOW: f"{_PREFIX}:queue:low",
}

# Default queue (worker consumes from all priority queues)
_DEFAULT_QUEUE = f"{_PREFIX}:queue:normal"

# TTL for completed task data (2 hours)
_RESULT_TTL = 7200


def _task_key(task_id: str) -> str:
    return f"{_PREFIX}:task:{task_id}"


def _status_key(task_id: str) -> str:
    return f"{_PREFIX}:task:{task_id}:status"


def _result_key(task_id: str) -> str:
    return f"{_PREFIX}:task:{task_id}:result"


def _events_channel(task_id: str) -> str:
    return f"{_PREFIX}:task:{task_id}:events"


def _cancel_key(task_id: str) -> str:
    return f"{_PREFIX}:task:{task_id}:cancel"


class QueuedTaskRunner(TaskRunner):
    """Execute agent tasks via Redis queue + worker containers.

    The runner itself is thin — it only enqueues tasks and reads results.
    Actual execution happens in worker processes (ARQ workers or custom).

    Configuration (env vars):
        REDIS_HOST          — Redis host (default: 127.0.0.1)
        REDIS_PORT          — Redis port (default: 6379)
        REDIS_PASSWORD      — Redis password (default: None)
        REDIS_DB            — Redis DB index (default: 0)
        TASK_QUEUE_PREFIX   — Key prefix (default: taskrunner)
        TASK_RESULT_TTL     — Seconds to keep completed task data (default: 7200)
    """

    def __init__(self) -> None:
        import redis as _redis
        import redis.asyncio as _aioredis

        self._host = os.environ.get("REDIS_HOST", "127.0.0.1")
        self._port = int(os.environ.get("REDIS_PORT", "6379"))
        self._password = os.environ.get("REDIS_PASSWORD") or None
        self._db = int(os.environ.get("REDIS_DB", "0"))

        # Sync client for simple operations
        self._redis = _redis.Redis(
            host=self._host,
            port=self._port,
            password=self._password,
            db=self._db,
            decode_responses=True,
            socket_timeout=5,
        )

        # Async client for streaming / blocking operations
        self._aredis = _aioredis.Redis(
            host=self._host,
            port=self._port,
            password=self._password,
            db=self._db,
            decode_responses=True,
        )

        logger.info(
            "QueuedTaskRunner initialized (redis=%s:%d, db=%d)",
            self._host, self._port, self._db,
        )

    @property
    def backend_name(self) -> str:
        return "queued"

    async def submit_task(self, task: AgentTask) -> TaskHandle:
        """Serialize task to Redis and push to priority queue."""
        task_id = task.task_id
        payload = task.model_dump_json()

        # Store task payload as hash
        self._redis.set(_task_key(task_id), payload, ex=_RESULT_TTL * 2)

        # Set initial status
        self._redis.set(_status_key(task_id), TaskStatusEnum.QUEUED.value, ex=_RESULT_TTL * 2)

        # Push to priority queue
        queue_key = _PRIORITY_QUEUES.get(task.priority, _DEFAULT_QUEUE)
        self._redis.lpush(queue_key, task_id)

        handle = TaskHandle(
            task_id=task_id,
            workspace_id=task.workspace_id,
            status=TaskStatusEnum.QUEUED,
            runner_backend=self.backend_name,
        )

        # Publish status event
        await self._publish_event(task_id, TaskEventType.STATUS_CHANGED, {
            "status": TaskStatusEnum.QUEUED.value,
            "queue": queue_key,
        })

        logger.info(
            "Task %s queued (workspace=%s, agent=%d, priority=%s, queue=%s)",
            task_id[:8], task.workspace_id, task.agent_id,
            task.priority.value, queue_key,
        )

        return handle

    async def get_status(self, handle: TaskHandle) -> TaskStatusEnum:
        """Read status from Redis."""
        raw = self._redis.get(_status_key(handle.task_id))
        if raw is None:
            return TaskStatusEnum.FAILED
        try:
            return TaskStatusEnum(raw)
        except ValueError:
            return TaskStatusEnum.FAILED

    async def get_result(self, handle: TaskHandle, timeout: float = 300.0) -> TaskResult:
        """Poll Redis for result with exponential backoff, up to timeout."""
        deadline = time.monotonic() + timeout
        poll_interval = 0.25  # start at 250ms

        while time.monotonic() < deadline:
            raw = self._redis.get(_result_key(handle.task_id))
            if raw is not None:
                try:
                    return TaskResult.model_validate_json(raw)
                except Exception as e:
                    logger.error("Failed to parse task result: %s", e)
                    return TaskResult(
                        task_id=handle.task_id,
                        status=TaskStatusEnum.FAILED,
                        error=f"Result parse error: {e}",
                    )

            # Check for terminal status (failed, cancelled, timed_out)
            status = await self.get_status(handle)
            if status in (TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED, TaskStatusEnum.TIMED_OUT):
                return TaskResult(
                    task_id=handle.task_id,
                    status=status,
                    error=f"Task ended with status: {status.value}",
                )

            await asyncio.sleep(min(poll_interval, deadline - time.monotonic()))
            poll_interval = min(poll_interval * 1.5, 5.0)  # Cap at 5s

        # Timeout: request cancellation
        await self.cancel_task(handle)
        return TaskResult(
            task_id=handle.task_id,
            status=TaskStatusEnum.TIMED_OUT,
            error=f"Task timed out waiting for result after {timeout}s",
        )

    async def cancel_task(self, handle: TaskHandle) -> bool:
        """Set cancellation flag in Redis. Worker checks this flag periodically."""
        task_id = handle.task_id
        self._redis.set(_cancel_key(task_id), "1", ex=_RESULT_TTL)
        self._redis.set(_status_key(task_id), TaskStatusEnum.CANCELLED.value, ex=_RESULT_TTL)

        await self._publish_event(task_id, TaskEventType.STATUS_CHANGED, {
            "status": TaskStatusEnum.CANCELLED.value,
        })

        # Try to remove from queue if not yet picked up
        for queue_key in _PRIORITY_QUEUES.values():
            self._redis.lrem(queue_key, 1, task_id)

        logger.info("Task %s cancellation requested", task_id[:8])
        return True

    async def stream_updates(self, handle: TaskHandle) -> AsyncIterator[TaskEvent]:
        """Subscribe to task-specific Redis pub/sub channel for real-time events."""
        channel_name = _events_channel(handle.task_id)
        pubsub = self._aredis.pubsub()

        try:
            await pubsub.subscribe(channel_name)

            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if msg is not None and msg["type"] == "message":
                    try:
                        data = json.loads(msg["data"])
                        event = TaskEvent(
                            task_id=handle.task_id,
                            event_type=TaskEventType(data.get("event_type", "status_changed")),
                            data=data.get("data", {}),
                        )
                        yield event

                        # Stop on terminal status
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
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("Failed to parse event: %s", e)
                else:
                    # Check if task is already done (no more events coming)
                    status = await self.get_status(handle)
                    if status in (
                        TaskStatusEnum.COMPLETED,
                        TaskStatusEnum.FAILED,
                        TaskStatusEnum.CANCELLED,
                        TaskStatusEnum.TIMED_OUT,
                    ):
                        return
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()

    async def health_check(self) -> bool:
        """Verify Redis is reachable."""
        try:
            return self._redis.ping()
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Close Redis connections."""
        try:
            self._redis.close()
            await self._aredis.close()
        except Exception as e:
            logger.warning("Error during QueuedTaskRunner shutdown: %s", e)
        logger.info("QueuedTaskRunner shutdown complete")

    # -- Internal --

    async def _publish_event(
        self,
        task_id: str,
        event_type: TaskEventType,
        data: Dict[str, Any],
    ) -> None:
        """Publish event to task-specific Redis channel."""
        channel = _events_channel(task_id)
        payload = json.dumps({
            "event_type": event_type.value,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        try:
            await self._aredis.publish(channel, payload)
        except Exception as e:
            logger.warning("Failed to publish event for task %s: %s", task_id[:8], e)

    # -- Worker helpers (used by ARQ workers or custom worker process) --

    @staticmethod
    def dequeue_task_sync(redis_conn, timeout: float = 5.0) -> Optional[str]:
        """Pop next task_id from priority queues (highest first).

        This is called by worker processes. Uses BRPOP for blocking pop.
        Workers call this in a loop.
        """
        queues = [
            _PRIORITY_QUEUES[TaskPriority.CRITICAL],
            _PRIORITY_QUEUES[TaskPriority.HIGH],
            _PRIORITY_QUEUES[TaskPriority.NORMAL],
            _PRIORITY_QUEUES[TaskPriority.LOW],
        ]
        result = redis_conn.brpop(queues, timeout=int(timeout))
        if result:
            _queue_name, task_id = result
            return task_id
        return None

    @staticmethod
    def get_task_payload_sync(redis_conn, task_id: str) -> Optional[AgentTask]:
        """Retrieve task payload from Redis. Used by workers."""
        raw = redis_conn.get(_task_key(task_id))
        if raw is None:
            return None
        try:
            return AgentTask.model_validate_json(raw)
        except Exception as e:
            logger.error("Failed to parse task payload %s: %s", task_id[:8], e)
            return None

    @staticmethod
    def is_cancelled_sync(redis_conn, task_id: str) -> bool:
        """Check if task has been cancelled. Workers call this periodically."""
        return redis_conn.get(_cancel_key(task_id)) == "1"

    @staticmethod
    def write_result_sync(redis_conn, result: TaskResult) -> None:
        """Write task result to Redis. Called by workers on completion."""
        redis_conn.set(
            _result_key(result.task_id),
            result.model_dump_json(),
            ex=_RESULT_TTL,
        )
        redis_conn.set(
            _status_key(result.task_id),
            result.status.value,
            ex=_RESULT_TTL,
        )

    @staticmethod
    def publish_event_sync(
        redis_conn,
        task_id: str,
        event_type: TaskEventType,
        data: Dict[str, Any],
    ) -> None:
        """Publish event from worker process (sync version)."""
        channel = _events_channel(task_id)
        payload = json.dumps({
            "event_type": event_type.value,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        redis_conn.publish(channel, payload)
