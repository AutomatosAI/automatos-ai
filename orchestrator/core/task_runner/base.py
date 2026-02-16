"""
TaskRunner Abstract Base Class
==============================
PRD-56: Infrastructure Scaling & Ephemeral Agent Compute

The core abstraction that decouples task dispatch from task execution.
All agent work flows through this interface. Swap implementations
without touching business logic:

  LocalTaskRunner      → asyncio in-process (Phase 1, Railway)
  QueuedTaskRunner     → Redis queue + worker containers (Phase 2)
  KubernetesTaskRunner → K8s Jobs with ephemeral pods (Phase 3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .models import AgentTask, TaskEvent, TaskHandle, TaskResult, TaskStatusEnum


class TaskRunner(ABC):
    """Abstract interface for agent task execution.

    Every TaskRunner implementation must support:
    - Asynchronous task submission (non-blocking)
    - Status polling
    - Result retrieval (blocking with timeout)
    - Cancellation
    - Real-time event streaming

    The runner is workspace-aware: all tasks carry a workspace_id,
    enabling per-workspace isolation, resource limits, and auditing.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier for this runner backend (e.g. 'local', 'queued', 'kubernetes')."""

    @abstractmethod
    async def submit_task(self, task: AgentTask) -> TaskHandle:
        """Submit a task for execution. Returns immediately with a handle.

        The task may not start executing immediately — it could be queued
        (Phase 2) or waiting for pod scheduling (Phase 3).
        """

    @abstractmethod
    async def get_status(self, handle: TaskHandle) -> TaskStatusEnum:
        """Get current task status. Non-blocking."""

    @abstractmethod
    async def get_result(self, handle: TaskHandle, timeout: float = 300.0) -> TaskResult:
        """Retrieve task result. Blocks until complete, failed, or timeout.

        Args:
            handle: Task reference from submit_task()
            timeout: Max seconds to wait. Raises TimeoutError if exceeded.
        """

    @abstractmethod
    async def cancel_task(self, handle: TaskHandle) -> bool:
        """Request task cancellation.

        Returns True if cancellation was accepted. The task may still
        take time to actually stop (graceful shutdown).
        """

    @abstractmethod
    async def stream_updates(self, handle: TaskHandle) -> AsyncIterator[TaskEvent]:
        """Stream real-time task progress events.

        Yields TaskEvent objects as the task executes. The stream ends
        when the task reaches a terminal state (completed, failed, cancelled).

        Maps to the existing SSE streaming pattern used by
        WorkflowStreamManager.broadcast_event().
        """

    async def health_check(self) -> bool:
        """Check if the runner backend is healthy and accepting tasks.

        Default returns True. Override for runners that depend on
        external infrastructure (Redis, K8s API).
        """
        return True

    async def shutdown(self) -> None:
        """Graceful shutdown. Wait for in-flight tasks, release resources.

        Default is a no-op. Override for runners with cleanup needs.
        """
        pass
