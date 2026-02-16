"""
Task Runner Package
===================
PRD-56: Infrastructure Scaling & Ephemeral Agent Compute

Provides a backend-agnostic interface for agent task execution.
The TaskRunner abstraction decouples task dispatch (API/orchestrator)
from task execution (local, queued workers, K8s pods).

Quick start:
    from core.task_runner import get_task_runner, AgentTask

    runner = get_task_runner()
    handle = await runner.submit_task(AgentTask(
        workspace_id=workspace_id,
        agent_id=agent.id,
        prompt="Fix the authentication bug",
        tools=["research", "file_ops", "shell"],
    ))
    result = await runner.get_result(handle)
"""

from .base import TaskRunner
from .factory import get_task_runner, reset_task_runner
from .local import LocalTaskRunner
from .models import (
    AgentTask,
    TaskEvent,
    TaskEventType,
    TaskHandle,
    TaskPriority,
    TaskResources,
    TaskResult,
    TaskStatusEnum,
    TaskType,
)

__all__ = [
    # Core interface
    "TaskRunner",
    "LocalTaskRunner",
    "get_task_runner",
    "reset_task_runner",
    # Models
    "AgentTask",
    "TaskEvent",
    "TaskEventType",
    "TaskHandle",
    "TaskPriority",
    "TaskResources",
    "TaskResult",
    "TaskStatusEnum",
    "TaskType",
]
