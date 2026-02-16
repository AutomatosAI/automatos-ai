"""
Task Runner Data Models
=======================
PRD-56: Infrastructure Scaling & Ephemeral Agent Compute

Pydantic models for the task lifecycle: submission, tracking, results, events.
All models are workspace-scoped and backend-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatusEnum(str, Enum):
    """Task lifecycle states. Phases 2/3 add QUEUED and SCHEDULED."""
    PENDING = "pending"
    QUEUED = "queued"           # Phase 2: sitting in Redis queue
    SCHEDULED = "scheduled"     # Phase 3: K8s pod scheduling
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    WORKFLOW_SUBTASK = "workflow_subtask"
    CHAT_AGENT = "chat_agent"
    RECIPE_STEP = "recipe_step"
    BACKGROUND_JOB = "background_job"


class TaskEventType(str, Enum):
    STATUS_CHANGED = "status_changed"
    PROGRESS_UPDATE = "progress_update"
    LOG_LINE = "log_line"
    TOKEN_USAGE = "token_usage"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class TaskResources(BaseModel):
    """Resource requirements for task execution.

    Informational in Phase 1 (LocalTaskRunner ignores them).
    Enforced in Phase 2 (Docker limits) and Phase 3 (K8s resource requests/limits).
    """
    cpu_millicores: int = 500       # 0.5 CPU default
    memory_mb: int = 512            # 512 MB default
    disk_mb: int = 1024             # 1 GB scratch space
    gpu: bool = False
    network_access: bool = True
    repo_clone: bool = False        # Needs git clone capability


class AgentTask(BaseModel):
    """A unit of agent work to be executed by a TaskRunner."""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: TaskType = TaskType.WORKFLOW_SUBTASK
    workspace_id: UUID

    # What to execute
    agent_id: int
    prompt: str
    system_prompt: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Execution parameters
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 300      # 5 min default
    max_retries: int = 2

    # Resource requirements (enforced Phase 2+)
    resources: TaskResources = Field(default_factory=TaskResources)

    # Tracing / lineage
    parent_execution_id: Optional[int] = None
    correlation_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskHandle(BaseModel):
    """Lightweight reference to a submitted task. Returned immediately by submit_task()."""
    task_id: str
    workspace_id: UUID
    status: TaskStatusEnum = TaskStatusEnum.PENDING
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    runner_backend: str = "local"   # "local", "queued", "kubernetes"


class TaskResult(BaseModel):
    """Outcome of a completed (or failed) task."""
    task_id: str
    status: TaskStatusEnum
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskEvent(BaseModel):
    """Real-time event emitted during task execution. Maps to existing SSE pattern."""
    task_id: str
    event_type: TaskEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
