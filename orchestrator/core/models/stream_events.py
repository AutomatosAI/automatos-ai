"""
Streaming Event Types — PRD-123 Pattern #11
============================================

Typed SSE events for real-time transparency in chat and orchestration.
Extends the existing AI SDK Data Stream format with structured event types.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class StreamEventType(str, Enum):
    """Typed event categories for SSE streaming."""

    # Existing event types (backward compatible)
    TOKEN = "token"
    THINKING = "thinking"
    TOOL_CALL = "tool-call"
    DONE = "done"

    # Agent lifecycle
    AGENT_ASSIGNED = "agent-assigned"
    AGENT_THINKING = "agent-thinking"

    # Tool lifecycle
    TOOL_RESOLVED = "tool-resolved"
    TOOL_PERMISSION_DENIED = "tool-permission-denied"
    TOOL_EXECUTING = "tool-executing"
    TOOL_RESULT = "tool-result"

    # Memory operations
    MEMORY_INJECTED = "memory-injected"
    MEMORY_STORED = "memory-stored"

    # Context management
    CONTEXT_COMPACTED = "context-compacted"

    # Budget and limits
    BUDGET_WARNING = "budget-warning"

    # Orchestration state
    TASK_STATE_CHANGE = "task-state-change"
    MISSION_STOP = "mission-stop"


@dataclass(frozen=True)
class StreamEvent:
    """
    Immutable SSE event with type, optional content, and metadata.

    Serializes to AI SDK Data Stream format for frontend consumption.
    """

    type: StreamEventType
    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sse(self) -> str:
        """Serialize to AI SDK Data Stream format: d:{json}\\n"""
        payload: dict[str, Any] = {"type": self.type.value}
        if self.content is not None:
            payload["content"] = self.content
        if self.metadata is not None:
            payload["data"] = self.metadata
        payload["timestamp"] = self.timestamp.isoformat()
        return f"d:{json.dumps(payload)}\n"

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.content is not None:
            result["content"] = self.content
        if self.metadata is not None:
            result["data"] = self.metadata
        return result
