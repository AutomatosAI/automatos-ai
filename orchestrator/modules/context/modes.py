"""
ContextMode enum and mode configurations.

Each mode declares which sections it needs, how tools are loaded,
and optional constraints like max token budgets.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContextMode(str, Enum):
    CHATBOT = "chatbot"
    TASK_EXECUTION = "task_execution"
    HEARTBEAT_ORCHESTRATOR = "heartbeat_orchestrator"
    HEARTBEAT_AGENT = "heartbeat_agent"
    RECIPE = "recipe"
    ROUTER = "router"
    ORCHESTRATOR_STAGE = "orchestrator_stage"
    NL2SQL = "nl2sql"


@dataclass(frozen=True)
class ModeConfig:
    """Declarative configuration for a context mode."""

    sections: list[str] = field(default_factory=list)
    tool_loading: str = "none"          # full, filtered, dispatcher_only, none
    personality: bool = False
    max_tokens: Optional[int] = None    # None = use model default


MODE_CONFIGS: dict[ContextMode, ModeConfig] = {
    ContextMode.CHATBOT: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "memory",
            "datetime_context", "conversation",
        ],
        tool_loading="filtered",
        personality=True,
        max_tokens=None,
    ),
    ContextMode.TASK_EXECUTION: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "memory",
            "task_context", "datetime_context", "conversation",
        ],
        tool_loading="full",
        personality=False,
        max_tokens=None,
    ),
    ContextMode.HEARTBEAT_ORCHESTRATOR: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "task_context",
            "datetime_context",
        ],
        tool_loading="dispatcher_only",
        personality=False,
        max_tokens=8000,
    ),
    # NOTE: memory intentionally excluded from HEARTBEAT_AGENT to keep context lean.
    # No memory section also means no daily logs. Heartbeat agents are stateless by
    # design — add "memory" here if agents need cross-run learning.
    # See PRD-81 Task 3.5 / Task 5.5.
    ContextMode.HEARTBEAT_AGENT: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "task_context", "datetime_context",
        ],
        tool_loading="full",
        personality=False,
        max_tokens=128000,
    ),
    ContextMode.RECIPE: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "memory",
            "recipe_context", "datetime_context",
        ],
        tool_loading="full",
        personality=False,
        max_tokens=None,
    ),
    ContextMode.ROUTER: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
    ContextMode.ORCHESTRATOR_STAGE: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
    ContextMode.NL2SQL: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
}
