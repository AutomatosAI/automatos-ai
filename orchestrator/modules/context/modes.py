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
    COORDINATOR = "coordinator"


@dataclass(frozen=True)
class ModeConfig:
    """Declarative configuration for a context mode."""

    sections: list[str] = field(default_factory=list)
    tool_loading: str = "none"          # full, filtered, dispatcher_only, none
    personality: bool = False
    max_tokens: Optional[int] = None    # None = use model default


MODE_CONFIGS: dict[ContextMode, ModeConfig] = {
    # personality=True: CHATBOT is the only user-facing conversational mode.
    # AutomatosPersonality.get_base_system_prompt() produces chatbot-specific
    # content (greetings, conversation awareness, "never show code" rules) that
    # is inappropriate for task-executing or orchestration agents.
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
    # personality=False: task agents should be professional/neutral regardless
    # of workspace personality settings. Identity + persona provide sufficient
    # agent identity without chatbot-specific tone. See PRD-81 Task 5.1.
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
    # personality=False: orchestrator tick is internal coordination, not
    # user-facing. Neutral tone keeps dispatcher prompts lean.
    ContextMode.HEARTBEAT_ORCHESTRATOR: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "task_context",
            "datetime_context",
        ],
        tool_loading="dispatcher_only",
        personality=False,
        max_tokens=8000,
    ),
    # personality=False: heartbeat agents execute scheduled tasks autonomously.
    # NOTE: memory intentionally excluded to keep context lean. No memory
    # section also means no daily logs. Heartbeat agents are stateless by
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
    # personality=False: recipes are multi-step automation pipelines.
    # Professional/neutral tone ensures consistent output across steps.
    ContextMode.RECIPE: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "memory",
            "playbook_context", "datetime_context",
        ],
        tool_loading="full",
        personality=False,
        max_tokens=None,
    ),
    # personality=False: internal routing — no user-facing output.
    ContextMode.ROUTER: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
    # personality=False: internal orchestration stage — no user-facing output.
    ContextMode.ORCHESTRATOR_STAGE: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
    # personality=False: SQL generation — precision over personality.
    ContextMode.NL2SQL: ModeConfig(
        sections=["identity", "datetime_context"],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
    # personality=False: coordinator is internal orchestration — decomposes goals,
    # dispatches tasks, reconciles state. Needs mission context + agent roster
    # to plan and dispatch effectively. 128k budget for full mission context +
    # agent roster + task history. PRD-82A Section 12, Phase 3.
    ContextMode.COORDINATOR: ModeConfig(
        sections=[
            "identity", "mission_context", "agent_roster",
            "platform_actions", "task_context", "datetime_context",
        ],
        tool_loading="full",
        personality=False,
        max_tokens=131072,
    ),
}
