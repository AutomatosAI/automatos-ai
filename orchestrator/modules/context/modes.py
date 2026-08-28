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
    PLANNING = "planning"


# PRD-229: ``ask_orchestrator`` (a worker asks the orchestrator a clarification)
# is admitted ONLY in the worker execution surface. It is nonsensical — and must
# be absent — everywhere else: in CHATBOT the user IS talking to Auto directly,
# and the orchestrator tick / planners are the orchestrator, not workers under
# it. The ladder is vertical (worker → orchestrator → human), never lateral, so
# this is the one context-scoped platform tool. Both the callable tool surface
# (ContextService._load_tools) and the prompt action catalog
# (PlatformActionsSection) strip these names outside the execution modes.
EXECUTION_ONLY_TOOLS: frozenset[str] = frozenset({"ask_orchestrator"})

# The modes that ARE worker execution lanes and may see EXECUTION_ONLY_TOOLS.
# TASK_EXECUTION is the coordinator's mission-task lane (the PRD-229 anchor).
EXECUTION_TOOL_MODES: frozenset["ContextMode"] = frozenset({ContextMode.TASK_EXECUTION})


def excluded_tool_names(mode: Optional["ContextMode"]) -> frozenset[str]:
    """Tool names that must NOT appear in *mode*'s surface (PRD-229).

    Empty for the execution lanes; ``EXECUTION_ONLY_TOOLS`` everywhere else.
    Accepts the enum or its string value (both flow through the context stack).
    """
    resolved = mode
    if isinstance(mode, str) and not isinstance(mode, ContextMode):
        try:
            resolved = ContextMode(mode)
        except ValueError:
            resolved = None
    return frozenset() if resolved in EXECUTION_TOOL_MODES else EXECUTION_ONLY_TOOLS


def strip_actions_from_surface(
    tools: list, excluded: "frozenset[str]"
) -> list:
    """Remove *excluded* action names from an assembled tool surface (PRD-229).

    Two removal shapes, rebuild-don't-mutate:
      * a first-class tool schema whose ``function.name`` is excluded → dropped;
      * the ``platform_execute`` dispatcher → its ``action.enum`` is rebuilt
        without the excluded names (a dispatched action not in the enum is not
        offered to the model).
    Pure: never mutates the input tools; returns a new list, copying only the
    dicts it changes.
    """
    if not excluded:
        return tools
    result: list = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if name in excluded:
            continue  # drop a first-class execution-only tool
        if name == "platform_execute":
            result.append(_prune_dispatcher_enum(tool, excluded))
            continue
        result.append(tool)
    return result


def _prune_dispatcher_enum(dispatcher: dict, excluded: "frozenset[str]") -> dict:
    """Return a copy of the platform_execute schema with excluded names dropped
    from ``parameters.properties.action.enum``. No-op copy if there is no enum."""
    fn = dispatcher.get("function", {})
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    action = props.get("action", {})
    enum = action.get("enum")
    if not isinstance(enum, list):
        return dispatcher
    pruned = [n for n in enum if n not in excluded]
    if pruned == enum:
        return dispatcher
    return {
        **dispatcher,
        "function": {
            **fn,
            "parameters": {
                **params,
                "properties": {
                    **props,
                    "action": {**action, "enum": pruned},
                },
            },
        },
    }


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
            "identity", "onboarding", "skills", "composio", "plugins",
            "platform_actions", "memory", "business_graph",
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
            "platform_actions", "memory", "business_graph",
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
    # PRD-179 S1 (F021 read-half): the workspace-scoped "field_memory" digest
    # gives recurring agents the cross-run learning they were previously blind to
    # (patterns earlier missions accumulated). User-memory / daily logs stay out
    # to keep the tick lean — this is the durable-field slice, not chat memory.
    ContextMode.HEARTBEAT_AGENT: ModeConfig(
        sections=[
            "identity", "skills", "composio", "plugins",
            "platform_actions", "task_context", "field_memory", "datetime_context",
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
    # personality=False: the planning pack (PRD-164 S1, Q61) is injected into
    # ANOTHER prompt, never sent to an LLM on its own. One section list for
    # every planner — MissionPlanner, board plan_task, AutoBrain — assembled
    # exclusively by ContextService.build_planning_context (the one assembler).
    # planning_history has priority 2 so recorded failures survive budget
    # pressure (the learning demo). PRD-179 S1 (F021 read-half): "field_memory"
    # adds the workspace-scoped field digest so a completed mission's promoted
    # distillation reaches the next mission's plan — the compounding arm PRD-164
    # left open (documents + KG but never the field).
    ContextMode.PLANNING: ModeConfig(
        sections=[
            # The full planner pack. Consumed ONLY by the real planners now —
            # MissionPlanner + board plan_task (they build execution plans and
            # want doc grounding). The hot-path AutoBrain classifier no longer
            # uses this pack (it takes a cheap roster instead), so its ~80-113s
            # cost is off the per-message path. (Restored planning_knowledge that
            # PR #517 had stripped from the shared pack, which had also removed
            # the planners' doc grounding as an unintended side effect.)
            "planning_knowledge", "planning_history",
            "business_graph", "field_memory", "agent_roster",
        ],
        tool_loading="none",
        personality=False,
        max_tokens=None,
    ),
}
