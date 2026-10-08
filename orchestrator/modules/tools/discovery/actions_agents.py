"""Agent-related ActionDefinitions (list, get, create, update, delete, heartbeat config).

PRD-256 FX-016: platform_create_agent and platform_update_agent take the agent's runtime.
They are registered inside ``register_agents_actions``, a function past the length rule;
``takes_the_runtime`` hands it a registry that adds ``runtime``, ``provider`` and ``model``
to those two schemas as they are registered (the definition is rebuilt, never changed in
place). What the handlers do with them is ``agent_runtime``. ``provider`` and ``model`` were
already read by create_agent (the API model's provider, and ``model`` as an older name of
``model_id``) and ``provider`` by update_agent: they are declared now, and their description
says what each means for each runtime.
"""

import dataclasses
import functools
from typing import Any, Callable, Dict, Tuple

from core.cli_runtime import CLI_PROVIDERS, RUNTIME_KINDS

from .action_registry import ActionDefinition, ActionRegistry

RUNTIME_DESCRIPTION = (
    "How the agent runs: 'api' (the platform's models, chosen by model_id) or 'cli' (a session of "
    "the owner's own Claude Code or Codex, on their paired machine). 'A Claude session', 'Claude Code', "
    "'a proper Claude agent' or 'runs on my machine' means runtime 'cli' with provider 'claude'. "
    "{absent}"
)
CREATE_ABSENT = "Omit it for the workspace's default runtime for new agents (api unless set)."
UPDATE_ABSENT = "Omit it to keep the agent's runtime."
PROVIDER_DESCRIPTION = (
    "For a cli agent, which CLI runs its session ({clis}): 'claude' is Claude Code, the default. "
    "For an api agent, the model_id's provider (optional)."
)
MODEL_DESCRIPTION = (
    "For a cli agent, the CLI's own model word, e.g. 'sonnet', 'opus' or 'haiku' for Claude Code "
    "(omit it for the CLI's default). An api agent's model goes in model_id."
)
DESCRIPTION_ADDS = (" Set runtime 'cli' (provider 'claude') when the user asks for a Claude session or "
                    "Claude Code agent.")
TAKES_THE_RUNTIME: Dict[str, str] = {
    "platform_create_agent": CREATE_ABSENT,
    "platform_update_agent": UPDATE_ABSENT,
}
# The keys these schemas now declare: no longer undeclared names the handler reads.
DECLARED = ("runtime", "provider", "model")


def runtime_properties(absent: str) -> Dict[str, Any]:
    """The three schema properties, ``absent`` saying what a call without a runtime gets."""
    return {
        "runtime": {"type": "string", "enum": list(RUNTIME_KINDS),
                    "description": RUNTIME_DESCRIPTION.format(absent=absent)},
        "provider": {"type": "string", "description": PROVIDER_DESCRIPTION.format(clis=", ".join(CLI_PROVIDERS))},
        "model": {"type": "string", "description": MODEL_DESCRIPTION},
    }


def _with_the_runtime(action: Any) -> Any:
    """``action`` rebuilt with the runtime properties, when it is one of the two tools."""
    absent = TAKES_THE_RUNTIME.get(getattr(action, "name", None))
    if absent is None:
        return action
    parameters = dict(action.parameters or {})
    parameters["properties"] = {**(parameters.get("properties") or {}), **runtime_properties(absent)}
    accepts: Tuple[str, ...] = tuple(key for key in action.accepts if key not in DECLARED)
    return dataclasses.replace(action, parameters=parameters, accepts=accepts,
                               description=f"{action.description}{DESCRIPTION_ADDS}")


class _TakesTheRuntime:
    """The registry ``register_agents_actions`` is given: it registers through to the real one."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def register(self, action: Any) -> None:
        self._registry.register(_with_the_runtime(action))


def takes_the_runtime(register_actions: Callable[[Any], None]) -> Callable[[Any], None]:
    """Wrap ``register_agents_actions`` (see the module docstring)."""
    @functools.wraps(register_actions)
    def wrapped(registry: Any) -> None:
        register_actions(_TakesTheRuntime(registry))
    return wrapped


@takes_the_runtime  # PRD-256 FX-016: create and update agent take runtime, provider and model
def register_agents_actions(registry: ActionRegistry) -> None:
    """Register all agent-related platform actions."""
    from core.llm.defaults import DEFAULT_LLM_MODEL

    # ── Read ─────────────────────────────────────────────────────────

    # PRD-234 S3: "who is best suited?" — the matcher's ranking with reasons, so
    # Auto proposes an informed pick instead of guessing (the human still confirms).
    registry.register(ActionDefinition(
        name="platform_recommend_agent",
        description=(
            "Rank the roster for a piece of work and explain why: skills, connected "
            "tools, model fit, availability and past outcomes (plus semantic similarity "
            "to each agent's capabilities when available). Returns the top candidates "
            "with a score, a one-line reason, their runtime ('cli' = the user's own "
            "Claude Code session on their machine; 'api' = a platform-run model) and "
            "model. Use BEFORE proposing an assignee when the user has not named one. "
            "This is advice: propose the top candidate and let the user confirm; never "
            "file a ticket for an agent the user did not agree to."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "What needs doing, in one or two sentences (the ticket's objective).",
                },
                "prefer_runtime": {
                    "type": "string",
                    "enum": ["any", "cli", "api"],
                    "description": "Restrict to Claude Code session agents ('cli'), platform-run agents ('api'), or consider all ('any', default).",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many candidates to return (default 3).",
                },
            },
            "required": ["objective"],
        },
        permission_level="read",
        tags=["agents", "read", "routing", "assign", "recommend"],
        examples=[
            "who should take the login-bug fix?",
            "which agent is best suited to summarise these documents?",
            "pick the right agent for a refactor in my repo",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_agents",
        description=(
            "List all agents in the current workspace with names, types, status, and descriptions. "
            "Use when asked about available agents or for an overview. "
            "For details about ONE specific agent, use platform_get_agent instead."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["active", "inactive", "all"],
                    "description": "Filter agents by status. Defaults to 'all'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["agents", "list", "overview"],
        examples=[
            "what agents do I have?",
            "list my agents",
            "show all agents",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_agent",
        description=(
            "Get detailed config for ONE specific agent by name or ID — model, tools, prompt, activity. "
            "Use when asked about a specific agent's setup. "
            "For listing ALL agents, use platform_list_agents instead. "
            "Provide agent_name or agent_id."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to look up.",
                },
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to look up (alternative to name).",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["agents", "details", "config"],
        examples=[
            "tell me about the DevOps agent",
            "what model does agent 5 use?",
        ],
    ))

    # ── Write ────────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_create_agent",
        description=(
            "Create a new agent in the workspace. Requires a name and agent type. "
            "Optionally accepts description, model, system prompt, temperature, tags, "
            "team, job_title, and reports_to_id. "
            "Use when the user asks to create, add, or set up a new agent."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new agent.",
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["chatbot", "worker", "researcher", "coder"],
                    "description": "Type of agent to create. Defaults to 'chatbot'.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the agent's purpose.",
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "LLM model ID, as platform_list_workspace_models lists it "
                        f"(e.g. '{DEFAULT_LLM_MODEL}'). Omit it for the default model, "
                        f"{DEFAULT_LLM_MODEL}."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "Custom system prompt that defines the agent's persona, behaviour, "
                        "and constraints. This is the instruction text the agent sees at the "
                        "start of every conversation."
                    ),
                },
                "temperature": {
                    "type": "number",
                    "description": (
                        "Sampling temperature (0.0–2.0). Lower values are more deterministic, "
                        "higher values are more creative. Defaults to 0.7."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorisation (e.g. ['support', 'customer-facing']).",
                },
                "team": {
                    "type": "string",
                    "description": "Department/team name (e.g. 'Engineering & DevOps', 'Growth & Marketing').",
                },
                "job_title": {
                    "type": "string",
                    "description": "Human-readable role title (e.g. 'Engineering Reliability & Security Lead').",
                },
                "reports_to_id": {
                    "type": "integer",
                    "description": "Agent ID of the manager this agent reports to (org hierarchy).",
                },
            },
            "required": ["name"],
        },
        permission_level="write",
        promoted=True,
        requires_confirmation=False,
        tags=["agents", "create", "write"],
        examples=[
            "create an agent called DevOps Bot",
            "make a new researcher agent",
            "create a support agent using claude sonnet with a helpful persona",
        ],
        accepts=("model", "provider"),
    ))

    registry.register(ActionDefinition(
        name="platform_update_agent",
        description=(
            "Update an existing agent's configuration. Can change name, description, "
            "status, model, system prompt, temperature, tags, team, job_title, or reports_to_id. "
            "Use when the user asks to modify, update, or reconfigure an agent. "
            "Provide agent_name or agent_id."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to update.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Current name of the agent (used to look up if no ID).",
                },
                "new_name": {
                    "type": "string",
                    "description": "New name for the agent.",
                },
                "description": {
                    "type": "string",
                    "description": "New description.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "New status.",
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "New LLM model ID, as platform_list_workspace_models lists it "
                        f"(e.g. '{DEFAULT_LLM_MODEL}'). An id the catalog does not have is refused."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "New system prompt / persona instructions for the agent."
                    ),
                },
                "temperature": {
                    "type": "number",
                    "description": "New sampling temperature (0.0–2.0).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replace the agent's tags with this list.",
                },
                "team": {
                    "type": "string",
                    "description": "Department/team name (e.g. 'Engineering & DevOps', 'Growth & Marketing').",
                },
                "job_title": {
                    "type": "string",
                    "description": "Human-readable role title (e.g. 'Engineering Reliability & Security Lead').",
                },
                "reports_to_id": {
                    "type": "integer",
                    "description": "Agent ID of the manager this agent reports to (org hierarchy).",
                },
            },
            "required": [],
        },
        permission_level="write",
        promoted=True,
        requires_confirmation=False,
        tags=["agents", "update", "write"],
        examples=[
            "rename agent 5 to CodeReview Bot",
            "deactivate the DevOps agent",
            "change the support agent's model to claude sonnet",
            "update agent 3's system prompt to be more formal",
        ],
        accepts=("provider",),
    ))

    # ── Destructive ──────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_delete_agent",
        description=(
            "Delete an agent from the workspace. This is permanent and cannot be undone. "
            "Use only when the user explicitly asks to delete or remove an agent. "
            "Provide agent_name or agent_id."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to delete.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to delete (alternative to ID).",
                },
            },
            "required": [],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["agents", "delete", "destructive"],
        examples=[
            "delete the test agent",
            "remove agent 12",
        ],
    ))

    # ── Heartbeat Configuration ──────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_configure_agent_heartbeat",
        description=(
            "Configure or update the heartbeat schedule for an agent. Controls how often "
            "the agent runs periodic checks, what it checks, active hours, and proactive "
            "behavior. Set enabled=false to disable the heartbeat entirely. "
            "Provide agent_id or agent_name."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to configure heartbeat for.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent (alternative to agent_id).",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable or disable the heartbeat. Defaults to true.",
                },
                "interval_minutes": {
                    "type": "integer",
                    "description": "How often the heartbeat runs, in minutes. Options: 15, 30, 60, 120, 240, 480 (8hr), 1440 (daily), 10080 (weekly). Defaults to 60.",
                },
                "prompt": {
                    "type": "string",
                    "description": "What the agent should check on each heartbeat tick (e.g., 'Check calendar for upcoming events').",
                },
                "auto_act": {
                    "type": "boolean",
                    "description": "Whether the agent can take action on findings or just report. Defaults to false.",
                },
                "active_hours_start": {
                    "type": "string",
                    "description": "Start of active window in HH:MM format (e.g., '08:00'). Heartbeats only run within active hours.",
                },
                "active_hours_end": {
                    "type": "string",
                    "description": "End of active window in HH:MM format (e.g., '20:00').",
                },
                "proactive_level": {
                    "type": "string",
                    "enum": ["silent", "notify", "act_notify", "autonomous"],
                    "description": "How proactive the agent should be. silent=log only, notify=report to user, act_notify=act and report, autonomous=act independently.",
                },
                "notification_channel": {
                    "type": "string",
                    "description": "Where to send heartbeat notifications (e.g., 'slack', 'email', 'in_app').",
                },
                "checklist": {
                    "type": "string",
                    "description": "Checklist of items for the agent to review each tick (newline-separated).",
                },
            },
            "required": [],
        },
        permission_level="write",
        tags=["agents", "heartbeat", "schedule", "configure"],
        examples=[
            "enable heartbeat for the communication agent every 30 minutes",
            "set agent heartbeat to check calendar every hour",
            "disable heartbeat for agent 5",
            "configure sentinel to run every 15 minutes with auto_act",
            "set active hours 9am to 6pm for the monitoring agent",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_agent_heartbeat",
        description=(
            "Read the heartbeat configuration for a specific agent. Returns "
            "every field the configure endpoint accepts (enabled, interval, "
            "prompt, checklist, auto_act, active hours, proactive_level, "
            "notification_channel) plus an is_configured flag. Use this before "
            "editing a heartbeat so the diff is informed — read current state, "
            "decide what changes, then call platform_configure_agent_heartbeat "
            "with only the fields you actually want to change. "
            "Provide agent_id or agent_name."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "integer", "description": "Agent id. Either agent_id or agent_name."},
                "agent_name": {"type": "string", "description": "Agent name (case-insensitive partial match)."},
            },
            "required": [],
        },
        permission_level="read",
        tags=["agents", "heartbeat", "read", "schedule"],
        examples=[
            "show me VECTOR's heartbeat config",
            "what's SENTINEL's heartbeat schedule?",
            "read the heartbeat for agent 188",
            "is ATLAS heartbeat enabled?",
        ],
    ))
