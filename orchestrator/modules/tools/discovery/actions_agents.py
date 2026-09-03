"""Agent-related ActionDefinitions (list, get, create, update, delete, heartbeat config)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_agents_actions(registry: ActionRegistry) -> None:
    """Register all agent-related platform actions."""

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
                        "LLM model ID to use. Examples: 'gpt-4o', 'gpt-4o-mini', "
                        "'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'. "
                        "Defaults to 'gpt-4o' if not specified."
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
                        "New LLM model ID (e.g. 'gpt-4o', 'claude-sonnet-4-20250514')."
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
