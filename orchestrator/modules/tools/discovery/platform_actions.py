"""
Platform Action Definitions (PRD-64)
=====================================

Curated set of platform actions that Auto can execute.
These are the operations Auto can perform on the Automatos platform itself.

Phase 1: 10 read-only actions
Phase 2: Write actions (added in commit 6)
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_all_actions(registry: ActionRegistry) -> None:
    """Register all platform actions with the registry."""
    _register_read_actions(registry)
    _register_write_actions(registry)
    _register_infra_actions(registry)


def _register_read_actions(registry: ActionRegistry) -> None:
    """Register read-only platform actions."""

    # ── Agents ──────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_agents",
        description=(
            "List all agents in the current workspace. Returns agent names, types, "
            "status, and brief descriptions. Use when the user asks about their agents, "
            "what agents exist, or wants an overview of available agents."
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
            "Get detailed information about a specific agent by name or ID. "
            "Returns configuration, assigned tools, model settings, and recent activity. "
            "Use when the user asks about a specific agent's details or configuration."
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
        tags=["agents", "details", "config"],
        examples=[
            "tell me about the DevOps agent",
            "what model does agent 5 use?",
        ],
    ))

    # ── Recipes / Workflows ─────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_recipes",
        description=(
            "List all workflow recipes in the current workspace. Returns recipe names, "
            "trigger types, status, and step counts. Use when the user asks about their "
            "recipes, workflows, or automations."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["active", "inactive", "all"],
                    "description": "Filter recipes by status. Defaults to 'all'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["recipes", "workflows", "automations"],
        examples=[
            "what recipes do I have?",
            "list my workflows",
            "show automations",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_recipe",
        description=(
            "Get detailed information about a specific recipe including its steps, "
            "trigger configuration, and execution history. Use when the user asks "
            "about a specific recipe's details."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Name of the recipe to look up.",
                },
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe (alternative to name).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["recipes", "details", "steps"],
        examples=[
            "show me the Jira Bug Triage recipe",
            "what does recipe 3 do?",
        ],
    ))

    # ── Analytics ───────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_llm_usage",
        description=(
            "Get LLM usage statistics for the workspace over a time period. "
            "Returns total requests, tokens consumed, and model breakdown. "
            "Use when the user asks about token usage, API calls, or LLM activity."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Defaults to 30.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "usage", "tokens", "llm"],
        examples=[
            "what's my token usage?",
            "how many API calls this month?",
            "show LLM usage stats",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_cost_breakdown",
        description=(
            "Get cost breakdown by model and agent for the workspace. "
            "Shows estimated costs based on token usage and model pricing. "
            "Use when the user asks about costs, spending, or budget."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Defaults to 30.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["model", "agent", "day"],
                    "description": "How to group the cost breakdown. Defaults to 'model'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "costs", "spending", "budget"],
        examples=[
            "what are my costs?",
            "how much am I spending on LLM?",
            "cost breakdown by agent",
        ],
    ))

    # ── Documents ───────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_documents",
        description=(
            "List documents uploaded to the workspace knowledge base. "
            "Returns document names, types, sizes, and processing status. "
            "Use when the user asks about their uploaded documents or files."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return. Defaults to 50.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "files", "knowledge"],
        examples=[
            "what documents have I uploaded?",
            "list my files",
            "show knowledge base documents",
        ],
    ))

    # ── Workspace ───────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_workspace_info",
        description=(
            "Get information about the current workspace including name, "
            "member count, creation date, and configuration summary. "
            "Use when the user asks about their workspace or account."
        ),
        category="workspace",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["workspace", "account", "info"],
        examples=[
            "what workspace am I in?",
            "show workspace info",
        ],
    ))

    # ── Memory ──────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_memory_stats",
        description=(
            "Get memory system statistics for the workspace. Shows total memories "
            "stored, memory types, and storage usage. Use when the user asks about "
            "memory, stored context, or what the system remembers."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["memory", "context", "storage"],
        examples=[
            "how many memories do you have?",
            "what do you remember?",
            "memory stats",
        ],
    ))

    # ── Integrations ────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_connected_apps",
        description=(
            "List external apps connected to the workspace via Composio. "
            "Shows app names, connection status, and available actions. "
            "Use when the user asks about their integrations or connected services."
        ),
        category="integrations",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["integrations", "apps", "composio", "connections"],
        examples=[
            "what apps are connected?",
            "list my integrations",
            "show connected services",
        ],
    ))


def _register_write_actions(registry: ActionRegistry) -> None:
    """Register write platform actions (PRD-64 Phase 2)."""

    # ── Agent Write Actions ─────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_create_agent",
        description=(
            "Create a new agent in the workspace. Requires a name and agent type. "
            "Optionally accepts a description, model, system prompt, temperature, and tags. "
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
            },
            "required": ["name"],
        },
        permission_level="write",
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
            "status, model, system prompt, temperature, or tags. "
            "Use when the user asks to modify, update, or reconfigure an agent."
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
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["agents", "update", "write"],
        examples=[
            "rename agent 5 to CodeReview Bot",
            "deactivate the DevOps agent",
            "change the support agent's model to claude sonnet",
            "update agent 3's system prompt to be more formal",
        ],
    ))

    # ── Recipe Write Actions ────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_create_recipe",
        description=(
            "Create a new workflow recipe in the workspace. Requires a name and "
            "description. The recipe starts as a draft with no steps — steps can "
            "be added later. Use when the user asks to create a new recipe or automation."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new recipe.",
                },
                "description": {
                    "type": "string",
                    "description": "What the recipe does.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization.",
                },
            },
            "required": ["name", "description"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "create", "write"],
        examples=[
            "create a recipe for daily standup summaries",
            "make an automation for code review",
        ],
    ))

    # ── Memory Write Actions ────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_store_memory",
        description=(
            "Store a piece of information in the workspace memory system. "
            "Use when the user explicitly asks Auto to remember something."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember.",
                },
            },
            "required": ["content"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["memory", "store", "write"],
        examples=[
            "remember that our deploy day is Thursday",
            "store this: API key rotates every 90 days",
        ],
    ))

    # ── Destructive Actions ─────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_delete_agent",
        description=(
            "Delete an agent from the workspace. This is permanent and cannot be undone. "
            "Use only when the user explicitly asks to delete or remove an agent."
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


def _register_infra_actions(registry: ActionRegistry) -> None:
    """Register infrastructure/observability actions."""

    registry.register(ActionDefinition(
        name="platform_get_logs",
        description=(
            "Fetch deployment logs from a Railway service. Returns recent log lines "
            "with timestamps and severity levels. Use to investigate errors, capture "
            "server-side context for bug reports, or monitor service health. "
            "Supports filtering by keyword (e.g. 'error', 'timeout', 'Exception')."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Railway service name to fetch logs from "
                        "(e.g. 'automatos-api', 'workspace-worker'). "
                        "Use 'list' to see all available services."
                    ),
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of log lines to retrieve (default 200, max 1000).",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Filter logs by keyword or severity. "
                        "Examples: 'error', 'Exception', 'timeout', 'WARNING'."
                    ),
                },
            },
            "required": ["service"],
        },
        permission_level="read",
        tags=["logs", "infrastructure", "railway", "observability", "debugging"],
        examples=[
            "get error logs from the API",
            "fetch recent logs from automatos-api",
            "show me the last 100 warning logs from workspace-worker",
            "list available services",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_services",
        description=(
            "List all Railway services in the project. Returns service names and IDs. "
            "Use to discover available services before fetching logs."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["services", "infrastructure", "railway"],
        examples=[
            "what services are running?",
            "list railway services",
        ],
    ))
