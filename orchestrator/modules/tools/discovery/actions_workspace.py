"""Workspace info, memory, integrations, and system health ActionDefinitions."""

from .action_registry import ActionDefinition, ActionRegistry


def register_workspace_actions_defs(registry: ActionRegistry) -> None:
    """Register workspace info, memory, integrations, and system health actions.

    Note: Named register_workspace_actions_defs to avoid collision with
    workspace_actions.register_workspace_actions (file I/O tools).
    """

    # ── Workspace Info ───────────────────────────────────────────────

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

    # ── Memory ───────────────────────────────────────────────────────

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
        promoted=True,
        requires_confirmation=False,
        tags=["memory", "store", "write"],
        examples=[
            "remember that our deploy day is Thursday",
            "store this: API key rotates every 90 days",
        ],
    ))

    # ── Memory Browse/Delete ─────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_browse_memories",
        description=(
            "Browse all stored memories for the workspace. Returns a paginated "
            "list of memories with content, agent, creation date, and metadata."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for semantic search (optional).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["memory", "read", "browse"],
        examples=[
            "show all memories",
            "browse memories",
            "what has been remembered",
            "search memories for user preferences",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_memory",
        description="Delete a specific memory by ID.",
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to delete.",
                },
            },
            "required": ["memory_id"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["memory", "write", "destructive"],
        examples=["delete memory abc-123", "remove that incorrect memory"],
    ))

    # ── Integrations ─────────────────────────────────────────────────

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

    # ── System Health ────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_system_health",
        description=(
            "Check system health — database, Redis, API, RAG pipeline, and server "
            "metrics (CPU, memory, disk). Use when the user asks if the system is "
            "healthy, working, or wants a status check."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        admin_only=True,
        promoted=True,
        tags=["health", "status", "infrastructure", "monitoring"],
        examples=[
            "is the system healthy?",
            "check system health",
            "is everything working?",
            "platform health check",
        ],
    ))
