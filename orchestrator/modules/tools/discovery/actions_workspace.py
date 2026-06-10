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
            "Get workspace metadata — name, member count, creation date, config "
            "summary. Use when the user asks about their workspace or account. "
            "For agent details, use platform_list_agents instead."
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
            "Get memory system statistics — total memories, types, storage usage. "
            "Use when the user asks about memory stats or what the system remembers. "
            "For searching specific memories, use platform_search_memory instead."
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
            "Store a curated fact in workspace long-term memory for future conversations. "
            "Use for: user facts, confirmed decisions, workspace patterns, user corrections. "
            "Do NOT use for: task artifacts, raw tool outputs, volatile data. "
            "Keep under 200 chars. For searching stored memories, use platform_search_memory.\n\n"
            "Wave 3 — provenance: when you set ``source_type``, future readers can tell "
            "platform_verified facts from claude_reports / current_status / inference. "
            "Always set ``source_type`` honestly; default is 'inference' when unsure."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember.",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["platform_verified", "claude_reports", "current_status", "inference"],
                    "description": (
                        "Provenance: platform_verified (queried + confirmed via tools), "
                        "claude_reports (the assistant's claim, unverified), current_status "
                        "(transient state read from a live source), inference (pattern-based)."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0 confidence in the claim. 1.0 = verified.",
                },
                "evidence_uri": {
                    "type": "string",
                    "description": "Optional pointer to the source — workspace file path, report id, run id, etc.",
                },
            },
            "required": ["content"],
        },
        permission_level="write",
        promoted=True,
        requires_confirmation=False,
        tags=["memory", "store", "write", "provenance"],
        examples=[
            "remember that our deploy day is Thursday",
            "store this: API key rotates every 90 days",
            "store with source_type=platform_verified after running the check",
        ],
    ))

    # ── Memory Browse/Delete ─────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_browse_memories",
        description=(
            "Browse all stored memories with optional keyword filter. Returns "
            "paginated list with content, agent, date. For storing new memories, "
            "use platform_store_memory instead."
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
        description="Permanently delete a specific memory by ID. Cannot be undone.",
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
            "List external apps connected to the workspace via Composio (Slack, "
            "Gmail, GitHub, etc.) with connection status. Use to check what "
            "integrations are available before using Composio tools."
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
            "Check platform health — database, Redis, API, RAG pipeline, CPU, "
            "memory, disk. Use for quick system status checks or when something "
            "seems broken."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        super_admin_only=True,
        promoted=True,
        tags=["health", "status", "infrastructure", "monitoring"],
        examples=[
            "is the system healthy?",
            "check system health",
            "is everything working?",
            "platform health check",
        ],
    ))

    # ── PRD-143 S11: administration surface — workspace + system settings ──
    # Operator tier by design (Rev 2). The workspace-settings tool is
    # fail-closed on a key whitelist (handlers_workspace.
    # OPERATOR_WORKSPACE_SETTINGS_KEYS); system-setting updates are
    # platform-wide, hence requires_confirmation=True.

    registry.register(ActionDefinition(
        name="platform_update_workspace_settings",
        description=(
            "Update a workspace setting. Supported keys: 'byok_overrides' "
            "(per-provider bring-your-own-key toggles, e.g. {\"openai\": true}) "
            "and 'default_notification_channel' (in_app, webhook, telegram or "
            "slack). Other settings have dedicated tools (power mode, widget "
            "config, autonomy) — this tool refuses keys outside its whitelist."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": ["byok_overrides", "default_notification_channel"],
                    "description": "Which workspace setting to update.",
                },
                "value": {
                    "description": (
                        "For byok_overrides: an object of provider -> boolean. "
                        "For default_notification_channel: the channel name."
                    ),
                },
            },
            "required": ["key", "value"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["settings", "workspace", "byok", "notifications", "configuration"],
        examples=[
            "use my own OpenAI key for this workspace",
            "turn off BYOK for anthropic",
            "set the default notification channel to telegram",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_system_settings",
        description=(
            "List platform system settings (optionally filtered by category) — "
            "the database-backed configuration that replaces .env. Sensitive "
            "values are always masked. Use before updating a setting or when "
            "the user asks how the platform is configured."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g. 'llm', 'email').",
                },
            },
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["settings", "system", "configuration", "platform"],
        examples=[
            "list the system settings",
            "show the llm configuration settings",
            "what platform settings exist?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_system_setting",
        description=(
            "Update one platform system setting by category + key. This is "
            "PLATFORM-WIDE — it changes behaviour for every workspace, which "
            "is why it always asks for confirmation first. Use "
            "platform_list_system_settings to find the category and key."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The setting's category (e.g. 'llm').",
                },
                "key": {
                    "type": "string",
                    "description": "The setting's key within the category.",
                },
                "value": {
                    "type": "string",
                    "description": "The new value (stored as a string).",
                },
            },
            "required": ["category", "key", "value"],
        },
        permission_level="write",
        requires_confirmation=True,
        tags=["settings", "system", "configuration", "platform"],
        examples=[
            "change the default LLM model setting",
            "update the email sender system setting",
        ],
    ))
