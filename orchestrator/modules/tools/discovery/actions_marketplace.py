"""Marketplace browse, workspace inventory, and install ActionDefinitions."""

from .action_registry import ActionDefinition, ActionRegistry


def register_marketplace_actions(registry: ActionRegistry) -> None:
    """Register marketplace discovery, workspace inventory, and install actions."""

    # ── Marketplace Discovery (read) ────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_browse_marketplace_plugins",
        description=(
            "Browse the marketplace for plugins to ADD to workspace. "
            "Use when discovering or searching for new plugins to install. "
            "For plugins already installed, use platform_list_workspace_plugins instead. "
            "Returns name, slug, description, category, skills count, and install status."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Search term to filter plugins by name or description.",
                },
                "category": {
                    "type": "string",
                    "description": "Category slug to filter by (e.g., 'development', 'devops').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Defaults to 20.",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["marketplace", "plugins", "browse", "search"],
        examples=[
            "browse marketplace plugins",
            "search plugins for code review",
            "what plugins are available?",
            "find a plugin for testing",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_browse_marketplace_agents",
        description=(
            "Browse marketplace for pre-built agent templates to hire/install. "
            "Use when designing teams or searching for new agents to add. "
            "For agents already in workspace, use platform_list_agents instead. "
            "Returns name, description, category, model, tools, skills, and install status."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Search term to filter agents by name or description.",
                },
                "category": {
                    "type": "string",
                    "description": "Category to filter by (e.g., 'sales', 'marketing', 'devops', 'finance').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Defaults to 20.",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["marketplace", "agents", "browse", "search", "hire"],
        examples=[
            "browse marketplace agents",
            "search agents for marketing",
            "what agents are available to hire?",
            "find an agent for sales",
            "show me available team members",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_browse_marketplace_skills",
        description=(
            "Browse global skills catalog for skills to ADD. "
            "Use when discovering or searching for new skills to install. "
            "For skills already enabled, use platform_list_workspace_skills instead. "
            "Returns name, description, category, token cost, and install status."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Search term to filter skills by name or description.",
                },
                "category": {
                    "type": "string",
                    "description": "Category to filter by (e.g., 'cognitive', 'technical').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Defaults to 20.",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["marketplace", "skills", "browse", "search"],
        examples=[
            "browse marketplace skills",
            "search skills for summarization",
            "what skills are available?",
            "find a skill for code review",
        ],
    ))

    # ── Workspace Inventory (read) ──────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_workspace_plugins",
        description=(
            "List plugins already ENABLED in this workspace. "
            "Use when the user asks what plugins they have or what's installed. "
            "For browsing new plugins to install, use platform_browse_marketplace_plugins instead. "
            "Returns name, slug, description, category, skills count, and enabled date."
        ),
        category="workspace",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["workspace", "plugins", "inventory", "list"],
        examples=[
            "what plugins do I have?",
            "list my plugins",
            "show enabled plugins",
            "what plugins are installed?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_workspace_skills",
        description=(
            "List skills already ENABLED in this workspace. "
            "Use when the user asks what skills they have or what's installed. "
            "For browsing new skills to install, use platform_browse_marketplace_skills instead. "
            "Returns name, description, category, token cost, and enabled date."
        ),
        category="workspace",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["workspace", "skills", "inventory", "list"],
        examples=[
            "what skills do I have?",
            "list my skills",
            "show enabled skills",
            "what skills are installed?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_workspace_models",
        description=(
            "List LLM models currently INSTALLED for this workspace with costs, capabilities, "
            "and context windows. Use when the user asks what models are available or wants to "
            "compare options. Includes both user-installed and default models."
        ),
        category="workspace",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["workspace", "models", "llm", "inventory", "list"],
        examples=[
            "what models do I have?",
            "list my models",
            "show installed models",
            "what LLMs are available in my workspace?",
        ],
    ))

    # ── Installation (write) ────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_install_plugin",
        description=(
            "Enable a marketplace plugin for this workspace. "
            "The plugin must be approved and active. Idempotent — re-enabling is safe. "
            "Browse available plugins with platform_browse_marketplace_plugins first. "
            "Accepts plugin_id (UUID) or plugin_slug. Provide plugin_id or plugin_slug."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "plugin_id": {
                    "type": "string",
                    "description": "UUID of the plugin to install.",
                },
                "plugin_slug": {
                    "type": "string",
                    "description": "Slug of the plugin to install (alternative to plugin_id).",
                },
            },
            "required": [],
        },
        permission_level="write",
        promoted=True,
        tags=["marketplace", "plugins", "install", "enable"],
        examples=[
            "install the code review plugin",
            "enable plugin code-review",
            "add the testing plugin",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_install_skill",
        description=(
            "Enable a marketplace skill for this workspace. "
            "Must be a global marketplace skill. Idempotent — re-enabling is safe. "
            "Browse available skills with platform_browse_marketplace_skills first. "
            "Accepts skill_id (int) or skill_name. Provide skill_id or skill_name."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "integer",
                    "description": "ID of the skill to install.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to install (alternative to skill_id).",
                },
            },
            "required": [],
        },
        permission_level="write",
        promoted=True,
        tags=["marketplace", "skills", "install", "enable"],
        examples=[
            "install the summarization skill",
            "enable skill code-review",
            "add the analysis skill",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_install_model",
        description=(
            "Install an LLM model from OpenRouter catalog. "
            "Use the model_id format like 'anthropic/claude-sonnet-4-20250514'. "
            "Idempotent — re-installing reactivates inactive installs. "
            "Auto-creates registry entry from OpenRouter cache if needed."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "OpenRouter model ID (e.g., 'anthropic/claude-sonnet-4-20250514').",
                },
            },
            "required": ["model_id"],
        },
        permission_level="write",
        tags=["marketplace", "models", "llm", "install", "enable"],
        examples=[
            "install anthropic/claude-sonnet-4-20250514",
            "add model google/gemini-2.5-pro",
            "enable openai/gpt-4o",
        ],
    ))

    # PRD-143 S11: the disable side of platform_install_plugin (administration
    # surface, operator tier). Destructive because it cascades: agent
    # assignments for this workspace are removed with the junction record.
    registry.register(ActionDefinition(
        name="platform_uninstall_plugin",
        description=(
            "Disable a marketplace plugin for this workspace. Removes the "
            "plugin and unassigns it from every agent in the workspace — the "
            "agents lose its skills/commands immediately. Use "
            "platform_list_workspace_plugins first to get the plugin_id. "
            "Re-enabling later is possible with platform_install_plugin, but "
            "agent assignments are not restored."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "plugin_id": {
                    "type": "string",
                    "description": "The plugin id (UUID) from platform_list_workspace_plugins.",
                },
            },
            "required": ["plugin_id"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["marketplace", "plugins", "disable", "uninstall"],
        examples=[
            "disable the shopify plugin",
            "remove that plugin from the workspace",
            "uninstall the SEO plugin",
        ],
    ))
