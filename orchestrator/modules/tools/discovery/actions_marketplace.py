"""Marketplace browse, workspace inventory, and install ActionDefinitions."""

from .action_registry import ActionDefinition, ActionRegistry


def register_marketplace_actions(registry: ActionRegistry) -> None:
    """Register marketplace discovery, workspace inventory, and install actions."""

    # ── Marketplace Discovery (read) ────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_browse_marketplace_plugins",
        description=(
            "Browse or search the marketplace for approved plugins. Returns plugin name, "
            "slug, description, category, skills count, and whether it's already enabled "
            "in this workspace. Use when the user wants to discover new plugins."
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
            "Browse or search the marketplace for pre-built agent templates. Returns agent name, "
            "description, category, model, tools, skills, install count, and whether it's already "
            "installed in this workspace. Use when designing teams or hiring new agents."
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
            "Browse or search the global skills catalog (marketplace skills). Returns "
            "skill name, description, category, estimated token cost, and whether it's "
            "already enabled in this workspace. Use when the user wants to discover skills."
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
            "List all plugins currently enabled for this workspace. Returns plugin name, "
            "slug, description, category, skills count, and when it was enabled. "
            "Use when the user asks what plugins they have installed."
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
            "List all skills currently enabled for this workspace. Returns skill name, "
            "description, category, estimated token cost, and when it was enabled. "
            "Use when the user asks what skills they have installed."
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
            "List all LLM models installed for this workspace, including default models. "
            "Returns model ID, display name, provider, costs, context length, capabilities, "
            "and source. Use when the user asks what models they have available."
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
            "Enable a marketplace plugin for this workspace. Accepts plugin_id (UUID) "
            "or plugin_slug. The plugin must be approved and active. Idempotent — "
            "re-enabling an already-enabled plugin is a no-op."
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
            "Enable a marketplace skill for this workspace. Accepts skill_id (int) "
            "or skill_name. The skill must be a global marketplace skill (not workspace-specific) "
            "and active. Idempotent — re-enabling an already-enabled skill is a no-op."
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
            "Install an LLM model for this workspace from the OpenRouter catalog. "
            "Accepts a model_id string (e.g., 'anthropic/claude-sonnet-4-20250514'). "
            "Auto-creates the LLM registry entry from the OpenRouter cache if needed. "
            "Idempotent — re-installing re-activates an inactive install."
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
