"""Tool and LLM listing/discovery ActionDefinitions."""

from .action_registry import ActionDefinition, ActionRegistry


def register_tools_llms_actions(registry: ActionRegistry) -> None:
    """Register tool listing, LLM listing, and datasource discovery actions."""

    registry.register(ActionDefinition(
        name="platform_list_tools",
        description=(
            "List all available tools — platform actions, Composio integrations, "
            "and internal tools. Grouped by category with descriptions and connection "
            "status. Use when the user asks what tools are available, what can be done, "
            "or when Auto needs to discover capabilities for workflow design."
        ),
        category="discovery",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["platform", "composio", "all"],
                    "description": "Filter by tool type. Defaults to 'all'.",
                },
                "search": {
                    "type": "string",
                    "description": "Fuzzy search across tool names and descriptions.",
                },
                "connected_only": {
                    "type": "boolean",
                    "description": "Only show tools with active connections. Defaults to false.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["tools", "discovery", "capabilities", "integrations"],
        examples=[
            "what tools can I use?",
            "list my tools",
            "what integrations are available?",
            "show connected tools",
            "search tools for github",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_llms",
        description=(
            "List available LLM models from the OpenRouter model cache with costs, "
            "capabilities, and context windows. Filterable by capability (tools, vision, "
            "reasoning) and tier (free, budget, mid, premium). Use when the user asks "
            "about available models, pricing, or model capabilities."
        ),
        category="discovery",
        parameters={
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "enum": ["tools", "vision", "reasoning", "json_mode"],
                    "description": "Filter by model capability.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["free", "budget", "mid", "premium"],
                    "description": "Filter by pricing tier.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["cost", "context_length", "name"],
                    "description": "Sort results. Defaults to 'cost'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Defaults to 20, max 50.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["models", "llm", "discovery", "pricing", "openrouter"],
        examples=[
            "what models are available?",
            "list LLMs",
            "cheapest model with tool support",
            "show models with vision capability",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_datasources",
        description=(
            "List all data sources — document collections (RAG knowledge base) and "
            "database connections (NL2SQL). Shows document counts, chunk totals, file "
            "types, and database connection details. Use when the user asks about their "
            "data, documents, databases, or what knowledge is available."
        ),
        category="discovery",
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["documents", "databases", "all"],
                    "description": "Filter by data source type. Defaults to 'all'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["data", "documents", "databases", "rag", "nl2sql", "discovery"],
        examples=[
            "what data sources do I have?",
            "show my databases",
            "what documents are indexed?",
            "list datasources",
        ],
    ))
