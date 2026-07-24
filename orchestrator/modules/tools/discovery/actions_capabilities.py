"""Capability-discovery ActionDefinitions (tool-surface deep review, PR-B).

``platform_find_tools`` is the WHEN-REQUIRED seam: instead of shipping the
whole action catalog into every turn, the agent searches the registry on
demand and gets back names, descriptions and parameter schemas it can then
call via ``platform_execute``. Promoted so its schema is always visible —
it IS the discovery surface and must never itself need discovering.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_capabilities_actions(registry: ActionRegistry) -> None:
    """Register capability-discovery platform actions."""

    registry.register(ActionDefinition(
        name="platform_find_tools",
        description=(
            "Search the platform's full action catalog by describing what you "
            "want to do. Returns matching actions with their description, "
            "required/optional parameters and how to call them via "
            "platform_execute. Use this whenever the task needs a capability "
            "you don't currently see a tool for — every platform action is "
            "reachable this way, so never assume something is impossible "
            "without checking here first."
        ),
        category="capabilities",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What you want to do, in plain language "
                        "(e.g. 'publish a blog post', 'index a github repo')."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return (default 8, max 25).",
                },
                "include_params": {
                    "type": "boolean",
                    "description": (
                        "Include each match's full parameter schema "
                        "(default true; set false for a compact name+description list)."
                    ),
                },
            },
            "required": ["query"],
        },
        permission_level="read",
        promoted=True,
        tags=["tools", "capabilities", "discover", "search", "actions", "help", "can you"],
        examples=[
            "what tools do you have for blogs?",
            "can you work with our codebase?",
            "find a tool to sync the shopify catalog",
            "what can you actually do on this platform?",
        ],
    ))
