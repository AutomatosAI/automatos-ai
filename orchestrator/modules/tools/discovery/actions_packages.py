"""
PRD-230 US-006 — Package tool schemas (the LLM-facing definitions).
==================================================================

Three tools let Auto staff a workspace from curated marketplace packages. Every
install rides the US-005 installer (full closure, workspace-owned, idempotent).
Schemas match the handlers in ``handlers_packages.py`` (tool-schema walker green).
"""
from .action_registry import ActionDefinition, ActionRegistry


def register_package_actions(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_search_packages",
        description=(
            "Search curated marketplace PACKAGES (starter teams) that match a "
            "business. A package bundles existing agents, tools, skills, playbooks "
            "and an LLM with a setup guide. Pass any business signals you have: "
            "platforms (e.g. 'shopify'), store/site urls, and free text; results "
            "are ranked with a contents summary and the apps each needs connected. "
            "Use this to propose ONE package during onboarding."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Free-text business description / the conversation so far.",
                },
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Platform names mentioned, e.g. ['shopify'].",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Store or business URLs mentioned.",
                },
                "vertical_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit vertical hints, e.g. ['ecommerce'].",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["marketplace", "packages", "search", "onboarding"],
        examples=[
            "find a package for a shopify store",
            "what starter team fits an online jewellery shop",
            "search packages for ecommerce",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_install_package",
        description=(
            "Install a marketplace package into this workspace with its FULL "
            "dependency closure — every agent, tool, skill, playbook and LLM, "
            "workspace-owned and editable. Idempotent. Returns a manifest of what "
            "was registered and which apps still need connecting (guided, never "
            "auto-connected). During onboarding exactly ONE package installs; more "
            "any time afterwards from the marketplace. If it would exceed your "
            "plan's agent limit, you'll get a plan recommendation instead of a "
            "partial install. Provide slug (from platform_search_packages)."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Slug of the package to install (from platform_search_packages).",
                },
            },
            "required": ["slug"],
        },
        permission_level="write",
        promoted=True,
        tags=["marketplace", "packages", "install", "onboarding"],
        examples=[
            "install the shopify management package",
            "set up the shopify-development team",
            "add the store package to my workspace",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_install_marketplace_agent",
        description=(
            "Install a single marketplace AGENT into this workspace with its full "
            "dependency closure (its LLM, skills, plugins) — workspace-owned and "
            "editable, idempotent. Its connected apps come back as guided connect "
            "steps, never auto-connected. Browse first with "
            "platform_browse_marketplace_agents. Provide agent_id or agent_name."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the marketplace agent to install.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the marketplace agent (alternative to agent_id).",
                },
            },
            "required": [],
        },
        permission_level="write",
        promoted=True,
        tags=["marketplace", "agents", "install"],
        examples=[
            "install the inventory manager agent",
            "add marketplace agent Store Analyst",
            "install agent 42",
        ],
    ))
