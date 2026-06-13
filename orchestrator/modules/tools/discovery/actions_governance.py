"""Governance ActionDefinitions — blueprints, agent validation, budget checks."""

from .action_registry import ActionDefinition, ActionRegistry


def register_governance_actions(registry: ActionRegistry) -> None:
    """Register governance platform actions."""

    registry.register(ActionDefinition(
        name="platform_list_blueprints",
        description=(
            "List all governance blueprints for the workspace. Blueprints define "
            "agent readiness rules: minimum tools, required system prompt, allowed "
            "models, budget limits. Use to see what standards are configured."
        ),
        category="governance",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["governance", "blueprints", "standards", "rules"],
        examples=[
            "list blueprints",
            "what governance rules exist?",
            "show agent standards",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_blueprint",
        description=(
            "Get a specific blueprint by ID with its full rule set. "
            "Use when you need details about a particular blueprint's requirements."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "blueprint_id": {
                    "type": "string",
                    "description": "UUID of the blueprint to retrieve.",
                },
            },
            "required": ["blueprint_id"],
        },
        permission_level="read",
        tags=["governance", "blueprints", "rules"],
        examples=[
            "get blueprint details",
            "show blueprint rules",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_create_blueprint",
        description=(
            "Create a governance blueprint of OPTIONAL quality rules for agent readiness "
            "(min_tools, require_system_prompt, max_budget_per_run, required_tags, allowed_models). "
            "Only use this when the user explicitly asks for governance / quality standards. "
            "Creating an agent does NOT require a blueprint — use platform_create_agent directly "
            "for that. Rules are optional and default to no constraints."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Blueprint name.",
                },
                "description": {
                    "type": "string",
                    "description": "What this blueprint enforces.",
                },
                "rules": {
                    "type": "object",
                    "description": (
                        "Optional rule set: {min_tools, require_system_prompt, max_budget_per_run, "
                        "required_tags, allowed_models}. Omit it for a permissive blueprint with no "
                        "constraints (the default) — never ask the user to supply rules just to proceed."
                    ),
                },
                "is_default": {
                    "type": "boolean",
                    "description": "Set as workspace default blueprint. Defaults to false.",
                },
            },
            "required": ["name"],
        },
        permission_level="write",
        tags=["governance", "blueprints", "create"],
        examples=[
            "create a blueprint requiring system prompts",
            "set up agent readiness rules",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_blueprint",
        description=(
            "Update an existing blueprint's rules, name, or description. "
            "Use when adjusting governance standards."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "blueprint_id": {
                    "type": "string",
                    "description": "UUID of the blueprint to update.",
                },
                "name": {"type": "string", "description": "New name."},
                "description": {"type": "string", "description": "New description."},
                "rules": {"type": "object", "description": "Updated rule set."},
                "is_default": {"type": "boolean", "description": "Set as default."},
            },
            "required": ["blueprint_id"],
        },
        permission_level="write",
        tags=["governance", "blueprints", "update"],
        examples=[
            "update blueprint rules",
            "change minimum tools requirement",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_validate_agent",
        description=(
            "Validate an agent against a governance blueprint. Returns pass/fail "
            "with specific failures and warnings. Uses the default blueprint if "
            "none specified. Use to check if an agent meets quality standards."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to validate.",
                },
                "blueprint_id": {
                    "type": "string",
                    "description": "Optional blueprint UUID. Uses workspace default if omitted.",
                },
            },
            "required": ["agent_id"],
        },
        permission_level="read",
        tags=["governance", "validation", "agents", "readiness"],
        examples=[
            "is this agent ready?",
            "validate agent against blueprint",
            "check agent readiness",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_check_budget",
        description=(
            "Check a mission's budget status — remaining cost, remaining tokens, "
            "and whether the budget is ok, warning, or exceeded. Use when monitoring "
            "mission spend or deciding whether to continue execution."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "UUID of the orchestration run (mission) to check.",
                },
            },
            "required": ["run_id"],
        },
        permission_level="read",
        tags=["governance", "budget", "cost", "missions"],
        examples=[
            "check mission budget",
            "how much budget is left?",
            "is the mission over budget?",
        ],
    ))
