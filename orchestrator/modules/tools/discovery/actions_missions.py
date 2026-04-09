"""Mission ActionDefinitions (create, list, get)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_mission_actions(registry: ActionRegistry) -> None:
    """Register mission actions (PRD-82A)."""

    registry.register(ActionDefinition(
        name="platform_create_mission",
        description=(
            "Launch an autonomous multi-agent mission. The coordinator decomposes the "
            "goal into tasks, assigns agents, and orchestrates execution. "
            "Use for complex work requiring multiple agents: research, content creation, "
            "code generation, audits. For single-agent tasks, use platform_create_task instead."
        ),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "Natural-language goal for the mission. Be specific about "
                        "the desired outcome, quality bar, and any constraints. "
                        "The coordinator will decompose this into agent tasks."
                    ),
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Optional mission config overrides. Keys: "
                        "max_retries (int), category (str), "
                        "output_format (str: 'markdown'|'json'|'code'), "
                        "publish (bool: auto-publish result if applicable)."
                    ),
                },
            },
            "required": ["goal"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "orchestration", "multi-agent", "research", "content"],
        examples=[
            "launch a mission to research and write a blog post about AI agents",
            "start a mission to audit our API security",
            "create a mission to build a landing page",
            "run a deep research mission on competitor pricing",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_missions",
        description=(
            "List recent missions with ID, goal, state (pending/planning/running/"
            "completed/failed), and task count. Use to check status or find past missions. "
            "For full details of one mission, use platform_get_mission instead."
        ),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["pending", "planning", "running", "paused", "completed", "failed"],
                    "description": "Filter by mission state (omit for all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["missions", "read", "list", "status"],
        examples=[
            "what missions are running?",
            "list my missions",
            "show completed missions",
            "any failed missions?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_mission",
        description=(
            "Get full details of ONE mission — goal, state, task DAG, step results, "
            "and timing. For listing all missions, use platform_list_missions instead."
        ),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "integer",
                    "description": "The mission/run ID to look up",
                },
            },
            "required": ["mission_id"],
        },
        permission_level="read",
        tags=["missions", "read", "details", "status"],
        examples=[
            "show me mission 5",
            "what's the status of mission 12?",
            "get mission details",
        ],
    ))
