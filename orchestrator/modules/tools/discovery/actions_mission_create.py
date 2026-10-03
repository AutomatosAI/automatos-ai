"""The tool that launches a mission (F242: it says how a mission waits for the owner).

Night 7: Auto launched missions with ``approval_mode: step_by_step`` (#0139) and
``review_mode: each_task`` (#0176), keys nothing read, so every step closed
without the owner. ``check_each_step`` is now the key, and the coordinator
honours it (modules/coordination/owner_checks.py). This tool left
actions_missions.py, whose one register function is past the length rule.
"""

from .action_registry import ActionDefinition, ActionRegistry

_CREATE_MISSION_PARAMETERS = {
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
                "auto_approve (bool: skip the awaiting_approval gate and "
                "start executing immediately — default false, the mission "
                "waits for human approval), "
                "max_retries (int), category (str), "
                "output_format (str: 'markdown'|'json'|'code'), "
                "publish (bool: auto-publish result if applicable), "
                "check_each_step (bool: every step waits for the owner's check before the "
                "mission goes on; set it whenever the owner says to wait for them)."
            ),
        },
        "staffing": {
            "type": "array",
            "description": (
                "Only when the owner says which agent does what: one entry per named "
                "agent, with its work in the owner's words. Each named agent gets that "
                "work and is pinned to it; anything else is routed by capability. A name "
                "several agents share is refused with their ids: ask the owner which one."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "The agent's name, slug or id"},
                    "does": {"type": "string", "description": "Its work, in the owner's words"},
                },
                "required": ["agent", "does"],
            },
        },
    },
    "required": ["goal"],
}


def register_mission_create_action(registry: ActionRegistry) -> None:
    """Register platform_create_mission (PRD-82A)."""
    registry.register(ActionDefinition(
        name="platform_create_mission",
        description=(
            "Launch an autonomous multi-agent mission. The coordinator decomposes the "
            "goal into tasks, assigns agents, and orchestrates execution. "
            "Use for complex work requiring multiple agents: research, content creation, "
            "code generation, audits. For single-agent tasks, use platform_create_task instead."
        ),
        category="missions",
        parameters=_CREATE_MISSION_PARAMETERS,
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
