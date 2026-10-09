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
        # PRD-256 US-006: the config's fields are named in the schema, not only in its description.
        "config": {
            "type": "object",
            "description": ("Optional mission config overrides: auto_approve, max_retries, category, "
                            "output_format, publish, check_each_step."),
            "properties": {
                "auto_approve": {
                    "type": "boolean",
                    "description": ("Skip the awaiting_approval gate and start executing immediately. Default "
                                    "false: the mission waits for human approval."),
                },
                "max_retries": {"type": "integer", "description": "How many times a failed step is retried."},
                "category": {"type": "string", "description": "The mission's category."},
                "output_format": {"type": "string", "enum": ["markdown", "json", "code"],
                                  "description": "The form of the mission's result."},
                "publish": {"type": "boolean", "description": "Publish the result when it applies."},
                "check_each_step": {
                    "type": "boolean",
                    "description": ("Every step waits for the owner's check before the mission goes on; set it "
                                    "whenever the owner says to wait for them."),
                },
            },
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
        # F262 (night 7b): the owner's words, as Auto sent them (mission_asks.py reads them).
        "wait_for_me": {
            "type": "boolean",
            "description": ("True when the owner wants to check each step before the mission goes on "
                            "('stop after every step for my approval'). The same as config.check_each_step."),
        },
        "steps": {
            "type": "array",
            "description": ("The owner's own steps, in order, when they listed them, each a line of text. "
                            "They join the goal, and the plan follows them."),
            "items": {"type": "string"},
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags for the mission's card, so the owner finds it by tag.",
        },
    },
    "required": ["goal"],
}

# F262 (night 7b): Auto sent a name ("sim-night-2026-10-03", the owner's tag) and a label.
_NAMED_BY_ITS_GOAL = ("a mission has no name of its own: its card is titled from its goal. Say what it is for in "
                      "goal, and put a tag in tags.")
_CREATE_MISSION_MISPLACED = {"name": _NAMED_BY_ITS_GOAL, "title": _NAMED_BY_ITS_GOAL, "label": _NAMED_BY_ITS_GOAL}


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
        misplaced=_CREATE_MISSION_MISPLACED,
        permission_level="write",
        promoted=True,  # PRD-256 US-006: a first-class tool, pinned
        requires_confirmation=False,
        tags=["missions", "write", "orchestration", "multi-agent", "research", "content"],
        examples=[
            "launch a mission to research and write a blog post about AI agents",
            "start a mission to audit our API security",
            "create a mission to build a landing page",
            # F288 (night 8): "run …" is a playbook's word; "Run my New Cafe Onboarding" became a mission 9 of 9.
            "begin a deep research mission on competitor pricing",
        ],
    ))
