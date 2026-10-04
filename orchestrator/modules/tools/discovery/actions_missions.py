"""Mission ActionDefinitions (list, get, and the rest). Create is in actions_mission_create.py (F242)."""

from .action_registry import ActionDefinition, ActionRegistry

# F241 (night 7b): a mission tool takes a card's number too (mission_refs.takes_card_numbers).
MISSION_REF_TEXT = ("The mission: its id, or its card's number as the board shows it (#0188). "
                    "A step's number (#0188.3) names its mission.")
# PRD-163 S1: lifecycle control tools. These are how Auto drives a mission
# through its states from chat (approve/reject the plan, pause/resume/cancel
# a run, replan a failure). Each maps to an existing CoordinatorService method.
_MISSION_ID_PARAM = {
    "mission_id": {"type": "string", "description": MISSION_REF_TEXT},
}
# F241 (night 7b): "Approve #0177" reached platform_approve_mission for a task card.
NOT_FOR_A_CARD = (" A card in Review is approved or sent back on the board with "
                  "platform_update_task_status, never here.")


def register_mission_actions(registry: ActionRegistry) -> None:
    """Register mission actions (PRD-82A), in this order."""
    _register_list_missions(registry)
    _register_get_mission(registry)
    _register_approve_and_reject(registry)
    _register_pause_and_cancel(registry)
    _register_replan_mission(registry)
    _register_update_mission_plan(registry)


def _register_list_missions(registry: ActionRegistry) -> None:
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


def _register_get_mission(registry: ActionRegistry) -> None:
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
                "mission_id": {"type": "string", "description": MISSION_REF_TEXT},
            },
            "required": ["mission_id"],
        },
        permission_level="read",
        tags=["missions", "read", "details", "status"],
        examples=[
            "show me that mission",
            "what's the status of the pricing-research mission?",
            "get mission details",
        ],
    ))


def _register_approve_and_reject(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_approve_mission",
        description=(
            "Approve an awaiting-approval mission plan and start execution. Use when "
            "the user approves the plan you proposed (or says 'go ahead', 'run it')." + NOT_FOR_A_CARD
        ),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                # F142 (e): no `modifications` here. The approval never applied
                # them (api/missions.py PRD-163 S4 note), so an agent that sent
                # agent_overrides believed it had pinned staff when it had not.
                # Plan edits go through platform_update_mission_plan.
                **_MISSION_ID_PARAM,
            },
            "required": ["mission_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "approve"],
        examples=["approve that mission", "go ahead and run the plan", "yes, start the mission"],
    ))

    registry.register(ActionDefinition(
        name="platform_reject_mission",
        description=("Reject an awaiting-approval mission plan: it never runs and is closed as cancelled, "
                     "with the reason. Use when the user declines the proposed plan." + NOT_FOR_A_CARD),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                **_MISSION_ID_PARAM,
                "reason": {"type": "string", "description": (
                    "Why the owner turned the plan down, in the owner's own words: quote what they "
                    "asked to change, do not summarise it. The next plan for this conversation reads it.")},
            },
            "required": ["mission_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "reject"],
        examples=["reject that plan", "no, don't run that mission", "cancel the proposed plan"],
    ))


def _register_pause_and_cancel(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_pause_mission",
        description="Pause a running mission. In-flight tasks finish; no new tasks dispatch until resumed.",
        category="missions",
        parameters={"type": "object", "properties": dict(_MISSION_ID_PARAM), "required": ["mission_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "pause"],
        examples=["pause that mission", "hold the running mission"],
    ))

    registry.register(ActionDefinition(
        name="platform_cancel_mission",
        description=("Cancel a mission. Pending/queued tasks are skipped; in-flight tasks finish. Terminal. "
                     "Any other card is cancelled with platform_update_task_status."),
        category="missions",
        parameters={"type": "object", "properties": dict(_MISSION_ID_PARAM), "required": ["mission_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "cancel"],
        examples=["cancel that mission", "stop the mission"],
    ))


def _register_replan_mission(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_replan_mission",
        description="Replan a failed mission — regenerate replacement tasks for the failed subtree while keeping completed work.",
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                **_MISSION_ID_PARAM,
                "notes": {"type": "string", "description": "Optional guidance for the replanner."},
                "staffing": {
                    "type": "array",
                    "description": (
                        "To re-staff the mission (omit to keep its staffing; [] clears it): one entry per named "
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
            "required": ["mission_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "replan"],
        examples=["replan that failed mission", "try the mission again with a different approach"],
    ))


# F282 (night 8): Auto said "switch on the check for each step" in keys the plan never
# read (plan_updates, approval gates on each step); check_each_step is the setting.
_PLAN_MISPLACED = {
    key: ("the plan takes task_edits; to make every step wait in Review for the owner's check, send "
          "check_each_step: true")
    for key in ("plan_updates", "config", "settings", "approval_mode", "wait_for_me")
}


def _update_mission_plan_parameters() -> dict:
    """platform_update_mission_plan's parameters: the step edits, and the check of each step (F282)."""
    return {
        "type": "object",
        "properties": {
            **_MISSION_ID_PARAM,
            "task_edits": {
                "type": "array",
                "description": (
                    "Per-step edits. Name each step by its card's number as the board shows it "
                    "(#0352.2), by sequence_number, or by task_id; set any of agent_id, agent_role, "
                    "title, description. To have a specific agent run the step, give its agent_id (or "
                    "its name in agent_role when only one active agent has it)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "temp_id": {"type": "string"},
                        "sequence_number": {"type": "integer"},
                        "agent_id": {"type": "integer"},
                        "agent_role": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "check_each_step": {
                "type": "boolean",
                "description": ("True makes every step of the mission wait in Review for the owner's check "
                                "before the next one starts. Works until the mission finishes; task_edits "
                                "can be left out."),
            },
        },
        "required": ["mission_id"],
    }


def _register_update_mission_plan(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_update_mission_plan",
        description=(
            "Edit an awaiting-approval mission's plan before it runs — reassign a "
            "task's agent or revise a task title/description — or make every step "
            "wait for the owner's check (check_each_step). Use when the user tweaks "
            "the proposed plan ('have the researcher do step 2 instead')."
        ),
        category="missions",
        parameters=_update_mission_plan_parameters(),
        misplaced=_PLAN_MISPLACED,
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "plan", "edit"],
        examples=[
            "have the researcher handle step 2 instead",
            "reassign that first task to the writer agent",
            "rename task 3 to 'draft the summary'",
            "make every step of this mission wait for my check",
        ],
    ))
