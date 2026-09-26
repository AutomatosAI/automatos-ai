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
                        "auto_approve (bool: skip the awaiting_approval gate and "
                        "start executing immediately — default false, the mission "
                        "waits for human approval), "
                        "max_retries (int), category (str), "
                        "output_format (str: 'markdown'|'json'|'code'), "
                        "publish (bool: auto-publish result if applicable)."
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
                    "type": "string",
                    "description": "The mission/run UUID to look up",
                },
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

    # PRD-163 S1: lifecycle control tools. These are how Auto drives a mission
    # through its states from chat (approve/reject the plan, pause/resume/cancel
    # a run, replan a failure). Each maps to an existing CoordinatorService method.
    _MISSION_ID_PARAM = {
        "mission_id": {"type": "string", "description": "The mission/run UUID."},
    }

    registry.register(ActionDefinition(
        name="platform_approve_mission",
        description=(
            "Approve an awaiting-approval mission plan and start execution. Use when "
            "the user approves the plan you proposed (or says 'go ahead', 'run it')."
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
                     "with the reason. Use when the user declines the proposed plan."),
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
        name="platform_resume_mission",
        description="Resume a paused mission (it goes back to running).",
        category="missions",
        parameters={"type": "object", "properties": dict(_MISSION_ID_PARAM), "required": ["mission_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "resume"],
        examples=["resume that mission", "continue the paused mission"],
    ))

    registry.register(ActionDefinition(
        name="platform_cancel_mission",
        description="Cancel a mission. Pending/queued tasks are skipped; in-flight tasks finish. Terminal.",
        category="missions",
        parameters={"type": "object", "properties": dict(_MISSION_ID_PARAM), "required": ["mission_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "cancel"],
        examples=["cancel that mission", "stop the mission"],
    ))

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

    registry.register(ActionDefinition(
        name="platform_update_mission_plan",
        description=(
            "Edit an awaiting-approval mission's plan before it runs — reassign a "
            "task's agent or revise a task title/description. Use when the user "
            "tweaks the proposed plan ('have the researcher do step 2 instead')."
        ),
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                **_MISSION_ID_PARAM,
                "task_edits": {
                    "type": "array",
                    "description": (
                        "Per-task edits. Identify each task by task_id, temp_id, or "
                        "sequence_number; set any of agent_id, agent_role, title, description. "
                        "To have a specific agent run the task, give its agent_id (or its "
                        "name when only one active agent has it)."
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
            },
            "required": ["mission_id", "task_edits"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "plan", "edit"],
        examples=[
            "have the researcher handle step 2 instead",
            "reassign that first task to the writer agent",
            "rename task 3 to 'draft the summary'",
        ],
    ))
