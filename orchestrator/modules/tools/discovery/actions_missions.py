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
                **_MISSION_ID_PARAM,
                "modifications": {
                    "type": "object",
                    "description": "Optional approval-time plan edits (task_overrides, agent_overrides, notes).",
                },
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
        description="Reject an awaiting-approval mission plan (it transitions to failed). Use when the user declines the proposed plan.",
        category="missions",
        parameters={
            "type": "object",
            "properties": {
                **_MISSION_ID_PARAM,
                "reason": {"type": "string", "description": "Why the plan was rejected (returned to Auto's context)."},
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
            },
            "required": ["mission_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "replan"],
        examples=["replan that failed mission", "try the mission again with a different approach"],
    ))
