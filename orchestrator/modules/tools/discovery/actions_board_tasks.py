"""Board task ActionDefinitions (create, list, get, assign, update status, summary)."""

from .action_registry import ActionDefinition, ActionRegistry


def _board_statuses() -> list:
    """The board's real status vocabulary, from the HTTP path's single source of
    truth (api.board_tasks.VALID_STATUSES). PRD-227 gave the agent HANDLER parity
    with it, but these SCHEMAS still listed five of seven: 'blocked' and 'failed'
    were invisible to the model, so "close all the blocked tickets" could not even
    list them (2026-09-02). Sorted for a stable schema; the fallback is the same
    set spelled out, so a registry build never depends on the API module importing."""
    try:
        from api.board_tasks import VALID_STATUSES
        return sorted(VALID_STATUSES)
    except Exception:  # pragma: no cover — import-order safety only
        return ["assigned", "blocked", "done", "failed", "in_progress", "inbox", "review"]


def register_board_task_actions(registry: ActionRegistry) -> None:
    """Register board task actions (PRD-72)."""

    registry.register(ActionDefinition(
        name="platform_create_task",
        description=(
            "Create a new task on the board. Use this to raise work items for yourself "
            "or another agent. Tasks appear in the Inbox and agents pick them up on their "
            "next heartbeat. Use for bug reports, follow-ups, sub-tasks from recipes."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short task title",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed task description / prompt for the agent",
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low"],
                    "description": "Task priority",
                },
                "assigned_agent_name": {
                    "type": "string",
                    "description": "Name of agent to assign (default: unassigned)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "parent_task_id": {
                    "type": "integer",
                    "description": "Parent task ID if this is a sub-task",
                },
                "review_mode": {
                    "type": "string",
                    "enum": ["auto", "manual"],
                    "description": "PRD-234: 'manual' parks the finished ticket in Review for a human; 'auto' (default) closes it Done.",
                },
                "sla_deadline": {
                    "type": "string",
                    "description": "PRD-234: due date/time as ISO 8601 (e.g. 2026-09-05T17:00:00Z). Shows on the board and the calendar.",
                },
                "approval_action": {
                    "type": "object",
                    "description": "If set, task goes to Review status with an approval gate. On user approve, the action executes. Example: {\"type\": \"publish_blog\", \"post_id\": \"uuid\"}",
                },
                "status": {
                    "type": "string",
                    "enum": ["inbox", "assigned", "review"],
                    "description": "Initial task status. Auto-set to 'review' if approval_action is provided.",
                },
                "auto_approve": {
                    "type": "boolean",
                    "description": "If true AND approval_action is set, immediately execute the action (skip human review). Use for automated pipelines.",
                },
            },
            "required": ["title", "description"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["tasks", "write", "board", "bug", "follow-up"],
        examples=[
            "create a task to fix the login bug",
            "raise a task for the researcher to check competitor pricing",
            "create sub-tasks for each test failure",
        ],
    ))

    # ── Board read tools ────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_tasks",
        description=(
            "List tasks on the board with optional filters. Returns task titles, "
            "statuses, priorities, assigned agents, and dates. Use when the user asks "
            "about their tasks, board status, what's in progress, or what's in the queue."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": _board_statuses(),
                    "description": "Filter by status (omit for all)",
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low"],
                    "description": "Filter by priority",
                },
                "assigned_agent_name": {
                    "type": "string",
                    "description": "Filter by assigned agent name",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max results (default 20, max 200). The result's "
                        "'total_matching' says how many match in all — raise the "
                        "limit when you need every one (e.g. to close all blocked tasks)."
                    ),
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["tasks", "read", "board", "list"],
        examples=[
            "what tasks are on the board?",
            "show me in-progress tasks",
            "what's assigned to the researcher?",
            "list urgent tasks",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_board_summary",
        description=(
            "Get a summary of the task board: counts by status, by priority, "
            "busiest agents, recent completions, and any failed tasks. "
            "Use when the user asks for a board overview, daily standup, "
            "or how the team is doing."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["tasks", "read", "board", "summary", "analytics", "standup"],
        examples=[
            "how's the board looking?",
            "give me a board summary",
            "daily standup",
            "how many tasks are in progress?",
            "which agent is busiest?",
            "any failed tasks?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_task",
        description=(
            "Get full details of a specific task by ID. Returns title, description, "
            "status, assigned agent, result, error message, and timestamps."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task ID to look up",
                },
            },
            "required": ["task_id"],
        },
        permission_level="read",
        tags=["tasks", "read", "details"],
        examples=[
            "show me task 42",
            "what's the status of task 15?",
            "get details for task 7",
        ],
    ))

    # ── Board write tools ───────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_assign_task",
        description=(
            "Assign a board task to an agent by name. Moves the task to 'assigned' "
            "status so the agent picks it up on its next heartbeat."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task ID to assign",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to assign",
                },
            },
            "required": ["task_id", "agent_name"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["tasks", "write", "assign", "agent"],
        examples=[
            "assign task 12 to the researcher",
            "give task 5 to devops",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_task_status",
        description=(
            "Change a task's status — one task via task_id, or MANY tasks to the "
            "same status in ONE call via task_ids (use this for 'close all …'; "
            "never loop one call per task). Moving to 'in_progress' triggers "
            "immediate agent execution if an agent is assigned. Moving to 'done' "
            "completes it. 'blocked' requires blocked_reason."
        ),
        category="tasks",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task ID (single task)",
                },
                "task_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Several task IDs to move to the same status in one call "
                        "(max 100). The result lists 'updated' and 'failed' ids — "
                        "report both to the user."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": _board_statuses(),
                    "description": "New status",
                },
                "blocked_reason": {
                    "type": "string",
                    "description": "Why the task is blocked (required when status is 'blocked')",
                },
            },
            "required": ["status"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["tasks", "write", "status", "trigger", "run"],
        examples=[
            "move task 8 to in progress",
            "mark task 3 as done",
            "start task 12",
            "run task 5 now",
        ],
    ))
