"""Scheduling ActionDefinitions (schedule task, list, cancel)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_scheduling_actions(registry: ActionRegistry) -> None:
    """Register agent self-scheduling actions (PRD-77)."""

    registry.register(ActionDefinition(
        name="platform_schedule_task",
        description=(
            "Schedule a follow-up task for yourself or another agent. "
            "One-shot tasks run once at a specific time. Recurring tasks use "
            "cron expressions (e.g. '0 9 * * 1' = every Monday at 9am). "
            "Use this when you discover something that needs revisiting later."
        ),
        category="scheduling",
        parameters={
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["one_shot", "recurring"],
                    "description": "one_shot runs once at schedule time; recurring uses cron.",
                },
                "description": {
                    "type": "string",
                    "description": "What the task should accomplish. Be specific — this becomes the opening message to the target agent.",
                },
                "schedule": {
                    "type": "string",
                    "description": "ISO datetime for one_shot (e.g. '2026-03-11T09:00:00Z'), cron for recurring (e.g. '0 9 * * 1').",
                },
                "target_agent_name": {
                    "type": "string",
                    "description": "Name of the agent to run the task (defaults to yourself).",
                },
                "max_runs": {
                    "type": "integer",
                    "description": "For recurring: max number of executions before auto-cancel. Omit for unlimited.",
                },
            },
            "required": ["task_type", "description", "schedule"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["scheduling", "write", "follow-up", "cron"],
        examples=[
            "schedule a follow-up check for tomorrow morning",
            "remind me to review this in 3 days",
            "set up a weekly check every Monday at 9am",
            "schedule the researcher to update competitor data weekly",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_scheduled_tasks",
        description=(
            "List all scheduled tasks for the workspace. Shows pending, active, "
            "and completed tasks with their schedules and run history."
        ),
        category="scheduling",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "cancelled", "failed"],
                    "description": "Filter by task status (optional).",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Filter by agent name (optional).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["scheduling", "read"],
        examples=[
            "what tasks are scheduled",
            "show my scheduled tasks",
            "list active scheduled tasks",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_cancel_scheduled_task",
        description="Cancel a scheduled task by ID.",
        category="scheduling",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to cancel.",
                },
            },
            "required": ["task_id"],
        },
        permission_level="write",
        requires_confirmation=True,
        tags=["scheduling", "write", "destructive"],
        examples=["cancel scheduled task 5", "stop that recurring task"],
    ))
