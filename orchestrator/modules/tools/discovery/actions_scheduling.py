"""Scheduling ActionDefinitions (schedule task, list, cancel)."""

from .action_registry import ActionDefinition, ActionRegistry
from .agent_refs import agent_id_property


def register_scheduling_actions(registry: ActionRegistry) -> None:
    """Register agent self-scheduling actions (PRD-77), one tool each, in this
    order. Each ActionDefinition is built inside registry.register(...), where
    scripts/check_hierarchy_gate.py reads it."""
    _register_schedule_task(registry)
    _register_list_scheduled_tasks(registry)
    _register_cancel_scheduled_task(registry)
    _register_get_schedule(registry)


def _register_schedule_task(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_schedule_task",
        description=(
            "Schedule a follow-up task for yourself or another agent. "
            "One-shot tasks run once at a specific time. Recurring tasks use "
            "cron expressions (e.g. '0 9 * * 1' = every Monday at 9am). "
            "Use this when you discover something that needs revisiting later. "
            "deliver_as='chat' (default) opens a chat with the target agent when it "
            "fires; deliver_as='board_task' files a ticket on the board instead — "
            "assigned to target_agent_name if given, otherwise into the Inbox — so "
            "use it when the user wants a task on the board on a date ('put a "
            "ticket on the board for Thursday'). Both show on the calendar."
        ),
        category="scheduling",
        parameters={
            "type": "object",
            "properties": _schedule_task_properties(),
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


def _schedule_task_properties() -> dict:
    return {
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
        # P256-FIX-RVW-23: several agents share a name in a workspace; an id names one.
        "agent_id": agent_id_property("The agent to run the task, by id instead of target_agent_name"),
        "max_runs": {
            "type": "integer",
            "description": "For recurring: max number of executions before auto-cancel. Omit for unlimited.",
        },
        "deliver_as": {
            "type": "string",
            "enum": ["chat", "board_task"],
            "description": "chat = open a chat with the target agent when it fires (default); board_task = file a board ticket when it fires.",
        },
        "title": {
            "type": "string",
            "description": "board_task only: the ticket title (defaults to the first line of the description).",
        },
        "priority": {
            "type": "string",
            "enum": ["urgent", "high", "medium", "low"],
            "description": "board_task only: the ticket priority (default medium).",
        },
        "review_mode": {
            "type": "string",
            # PRD-252 D7: no 'llm' until a model reviewer exists; it behaved as 'human'.
            "enum": ["auto", "human"],
            "description": ("board_task only: the ticket's review gate (default auto; human when a person's "
                            "chat schedules a brief that sends, orders or publishes)."),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "board_task only: tags for the ticket.",
        },
    }


def _register_list_scheduled_tasks(registry: ActionRegistry) -> None:
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


def _register_cancel_scheduled_task(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_cancel_scheduled_task",
        description=(
            "Cancel a scheduled (recurring or one-off) task by its ID so it stops "
            "running on its schedule. Use when the user wants to stop, disable, or "
            "remove an automation/cron they previously set up. List scheduled tasks "
            "first if you don't know the ID."
        ),
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


def _register_get_schedule(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_get_schedule",
        description=(
            "Show everything scheduled in the workspace right now — agent heartbeat "
            "routines, cron-scheduled playbooks, scheduled tasks, and the SLA "
            "deadlines on open missions and board tasks — each with its next run "
            "or due time (a past due time on an open item means it is overdue). "
            "This is the same source of truth the calendar shows. Use when asked "
            "'what's scheduled?', 'what runs next?', 'what is due?', or to review "
            "what automations are set up."
        ),
        category="scheduling",
        parameters={
            "type": "object",
            "properties": {
                "range_days": {
                    "type": "integer",
                    "description": "How many days ahead to include (default 30).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["scheduling", "read", "calendar"],
        examples=[
            "what's scheduled",
            "what runs next",
            "show the schedule for the next week",
            "what automations do I have set up",
        ],
    ))
