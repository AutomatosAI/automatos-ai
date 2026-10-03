"""The tool that puts a playbook on a timer (F266: it keeps the owner's zone and wait-for-me).

Night 7b: Auto set the owner's timer in UTC (the owner is in Bristol), then sent
frequency/time, cron_schedule and wait_for_approval, which the tool didn't take, so the
timer couldn't be changed. The zone is kept by modules/tools/discovery/timer_zones.py;
this tool left actions_playbooks.py, whose one register function is past the length rule.
"""

from .action_registry import ActionDefinition, ActionRegistry

# F266 (night 7b): what Auto sent for the time, and where it goes.
_ONE_CRON = ("a timer's time is one cron_expression, minute hour day-of-month month day-of-week: every day at "
             "20:25 is '25 20 * * *', every Monday at 09:00 is '0 9 * * 1'.")
_SCHEDULE_MISPLACED = {"frequency": _ONE_CRON, "time": _ONE_CRON, "at": _ONE_CRON, "days": _ONE_CRON,
                       "schedule": _ONE_CRON, "interval": _ONE_CRON}


def register_playbook_schedule_action(registry: ActionRegistry) -> None:
    """Register platform_schedule_playbook."""
    registry.register(ActionDefinition(
        name="platform_schedule_playbook",
        description=(
            "Schedule a playbook to run automatically on a cron schedule. "
            "Sets the playbook's schedule_config so it fires at the specified "
            "times. Use platform_execute_playbook for immediate one-off runs. "
            "For one run at a set time, give a dated cron: '20 18 23 9 *' is "
            "18:20 on 23 September, once. The cron is read in the schedule's "
            "timezone: pass the owner's own (UK time is 'Europe/London'), and ask "
            "them first when you do not know where they are — never assume UTC. "
            "The reply names the zone used. Provide playbook_id or playbook_name. "
            "Calling it again on a playbook with a timer changes that timer."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "playbook_id": {
                    "type": "integer",
                    "description": "ID of the playbook to schedule.",
                },
                "playbook_name": {
                    "type": "string",
                    "description": "Name of the playbook to schedule (alternative to ID).",
                },
                "cron_expression": {
                    "type": "string",
                    "description": "5-field cron expression, read in the schedule's timezone (e.g. '0 9 * * 1' = every Monday at 09:00).",
                },
                "timezone": {
                    "type": "string",
                    "description": (
                        "The owner's IANA timezone, which the cron is read in (UK time is "
                        "'Europe/London'). Ask the owner when you do not know it; never assume "
                        "UTC. Left out, the workspace's zone (its heartbeat setting, else the zone "
                        "its other timers use) is used, else UTC."
                    ),
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Whether to activate the schedule immediately. Defaults to true.",
                },
                "wait_for_me": {
                    "type": "boolean",
                    "description": ("True when the owner wants every run's card to wait for their check in "
                                    "Review: the playbook's own wait-for-me, set in the same call."),
                },
            },
            "required": ["cron_expression"],
        },
        misplaced=_SCHEDULE_MISPLACED,
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "schedule", "cron", "automate", "recurring"],
        examples=[
            "schedule the daily briefing playbook to run at 9am",
            "set playbook 5 to run every Monday",
            "schedule this playbook on a cron",
            "automate the weekly review playbook",
        ],
    ))
