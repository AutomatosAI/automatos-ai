"""Auto reporting platform tools — Wave 2.

Three tools that let Auto (or any agent) read/write the workspace's
auto_reporting preferences and emit notifications without going round
the back of the platform.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_auto_reporting_actions(registry: ActionRegistry) -> None:
    """Register Auto-reporting + send-notification tools."""

    registry.register(ActionDefinition(
        name="platform_get_auto_reporting_prefs",
        description=(
            "Read this workspace's auto_reporting preferences — primary/fallback "
            "channels, quiet hours, digest frequency, and per-event routing rules. "
            "Use before deciding where to send a notification or before proposing "
            "preference changes to the user."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["settings", "auto-reporting", "notifications", "preferences"],
        examples=[
            "what are my notification preferences?",
            "where do I send reports?",
            "show auto reporting config",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_auto_reporting_prefs",
        description=(
            "Update the workspace's auto_reporting preferences. Accepts a partial "
            "object — only the supplied keys are merged into the existing config. "
            "Use this to set the primary channel (telegram/slack/in_app/webhook), "
            "configure quiet hours, change digest cadence, or set per-event routes. "
            "Always confirm with the user before changing the primary channel."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "primary_channel": {
                    "type": "string",
                    "enum": ["telegram", "slack", "in_app", "webhook"],
                },
                "fallback_channel": {
                    "type": "string",
                    "enum": ["in_app", "webhook"],
                },
                "quiet_hours": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "start": {"type": "string", "description": "HH:MM 24-hour."},
                        "end": {"type": "string", "description": "HH:MM 24-hour."},
                        "timezone": {"type": "string", "description": "IANA tz name, e.g. Europe/Dublin."},
                    },
                },
                "digest_frequency": {
                    "type": "string",
                    "enum": ["immediate", "daily", "weekly"],
                },
                "digest_time": {"type": "string", "description": "HH:MM the digest should fire."},
                "routes": {
                    "type": "object",
                    "description": (
                        "Map of event_type (or event_type:severity) to a destination. "
                        "Destination may be a literal channel (telegram/slack/in_app/webhook/silent) "
                        "or an alias ('primary'/'fallback'). Example: "
                        "{\"agent_error\": \"primary\", \"task_complete:info\": \"silent\"}."
                    ),
                },
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=True,
        tags=["settings", "auto-reporting", "notifications", "preferences"],
        examples=[
            "set telegram as the primary notification channel",
            "enable quiet hours from 10pm to 8am",
            "route urgent errors to telegram, info to in_app",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_send_notification",
        description=(
            "Send a notification through the workspace's configured channels. "
            "Honours auto_reporting routes, quiet hours, and the existing "
            "notification_preferences fan-out. Use for proactive Auto pings — "
            "approval requests, urgent alerts, decision summaries — not for "
            "every routine event (those auto-fire through the dispatcher)."
        ),
        category="notifications",
        parameters={
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "enum": [
                        "heartbeat_complete",
                        "task_complete",
                        "mission_step_complete",
                        "mission_complete",
                        "playbook_step_complete",
                        "playbook_complete",
                        "trigger_fired",
                        "report_submitted",
                        "agent_error",
                    ],
                    "description": "Platform event_type. Drives prefs lookup and routing.",
                },
                "title": {"type": "string"},
                "message": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "urgent", "security"],
                    "description": "Routing hint. Maps onto auto_reporting.routes for severity-based delivery.",
                },
                "status": {
                    "type": "string",
                    "enum": ["ok", "warning", "error", "info"],
                    "description": "Status icon used in formatted external messages.",
                },
                "link_type": {
                    "type": "string",
                    "description": "Optional link type, e.g. 'report' / 'task' / 'mission'.",
                },
                "link_id": {
                    "type": "string",
                    "description": "Optional link target id for in-app deep-link.",
                },
            },
            "required": ["event_type", "title"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["notifications", "send", "auto-reporting"],
        examples=[
            "ping me on telegram about the failing harness run",
            "send an approval request notification",
            "notify Gerard that the daily brief is ready",
        ],
    ))
