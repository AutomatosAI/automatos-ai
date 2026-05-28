"""Notification ActionDefinitions — Auto's escalation channel to the owner."""

from .action_registry import ActionDefinition, ActionRegistry


def register_notifications_actions(registry: ActionRegistry) -> None:
    """Register owner-notification actions."""

    registry.register(ActionDefinition(
        name="platform_notify_owner",
        description=(
            "Send the workspace owner a direct message via their preferred channel "
            "(Telegram, Slack, webhook, or in-app) when you need a decision, approval, "
            "or want to surface something time-sensitive that shouldn't wait for them "
            "to check the board. Also creates a BoardTask so the request has a "
            "persistent record. Use this for: approval requests, urgent risks, "
            "blocking decisions, status that needs eyes within the hour. Do NOT use "
            "this for routine completion reports — use platform_submit_report."
        ),
        category="notifications",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Body of the message. Be specific: state the decision needed, "
                        "the options, your recommendation, and what you'll do if no "
                        "reply arrives. Markdown is supported on Telegram/Slack."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": "Short headline (e.g. 'Approve $500 LinkedIn ad?'). Defaults to 'Auto needs your input'.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": (
                        "low/normal = informational. high = decision needed today. "
                        "urgent = blocking right now (rare — only use when work stops "
                        "without a reply). Affects message prefix and BoardTask priority."
                    ),
                },
                "channel": {
                    "type": "string",
                    "description": (
                        "Optional explicit override (telegram | slack | webhook | in_app). "
                        "Default: workspace.settings.orchestrator.preferred_channel."
                    ),
                },
                "create_task": {
                    "type": "boolean",
                    "description": "Also create a BoardTask as a persistent backup record. Default: true.",
                },
            },
            "required": ["message"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["notifications", "escalation", "approval", "owner"],
        examples=[
            "ask Gerard whether to approve the $500 LinkedIn ad spend",
            "tell the owner the daily-social-post agent has been failing for 3 days",
            "escalate to Gerard — mission 22c489bb is blocked on a config decision",
            "notify the owner that the cost spike on agent 337 needs review",
        ],
    ))
