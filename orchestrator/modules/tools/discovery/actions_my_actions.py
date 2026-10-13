"""platform_my_actions (P256-FIX-N14-3, night 14, F417): Auto's record of what it did.

Night 14, after a night of writes, Auto said it had only read things: each turn's receipts were
saved, and nothing read them across chats. This read does (``handlers_my_actions``).
"""

from .action_registry import ActionDefinition, ActionRegistry
from .my_actions_window import DEFAULT_WINDOW, TICKET_LOOKBACK_DAYS, WINDOWS


def register_my_actions_action(registry: ActionRegistry) -> None:
    """Register platform_my_actions."""
    registry.register(ActionDefinition(
        name="platform_my_actions",
        description=(
            "What you (Auto) did across ALL of this person's chats, and what they decided: every write "
            "that went through (cards, agents, skills, memories, playbook runs, drafts, sends), every write "
            "still waiting for their click, and each approval card they approved or declined, with when, "
            "and each card by its board number (#1437). Read from the platform's own receipts, never your "
            "memory of the chat. Use it BEFORE answering 'what have you done / created / sent / changed "
            "tonight?' or 'what did I decide about #1437?' (give ticket). Report what it lists as earlier "
            "work, each with its time."
        ),
        category="memory",
        parameters={
            "type": "object",
            "properties": {
                "window": {
                    "type": "string",
                    "enum": list(WINDOWS),
                    "description": (
                        "Which stretch of time: tonight (since 18:00 UTC the evening before), today, "
                        f"yesterday, last_24_hours or this_week. Defaults to {DEFAULT_WINDOW}."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": "Instead of window: from this time on, ISO 8601 (e.g. 2026-10-12T18:00:00Z).",
                },
                "ticket": {
                    "type": "string",
                    "description": (
                        "Only what touched this card, by its board number (#1437). With no window or "
                        f"since, looks back {TICKET_LOOKBACK_DAYS} days."
                    ),
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["history", "receipts", "what-i-did", "decisions", "clicks", "audit", "recap"],
        examples=[
            "what have you done tonight?",
            "what did you create today?",
            "what did I decide about #1437?",
            "did you send anything overnight?",
        ],
    ))
