"""PRD-268 US-C11c (§6.3, K10) — Auto's follow-ups: schedule, list and cancel the person's own.

Auto says "I'll check again at 15:10" and keeps to it. ``platform_schedule_follow_up``
(promoted: it reaches every Auto chat turn) keeps one; ``platform_list_follow_ups`` and
``platform_cancel_follow_up`` are reached through the intent router and
``platform_execute``. They run in a person's own chat only (the handlers,
``handlers_follow_ups``, and the check-in lane's ``CHAT_ONLY_ACTIONS``). None carries
``examples`` (``tests/test_prd232_us005_utterance_corpus.py`` requires each in the
utterance corpus).

CONTRACT GUARD (as ``actions_proactive``): ``required`` lists only what the handler cannot
default. The time is ``in_minutes`` or ``at``: the handler takes exactly one. The limits
are said in words; the handler enforces them.
"""

from .action_registry import ActionDefinition, ActionRegistry

from config import config
from modules.proactive.follow_up_caps import ABOUT_KINDS

# The limits the model is told are the ones the handler holds it to (K9).
MIN_MINUTES = config.PROACTIVE_FOLLOW_UP_MIN_MINUTES
MAX_DAYS = config.PROACTIVE_FOLLOW_UP_MAX_DAYS
REASON_CHARS = config.PROACTIVE_FOLLOW_UP_REASON_CHARS

_SCHEDULE_PARAMETERS = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": (
                f"What to check when the follow-up comes, in one sentence, as you'd say it to yourself; at most "
                f"{REASON_CHARS} characters."
            ),
        },
        "in_minutes": {
            "type": "integer",
            "description": f"How many minutes from now; at least {MIN_MINUTES}. Leave out when you give at.",
        },
        "at": {
            "type": "string",
            "description": (
                f"When, as an ISO time with its offset, e.g. 2026-10-30T15:10:00+01:00; at most {MAX_DAYS} days "
                "ahead. Leave out when you give in_minutes."
            ),
        },
        "about": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(ABOUT_KINDS)},
                "id": {"type": "string", "description": "Its id or number, when it has one."},
            },
            "required": ["kind"],
            "description": (
                "What the follow-up is about, when it is about one thing. A launched Mission, a playbook run or a "
                "ticket is refused: use a watch, which supervises it to a verdict."
            ),
        },
    },
    "required": ["reason"],
}

_LIST_PARAMETERS = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["pending", "all"],
            "description": "'pending' (the default) for the ones still to come; 'all' for those that came or were "
                           "cancelled too.",
        },
    },
    "required": [],
}

_CANCEL_PARAMETERS = {
    "type": "object",
    "properties": {
        "follow_up_id": {
            "type": "integer",
            "description": "The follow-up's number, as platform_schedule_follow_up or platform_list_follow_ups gave it.",
        },
    },
    "required": ["follow_up_id"],
}


def register_follow_up_actions(registry: ActionRegistry) -> None:
    """Register the three follow-up tools (PRD-268 US-C11c)."""
    registry.register(ActionDefinition(
        name="platform_schedule_follow_up",
        description=(
            "Keep your word to check back: when you tell the person you'll look again later (\"I'll check again "
            "at 15:10\"), schedule it here, in their chat. When it comes, you look and either say one short "
            "message here, record a suggestion, or stay silent when nothing changed. Give the reason and "
            "in_minutes or at. Never for a launched Mission, a playbook run or a ticket: use a watch."
        ),
        category="proactive",
        parameters=_SCHEDULE_PARAMETERS,
        permission_level="write",
        requires_confirmation=False,
        promoted=True,
        tags=["proactive", "follow_up", "schedule", "check_back", "remind"],
    ))
    registry.register(ActionDefinition(
        name="platform_list_follow_ups",
        description=(
            "The person's own follow-ups you scheduled: each one's number, time (UTC), reason and status "
            "(pending, fired or cancelled). Pending only unless status is 'all'."
        ),
        category="proactive",
        parameters=_LIST_PARAMETERS,
        permission_level="read",
        requires_confirmation=False,
        tags=["proactive", "follow_up", "list"],
    ))
    registry.register(ActionDefinition(
        name="platform_cancel_follow_up",
        description=(
            "Cancel one of the person's pending follow-ups, by its number, when they no longer want you to check "
            "back or the thing is settled."
        ),
        category="proactive",
        parameters=_CANCEL_PARAMETERS,
        permission_level="write",
        requires_confirmation=False,
        tags=["proactive", "follow_up", "cancel"],
    ))
