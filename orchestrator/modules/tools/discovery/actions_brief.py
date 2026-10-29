"""PRD-268 US-C19d (§6.11.2, §6.12.6) — ``platform_get_brief``: the person's brief, in any chat.

Auto answers "what needs me?", "brief me" and "what's on today" from the same ``Brief@1``
the brief route answers, in any chat, voice and quick ask included. A read: it claims no
day and posts nothing (``handlers_brief``). Its one parameter, ``section``, is optional.
"""

from .action_registry import ActionDefinition, ActionRegistry

from services.brief.model import SECTIONS

_GET_BRIEF_PARAMETERS = {
    "type": "object",
    "properties": {
        "section": {
            "type": "string",
            "enum": list(SECTIONS),
            "description": (
                "Leave out for the whole brief. Or one section to answer from: 'needs_you' (questions, "
                "approvals and tickets waiting on the person), 'suggestions' (Auto's suggestions), 'today' "
                "(what is scheduled today), 'since' (what finished since their last brief), 'unchecked' "
                "(what Auto couldn't check)."
            ),
        },
    },
    "required": [],
}


def register_brief_actions(registry: ActionRegistry) -> None:
    """Register ``platform_get_brief`` (PRD-268 US-C19d)."""
    registry.register(ActionDefinition(
        name="platform_get_brief",
        description=(
            "The person's brief for today, the same one the app shows: what needs them (questions, "
            "approvals, tickets to review), Auto's open suggestions, what is scheduled today, what finished "
            "since their last brief, and what Auto couldn't check, with exact counts. Use it to answer "
            "'what needs me?', 'brief me', 'what's on today' or 'catch me up'. Answer from it; in voice, "
            "say its spoken words."
        ),
        category="proactive",
        parameters=_GET_BRIEF_PARAMETERS,
        permission_level="read",
        requires_confirmation=False,
        tags=["proactive", "brief", "digest", "needs_you", "today"],
        examples=["what needs me?", "brief me", "what's on today?", "catch me up"],
    ))
