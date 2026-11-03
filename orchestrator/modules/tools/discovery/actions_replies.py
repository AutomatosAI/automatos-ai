"""PRD-268 US-C12a (§6.6, K10, K12) — reply chips: ``platform_offer_replies``.

When Auto asks the person to choose, it offers the likely answers as taps; a tap is sent as
the person's own message, and free text stays open. The tool changes nothing (``read``: it
shapes the reply), is ``promoted`` (it reaches every Auto chat turn) and is chat only
(``check_in_lane.CHAT_ONLY_ACTIONS``: never in a check-in, so never on a follow-up's turn).
It carries no ``examples`` (``tests/test_prd232_us005_utterance_corpus.py`` requires each in
the utterance corpus).

CONTRACT GUARD (as ``actions_proactive``): the limits are said in words, since not every
provider reads ``maxLength``; the handler (``handlers_replies``) enforces them.
"""

from .action_registry import ActionDefinition, ActionRegistry
from .handlers_replies import MAX_CHARS, MAX_REPLIES, MAX_WORDS, MIN_REPLIES

_REPLIES_PARAMETERS = {
    "type": "object",
    "properties": {
        "replies": {
            "type": "array",
            "minItems": MIN_REPLIES,
            "maxItems": MAX_REPLIES,
            "items": {"type": "string"},
            "description": (
                f"{MIN_REPLIES} to {MAX_REPLIES} different answers to your question, as the person would say "
                f"them: each 1 to {MAX_WORDS} plain words and at most {MAX_CHARS} characters, no markdown."
            ),
        },
    },
    "required": ["replies"],
}


def register_replies_actions(registry: ActionRegistry) -> None:
    """Register ``platform_offer_replies`` (PRD-268 US-C12a)."""
    registry.register(ActionDefinition(
        name="platform_offer_replies",
        description=(
            "Offer the likely answers to your question as taps under your reply. Call it last, after you have "
            "written the question, and only when the person must choose between a few answers (\"Window or "
            "aisle?\"). Never for an open question, and never on the brief. The person can still answer in "
            "their own words."
        ),
        category="chat",
        parameters=_REPLIES_PARAMETERS,
        permission_level="read",
        requires_confirmation=False,
        promoted=True,
        tags=["chat", "reply", "choice", "quick_reply"],
    ))
