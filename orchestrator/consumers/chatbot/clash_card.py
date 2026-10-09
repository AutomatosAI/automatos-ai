"""P256-FIX-RVW-33: the card a name clash was about rides to the reply that picks the agent.

Night 12: "Give #1057 to OPS" with two active agents called OPS asked which one (the clash directive,
``addressed_agents.clash_directive``), and the answer, "267, the operations one", was read on its own:
OPS 267's new ticket, so a copy of #1057 was filed and started (F241). Nothing carried #1057 into the
second turn. The card is read here from the owner's message before the answer (the conversation's
history, which the chat route already reads), so no state is held between turns: when that message
handed a card to the name the picked agent carries, the answer hands that card on. An id given after a
ticket clash ("Get OPS to …", no card) carries nothing and still files the ticket.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from consumers.chatbot.addressed_agents import ID_REPLY
from consumers.chatbot.handoffs import handed_card, handed_to

USER, TEXT = "user", "text"


def _first_text(parts: Any) -> str:
    """A message's text as AutoBrain classified it: its first part, when that part is text."""
    first = parts[0] if isinstance(parts, list) and parts else None
    return str(first.get(TEXT) or "") if isinstance(first, dict) and first.get("type") == TEXT else ""


def said_before(history: Sequence[Any]) -> str:
    """The owner's message before their latest one in ``history`` (the turn's own is last), else ''."""
    said = [entry for entry in history or () if isinstance(entry, dict) and entry.get("role") == USER]
    return _first_text(said[-2].get("parts")) if len(said) >= 2 else ""


def clashed_card(assessment: Any, message: Optional[str], before: Optional[str]) -> Optional[str]:
    """The card ``before`` handed to the name the agent ``message`` picks by its id carries ("Give #1057
    to OPS", then "267, the operations one" for OPS 267), else None: the answer is no id reply, picked
    nobody, or the earlier message handed no card to that name."""
    name = str(getattr(assessment, "target_agent_name", "") or "").strip()
    if not name or getattr(assessment, "target_agent_id", None) is None or not ID_REPLY.match(str(message or "")):
        return None
    card, who = handed_card(before), handed_to(before)
    if not card or not who:
        return None
    named = re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])", who.lower())
    return card if named else None


__all__ = ["clashed_card", "said_before"]
