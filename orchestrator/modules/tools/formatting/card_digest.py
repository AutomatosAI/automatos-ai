"""How much of a call's value an approval card shows (PRD-193 S3), in one place.

The card's digest (``ToolResultFormatter._tool_params_digest``) cuts a long value; the
owner's click signs only what the card shows, so a check that lets a call's words reach
the card on the strength of the click (PRD-256 FX-003, ``follows_the_owner``) reads the
same limit. The card shows the ask's question too (``carries_the_question``, PRD-256 FX-008),
and each ticket the call names by its board number (P256-FIX-N13-7, ``CARD_NUMBERS``).
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict

# A value longer than DIGEST_CHARS is cut to its first KEPT_CHARS, ending in "…".
DIGEST_CHARS = 120
KEPT_CHARS = 117
ELLIPSIS = "…"
QUESTION = "question_md"
CARD = "tool_approval"
CARD_NUMBERS = "card_numbers"  # owner_only._ask (discovery.card_ticket_numbers): {"task_id": "#1394"}
PARAMS = "params"


def shown_on_the_card(text: str) -> str:
    """``text`` as the approval card shows it: whole, or cut to ``DIGEST_CHARS``."""
    return text if fits_on_the_card(text) else text[:KEPT_CHARS] + ELLIPSIS


def fits_on_the_card(text: str) -> bool:
    """Whether the card shows ``text`` whole."""
    return len(text) <= DIGEST_CHARS


def carries_the_question(format_for_frontend: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap ToolResultFormatter.format_for_frontend: the chat's approval card
    (``tool_approval``) carries the ask's ``question_md``, the subject and the change
    the owner approves (owner_only._ask), so the card renders it (FX-008), and its digest
    names each ticket by board number, never by row id (P256-FIX-N13-7)."""
    @functools.wraps(format_for_frontend)
    def wrapped(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        data = format_for_frontend(result, tool_name)
        card = data.get(CARD) if isinstance(data, dict) else None
        if not isinstance(card, dict) or not isinstance(result, dict):
            return data
        asked = result.get(QUESTION)
        card = {**card, QUESTION: asked} if asked else card
        return {**data, CARD: _numbered(card, result.get(CARD_NUMBERS))}
    return wrapped


def _numbered(card: Dict[str, Any], numbers: Any) -> Dict[str, Any]:
    """The card with each ticket parameter of its digest shown by board number (a new dict)."""
    shown = card.get(PARAMS)
    if not isinstance(numbers, dict) or not numbers or not isinstance(shown, dict):
        return card
    return {**card, PARAMS: {key: numbers.get(key, value) for key, value in shown.items()}}


__all__ = ["DIGEST_CHARS", "carries_the_question", "fits_on_the_card", "shown_on_the_card"]
