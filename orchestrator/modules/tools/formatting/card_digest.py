"""How much of a call's value an approval card shows (PRD-193 S3), in one place.

The card's digest (``ToolResultFormatter._tool_params_digest``) cuts a long value; the
owner's click signs only what the card shows, so a check that lets a call's words reach
the card on the strength of the click (PRD-256 FX-003, ``follows_the_owner``) reads the
same limit. The card shows the ask's question too (``carries_the_question``, PRD-256 FX-008).
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


def shown_on_the_card(text: str) -> str:
    """``text`` as the approval card shows it: whole, or cut to ``DIGEST_CHARS``."""
    return text if fits_on_the_card(text) else text[:KEPT_CHARS] + ELLIPSIS


def fits_on_the_card(text: str) -> bool:
    """Whether the card shows ``text`` whole."""
    return len(text) <= DIGEST_CHARS


def carries_the_question(format_for_frontend: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap ToolResultFormatter.format_for_frontend: the chat's approval card
    (``tool_approval``) carries the ask's ``question_md``, the subject and the change
    the owner approves (owner_only._ask), so the card renders it (FX-008)."""
    @functools.wraps(format_for_frontend)
    def wrapped(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        data = format_for_frontend(result, tool_name)
        card = data.get(CARD) if isinstance(data, dict) else None
        asked = result.get(QUESTION) if isinstance(result, dict) else None
        if not isinstance(card, dict) or not asked:
            return data
        return {**data, CARD: {**card, QUESTION: asked}}
    return wrapped


__all__ = ["DIGEST_CHARS", "carries_the_question", "fits_on_the_card", "shown_on_the_card"]
