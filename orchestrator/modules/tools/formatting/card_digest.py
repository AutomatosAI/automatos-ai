"""How much of a call's value an approval card shows (PRD-193 S3), in one place.

The card's digest (``ToolResultFormatter._tool_params_digest``) cuts a long value; the
owner's click signs only what the card shows, so a check that lets a call's words reach
the card on the strength of the click (PRD-256 FX-003, ``follows_the_owner``) reads the
same limit.
"""
from __future__ import annotations

# A value longer than DIGEST_CHARS is cut to its first KEPT_CHARS, ending in "…".
DIGEST_CHARS = 120
KEPT_CHARS = 117
ELLIPSIS = "…"


def shown_on_the_card(text: str) -> str:
    """``text`` as the approval card shows it: whole, or cut to ``DIGEST_CHARS``."""
    return text if fits_on_the_card(text) else text[:KEPT_CHARS] + ELLIPSIS


def fits_on_the_card(text: str) -> bool:
    """Whether the card shows ``text`` whole."""
    return len(text) <= DIGEST_CHARS


__all__ = ["DIGEST_CHARS", "fits_on_the_card", "shown_on_the_card"]
