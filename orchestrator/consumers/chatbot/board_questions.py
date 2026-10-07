"""A message about the board, or about a card by its number (F263, F241: night 7b).

- "What's on my board right now, and what's waiting for me?" was classified a
  general question: the turn had no tool at all, and Auto answered from the five
  passages the automatic document search found. It named four done or cancelled
  cards as "in Review", with wrong titles, and missed the five real ones.
- "Approve #0177 …", "Update #0199 with that brief …" and "Agree a better brief …
  then update the card" were handed to the card's own agent (DELEGATE), which had
  neither Auto's board tools nor its permissions. "Give #0192 to the Support Agent"
  went down the ASSIGN lane, whose directive files a new ticket: #0194, a copy.

A message that names a card by its number, asks what is on the board, or tells
Auto to act on "the card" is Auto's to handle with its platform tools
(AutoBrain's fast path), and its answer comes from the board, not from documents
(no retrieval first). A hashtag ("#HarbourBlend") or an order's number ("order #1043")
is not a card's number. One exception since PRD-256 US-010: a card handed to a named
agent is the ASSIGN lane on that card (consumers.chatbot.handoffs), never a copy.
"""
from __future__ import annotations

import re

# The board's own form, "#0177" or "#0188.3" (an order's "#1043" is not one), or a
# number the message says is a card's: "card #1043", "ticket 12", "task 7".
CARD_NUMBER = re.compile(
    r"(?<![\w&#])#0\d{3,5}(?:\.\d{1,3})?\b"
    r"|\b(?:card|ticket|task)\s+(?:number\s+)?#?\d{1,6}(?:\.\d{1,3})?\b",
    re.IGNORECASE,
)
BOARD_STATE = re.compile(
    r"\b(?:on|in) (?:my|the|our) board\b|\bmy board\b|\bwaiting (?:for|on) me\b|\bneeds? me\b"
    r"|\bneeds you\b|\b(?:in|for) (?:my )?review\b|\b(?:cards?|tickets?|tasks?) (?:are |is )?(?:waiting|stuck|failed)\b",
    re.IGNORECASE,
)
BOARD_ACTION = re.compile(
    r"\b(?:approve|reject|cancel|re-?brief|update|redo|assign|give|reassign)\b[^.?!\n]{0,40}"
    r"\b(?:the|this|that) (?:card|ticket)\b"
    r"|\bsend (?:it|them|the card|the ticket) back\b",
    re.IGNORECASE,
)


def names_a_card(text: object) -> bool:
    """Whether ``text`` names a card by its number."""
    return bool(CARD_NUMBER.search(str(text or "")))


def about_the_board(text: object) -> bool:
    """Whether ``text`` names a card, asks about the board, or acts on "the card"."""
    said = str(text or "")
    return names_a_card(said) or bool(BOARD_STATE.search(said)) or bool(BOARD_ACTION.search(said))


__all__ = ["about_the_board", "names_a_card"]
