"""PRD-256 FX-010 (Decision D7): a ticket Auto files from a person's chat, whose brief
sends, orders, publishes, books or pays, is reviewed by a person before it closes.

Night 12: Auto filed #2318 'Confirm the order with the supplier' from the owner's chat.
``platform_create_task`` defaulted ``review_mode`` to 'auto', so once its agent had
emailed the supplier the ticket closed itself, and nobody saw what went out. Now such a
ticket is filed with ``review_mode: human`` (it waits in Review for the owner) unless the
call named a review_mode, and the receipt says so. The words are the brief's verbs, read
from the title and description: one small list (``SENDS_WORDS``), not a reading of what
the owner said. A ticket an agent files on its own run (no person behind the turn) and a
brief that only drafts keep the default.

P256-FIX-RVW-19: "Email the supplier to confirm delivery", "Post the spring menu on
Instagram" and "Reply to Declan" send too. A message verb (``MESSAGE_VERBS``) counts only
in the brief's verb position, each sentence's first word or the one after "and"/"then",
since as a noun it only drafts ("Draft a reply to Declan", "Draft the email").
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, Tuple

from modules.tools.discovery.send_words import ORDER_WORDS
from modules.tools.execution.call_effects import REVIEWED_BY_YOU as REVIEW_HELD

# The verbs of a brief whose work leaves the workspace: a message, an order, a post, a booking, a payment
# (the order words are the owner's-click gate's, send_words.ORDER_WORDS; the rest a brief's own forms).
SENDS_WORDS = ORDER_WORDS | frozenset({
    "send", "sends", "sending", "ordering", "reorder", "publish", "publishes", "publishing", "books", "pays",
    "paying",
})
# A message sent, as a brief's verb only: as a noun it is what a draft makes.
MESSAGE_VERBS = frozenset({"email", "post", "reply", "message", "forward", "tweet", "text", "dm"})
# The words after which the next is a verb: a brief's second act ("Draft it and email Declan").
VERB_JOINS = frozenset({"and", "then"})
_WORD = re.compile(r"[a-z]+")
_SENTENCE_END = re.compile(r"[.!?;:\n]+")
HUMAN = "human"
REVIEW_MODE = "review_mode"
# Server-injected by the platform executor from the driving user (strip-then-inject): a person drives the turn.
DRIVER = "_user_id"
BRIEF_FIELDS = ("title", "description")
# What the answer tells the model beside its flag (``REVIEW_HELD``; the receipt words it for the owner).
REVIEW_NOTE = ("It waits in Review for the owner: its brief sends or orders, so they see the work before it "
               "closes. Tell them so.")


def sends_or_orders(*texts: Any) -> bool:
    """Whether any of ``texts`` carries a send, order, publish, book or pay word, or a
    message verb (email, post, reply, …) in a verb position."""
    briefs = [text.lower() for text in texts if isinstance(text, str)]
    return any(word in SENDS_WORDS for text in briefs for word in _WORD.findall(text)) or any(
        word in MESSAGE_VERBS for text in briefs for word in _verbs(text))


def _verbs(text: str) -> Iterator[str]:
    """The words of ``text`` in a verb position: each sentence's first, and each one after
    "and" or "then"."""
    for sentence in _SENTENCE_END.split(text):
        words = _WORD.findall(sentence)
        yield from words[:1]
        yield from (word for before, word in zip(words, words[1:]) if before in VERB_JOINS)


def reviewed_by_a_person(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """``params`` with ``review_mode: human`` when a person's chat files a brief that sends
    or orders and the call named no review_mode; and whether it was set."""
    if not params.get(DRIVER) or params.get(REVIEW_MODE) is not None:
        return params, False
    if not sends_or_orders(*(params.get(field) for field in BRIEF_FIELDS)):
        return params, False
    return {**params, REVIEW_MODE: HUMAN}, True


def says_it_is_reviewed(result: Any, held: bool) -> Any:
    """The answer of a ticket ``reviewed_by_a_person`` held for review, saying so (its
    receipt reads ``REVIEW_HELD``); any other answer as it is."""
    if held and isinstance(result, dict) and result.get("success") and result.get(REVIEW_MODE) == HUMAN:
        return {**result, REVIEW_HELD: True, "review_note": REVIEW_NOTE}
    return result


__all__ = ["MESSAGE_VERBS", "REVIEW_HELD", "REVIEW_NOTE", "SENDS_WORDS", "reviewed_by_a_person", "says_it_is_reviewed",
           "sends_or_orders"]
