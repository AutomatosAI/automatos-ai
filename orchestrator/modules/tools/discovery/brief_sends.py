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

P256-FIX-RVW-27: "Please email the supplier", "Also post the menu", "Contact the supplier",
"Reach out to Declan", "Let Declan know", "Get back to the customer", "Notify the team" and
"Ask Sam to email Declan" kept auto. A clause's verb is now read past its lead-in words
(``LEAD_INS``, ``LEAD_IN_PHRASES``) and, after "ask/get/have <someone> to", the verb that
follows; the outbound verbs and phrases join the lists.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Tuple

from modules.tools.discovery.send_words import ORDER_WORDS
from modules.tools.execution.call_effects import REVIEWED_BY_YOU as REVIEW_HELD

# The verbs of a brief whose work leaves the workspace: a message, an order, a post, a booking, a payment
# (the order words are the owner's-click gate's, send_words.ORDER_WORDS; the rest a brief's own forms).
SENDS_WORDS = ORDER_WORDS | frozenset({
    "send", "sends", "sending", "ordering", "reorder", "publish", "publishes", "publishing", "books", "pays",
    "paying",
})
# A message sent, as a brief's verb only: as a noun it is what a draft makes ("Draft the invite").
MESSAGE_VERBS = frozenset({
    "email", "post", "reply", "message", "forward", "tweet", "text", "dm", "contact", "notify", "respond", "invite",
    "submit", "share",
})
# Outbound phrases: a verb and the words straight after it ("reach out", "get back to") ...
OUTBOUND_PHRASES = (("reach", "out"), ("get", "back", "to"))
# ... and a verb whose clause ends the message after its object ("let Declan know").
TELL_PHRASES = (("let", "know"),)
# A verb that hands the act to someone: its message verb follows "to" ("Ask Sam to email Declan").
ASKS = frozenset({"ask", "get", "have"})
TO = "to"
# The words before a clause's verb that are not its verb ("Please email", "Can you post").
LEAD_INS = frozenset({"please", "also", "just", "now", "kindly", "then", "first"})
LEAD_IN_PHRASES = (("remember", "to"), ("make", "sure", "to"), ("can", "you"), ("could", "you"))
# The words after which the next is a verb: a brief's second act ("Draft it and email Declan").
VERB_JOINS = frozenset({"and", "then"})
_WORD = re.compile(r"[a-z]+")
_SENTENCE_END = re.compile(r"[.!?;:\n]+")
_JOIN = re.compile(r"\b(?:%s)\b" % "|".join(sorted(VERB_JOINS)))
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
    message verb or phrase (email, post, reach out, …) in a clause's verb position."""
    briefs = [text.lower() for text in texts if isinstance(text, str)]
    return any(word in SENDS_WORDS for text in briefs for word in _WORD.findall(text)) or any(
        _messages(clause) for text in briefs for clause in _clauses(text))


def _clauses(text: str) -> Iterator[List[str]]:
    """The clauses of ``text``, each sentence split at "and"/"then", as their words from
    the verb on (the lead-in words dropped)."""
    for sentence in _SENTENCE_END.split(text):
        for clause in _JOIN.split(sentence):
            yield _without_lead_in(_WORD.findall(clause))


def _without_lead_in(words: List[str]) -> List[str]:
    """``words`` from the first one that is not a lead-in ("please", "can you", …)."""
    while words:
        if words[0] in LEAD_INS:
            words = words[1:]
            continue
        phrase = next((p for p in LEAD_IN_PHRASES if tuple(words[:len(p)]) == p), None)
        if phrase is None:
            return words
        words = words[len(phrase):]
    return words


def _messages(words: List[str]) -> bool:
    """Whether a clause, read from its verb, sends a message: a message verb, an outbound
    phrase, or an ask whose verb after "to" does."""
    if not words:
        return False
    verb, rest = words[0], words[1:]
    if verb in MESSAGE_VERBS or _outbound_phrase(words):
        return True
    if verb in ASKS and TO in rest:
        return _messages(_without_lead_in(rest[rest.index(TO) + 1:]))
    return False


def _outbound_phrase(words: List[str]) -> bool:
    """Whether ``words`` open with an outbound phrase ("reach out", "let Declan know")."""
    return any(tuple(words[:len(phrase)]) == phrase for phrase in OUTBOUND_PHRASES) or any(
        words[0] == verb and end in words[1:] for verb, end in TELL_PHRASES)


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
