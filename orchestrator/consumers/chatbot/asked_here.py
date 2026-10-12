"""P256-FIX-T5 (F314): work the owner asks for here comes back in Auto's reply, not on a card.

Benchmark 11 Oct (eval F314-a): "Draft the intro for this fortnight's Harbour Log newsletter, about the
October box. Just the draft here, please." was filed as card #1312 for HARBOUR LOG on Haiku, Sonnet and
Opus 5.5. Two readings sent it there: the hand-off table read "newsletter" as customer paperwork, whose
note tells Auto not to write it and to file it for the team; and the tiers could take the newsletter's
name, which is HARBOUR LOG's too, for a named assignee. PRD-256 US-010 (thesis D): Auto answers unless
an agent is named or the work is long; "here", "in the chat", "just the draft", "reply with it",
"paste it" and "show me" ask for the text in the reply.

One rule (``asks_for_it_here``), read in the hand-off lane: ``handoffs.the_lane`` reads it after a
hand-over by name or id and before the delegate default, so the turn is Auto's to answer in the reply
(``answered_here``: RESPOND, with ``HERE_DIRECTIVE``); the paperwork row reads it too, so its note never
files the ticket. Work handed to an agent by name ("Have HARBOUR LOG draft it"), long work (a mission,
several agents, an organ) and an ask that also puts it on a card ("draft it and put it on a card")
stay as they were. Brand work is design, not text: its row keeps the Brand designer's ticket (F362).
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

# The text in the reply: "just the draft", "in the chat", "in your reply", "reply with it", "paste it",
# "show me" closing what is asked ("…and show me", not "show me my playbooks"), or "here" closing it
# ("Draft it here, please."; "Just the draft here.").
_IN_THE_REPLY = re.compile(
    r"\b(?:just|only)\s+(?:the\s+)?(?:draft|text|copy|wording|words)\b"
    r"|\b(?:in|into)\s+(?:the|this|our)\s+chat\b|\bin\s+(?:your|the|this)\s+reply\b"
    r"|\breply\s+with\s+(?:it|them|that|the)\b|\bpaste\s+(?:it|them|that|the)\b"
    r"|\bshow\s+me\b(?=\s*(?:$|[,.!?;)]|please\b|first\b|it\b|the\s+(?:draft|text|copy|wording|words)\b))",
    re.IGNORECASE,
)
_HERE = re.compile(r"\bhere\b(?=\s*(?:$|[,.!?;)]|please\b|thanks\b|thank\s+you\b|in\s+the\s+chat\b))", re.IGNORECASE)
# "The new barista is here.", "Come over here": a "here" that says where someone is asks for nothing.
NOT_AN_ASK_BEFORE_HERE = frozenset({"is", "are", "was", "were", "am", "be", "been", "s", "re", "come", "came",
                                    "coming", "over", "from", "out", "get", "got"})
# How far back the word before "here" is read: a word, never the whole message.
WORD_WINDOW = 40
# A "here" that opens a sentence ("Make the invoice. Here, the amounts: …") closes no ask.
_SENTENCE_BREAK = re.compile(r"[.!?;:\n]")
# The owner asks for a card as well: "draft it and put it on a card", "on the board", "file it",
# "create a card for it and show me".
_ON_A_CARD = re.compile(r"\b(?:on|onto|as)\s+(?:a|an|the)\s+(?:card|ticket)\b|\bon(?:to)?\s+the\s+board\b"
                        r"|\bfile\s+(?:it|them|that|this)\b"
                        r"|\b(?:make|create|add|open|raise|file)\s+(?:(?:a|an|the|new)\s+){0,2}(?:card|ticket|task)s?\b",
                        re.IGNORECASE)
# A card the owner refuses is no ask for one: "no need to create a card", "don't file it", "without a ticket".
_NO_CARD = re.compile(r"\b(?:no|not|never|without|don['’]?t|do\s+not)\b(?:\s+[\w'’]+){0,4}?\s+"
                      r"(?:cards?|tickets?|tasks?|board|file)\b", re.IGNORECASE)
LONG_WORK = frozenset({"organ", "organism"})
REASONING_HERE = "PRD-256 P256-FIX-T5: the owner asked for it here, so Auto answers in the reply"
HERE_DIRECTIVE = (
    "\n\n## Manager directive — the owner asked for it here\n"
    "The owner asked for this to come back here, in the chat. Write the work itself (the draft, the text, "
    "the answer) in full in this reply. Do NOT file it as a card or hand it to an agent: no "
    "platform_create_task, no platform_assign_task. Read what you need with your tools first if you must, "
    "then give the owner the text.\n"
)


def _closing_here(said: str) -> bool:
    """Whether a "here" closes what is asked ("Draft it here, please."), not where someone is."""
    for found in _HERE.finditer(said):
        window = said[max(0, found.start() - WORD_WINDOW):found.start()]
        before = _SENTENCE_BREAK.split(window)[-1].split()
        word = re.split(r"['’]", before[-1].lower())[-1] if before else ""
        if word and word not in NOT_AN_ASK_BEFORE_HERE:
            return True
    return False


def asks_for_it_here(text: object) -> bool:
    """Whether the owner asks for the work to come back in the reply, and not also on a card. Pure."""
    said = str(text or "")
    if not said.strip() or _ON_A_CARD.search(_NO_CARD.sub(" ", said)):
        return False
    return bool(_IN_THE_REPLY.search(said)) or _closing_here(said)


def is_long_work(assessment: Any) -> bool:
    """Whether the tiers read the ask as long work (a mission, several agents, an organ): it keeps its lane."""
    from consumers.chatbot.auto import Action

    complexity = getattr(getattr(assessment, "complexity", None), "value", None)
    return (assessment.action == Action.MISSION or bool(getattr(assessment, "needs_multi_agent", False))
            or complexity in LONG_WORK)


def answered_here(answered: Any) -> Any:
    """Auto's answer (``handoffs.answered_by_auto``: RESPOND, its platform tools kept) given in the reply:
    no named target, the here directive."""
    return replace(
        answered, target_agent_id=None, target_agent_name=None,
        context_directive=HERE_DIRECTIVE, reasoning=f"{answered.reasoning} ({REASONING_HERE})",
    )


__all__ = ["HERE_DIRECTIVE", "answered_here", "asks_for_it_here", "is_long_work"]
