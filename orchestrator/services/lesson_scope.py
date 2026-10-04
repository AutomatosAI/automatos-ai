"""A lesson from one card is a rule, never that card's content (F315, night 9).

Night 9: #1871, the Support Agent's "Draft reply to Rosa at Lantern Kitchen — delivery
on 10 kg", was sent back at 13:52: "You've signed it as Lantern Kitchen - that's Rosa's
café, not us. We're Harbourline. Start 'Hi Rosa,' and sign off 'Gerard, Harbourline
Coffee Roasters' as in the brand voice doc." F249's lessons
(``services.ticket_redo.agent_lessons``) carry an agent's corrections on its other cards
into every run of its work. #1849, the Support Agent's general question "Delivery charge
on 10 kg (second opinion)", re-ran at 14:15 with that correction among its standing
notes, and came back as a letter beginning "Hi Rosa". The block's FIGURES_STAY line
("never copy its figure, name or date") lost to a note that said "Start 'Hi Rosa,'".

A correction that names what is particular to its own card, a name in that card's title
or brief that the card being run doesn't share ("Rosa", "Lantern Kitchen"), is about
that card. It stays with it: its own redo carries it, and so does any card that shares
the name. It is never another card's lesson. A rule with no such name ("Leave off
'Perfect!'", "Just the table: no working under it") is carried as before.

F318 (night 9b): the notes still travelled, because every send-back note was a lesson
and most notes about one card's facts name nothing particular to it. #0099's "Does the
30 kg allow for what we lose in the roaster? Where did the figures come from?" came back
on the Watchdog's #1976 ("Addressing your question about roaster loss") and #1991
("roasting loss factor (mentioned in your corrections)"); #0072's "the September margin
sheet is my real costing … £2.87 is per bag … Cerrado and Sumatra go in the blend" on
the BA's #0088 and #1977 ("not per kg, as you corrected"). A standing lesson is now a
note the owner said holds in general ("next time", "always", "never", "every time") or
a note about the form of the work (its length, opening, sign-off, layout, tone, where
its sources are said): ``is_standing``. A note about one card's facts, a question or a
pointer to a paper, stays with its card.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Set

# A name: a capitalised word with a lowercase second letter ("Rosa", "Kirinyaga"), not an
# acronym ("VAT") and not "I"; "Rosa's" is "Rosa".
_NAME = re.compile(r"\b[A-Z][a-z][\w'’-]*")
_POSSESSIVE = re.compile(r"['’]s$")
# Where a capital only starts a sentence: the text's start, or after . ! ? or a line break.
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\s*)$")
# Capitalised mid-sentence on any card, so never one card's own.
_EVERY_CARD = frozenset({"Please", "Hi", "Hello", "Dear", "Thanks", "Auto", "Draft", "Reply", "Mission", "Recipe",
                         "Playbook", "Step", "Ticket", "Card"})
_WORD = re.compile(r"[\w'’-]+")


def particulars(*texts: Optional[str]) -> Set[str]:
    """The names a card's own words carry mid-sentence: its people, places and things."""
    names: Set[str] = set()
    for text in texts:
        for match in _NAME.finditer(text or ""):
            if _SENTENCE_START.search(text[:match.start()]):
                continue
            name = _POSSESSIVE.sub("", match.group(0))
            if name not in _EVERY_CARD:
                names.add(name)
    return names


def is_that_cards(note: str, its_card: Iterable[Optional[str]], this_card: str) -> bool:
    """Whether ``note`` is about the card it was written on (``its_card``: that card's
    title and brief): it names what is particular to that card, and ``this_card``'s
    words share none of what it names (a card about Rosa too still gets it)."""
    named = {name for name in particulars(*its_card) if re.search(rf"\b{re.escape(name)}\b", note or "")}
    shared = {_POSSESSIVE.sub("", word).lower() for word in _WORD.findall(this_card or "")}
    return bool(named) and not any(name.lower() in shared for name in named)


# F318: the owner says a note holds beyond its card ("next time count", "never use Perfect!").
_GENERAL = re.compile(
    r"\b(?:next time|from now on|in (?:the )?future|going forward|always|never|whenever|each time|any ?time|"
    r"as a rule|every (?:time|run|card|answer|draft|email|reply|post|letter|report|one)|"
    r"each (?:figure|number|card|answer|draft)|that(?:'s|’s| is) how (?:every|i want))\b", re.IGNORECASE)
# F318: a note about the work's form: its length, opening and sign-off, layout, tone, where its
# sources are said. Not a figure, a fact, a question about this card's answer or a paper to read.
_FORM = re.compile(
    r"\b\d+\s*(?:[-–—]|to)\s*\d+\s*words\b|\b\d+\s*words\b|\bhow long\b|\blength\b|\btoo (?:long|short|wordy)\b"
    r"|\bsign(?:ed|s|ing)?(?:[ -]?off)?\b|\bsignature\b|\bgreeting\b|\bopen(?:s|ing)? (?:with|on)\b"
    r"|\bstart(?:s|ing)? (?:with|it|the (?:email|reply|draft|answer|post))\b|['\"‘“](?:hi|hello|dear)\b"
    r"|\bfirst line\b|\b(?:put|move|keep|leave|start|end)\b[^.!?\n]{0,40}\b(?:at|on) the (?:start|top|end|bottom)\b"
    r"|\b(?:no|nothing)\b(?: \w+){0,2} (?:before|after|under|above|below) (?:it|the (?:answer|table|email|draft|reply))\b"
    r"|\bjust the (?:email|table|caption|draft|answer|text|post|reply|letter|figures?|numbers?|list)\b"
    r"|\bno (?:working|notes? to me|preamble|intro(?:duction)?|subject line)\b|\bbold\b|\bbullets?\b|\bheadings?\b"
    r"|\blayout\b|\bformat(?:ting)?\b|\bparagraphs?\b|\bsubject lines?\b|\bone line\b|\btone\b|\bbrand voice\b"
    r"|\bbanned\b|\bexclamation|\bemojis?\b|\bleave (?:off|out)\b|\bplain (?:english|words)\b"
    r"|\b(?:less|fewer|drop|cut|lose|without)\s+(?:the\s+)?['\"‘“]|\bsay (?:which|where)\b[^.!?\n]{0,60}\bfrom\b",
    re.IGNORECASE)


def is_standing(note: Optional[str]) -> bool:
    """Whether ``note``, written on one card, is a lesson for the agent's other cards: the
    owner said it holds in general, or it is about the work's form (F318). A note about
    that card's facts, its figures or a paper to read is that card's alone."""
    text = note or ""
    return bool(_GENERAL.search(text) or _FORM.search(text))


__all__ = ["is_standing", "is_that_cards", "particulars"]
