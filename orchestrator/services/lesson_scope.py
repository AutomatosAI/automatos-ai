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


__all__ = ["is_that_cards", "particulars"]
