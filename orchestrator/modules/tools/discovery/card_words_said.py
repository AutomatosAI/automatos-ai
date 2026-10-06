"""The cards the owner names in words, and the words they give for one (F309, night 9).

Night 9 (build 13): the night-8 card note and guard (``owner_turn.CARD_REF``) read only
the board's own form, "#0201". The owner wrote "card 1879", "Card 1869" and "card 27.2":

- "Please approve card 1879 with this note: …" became platform_approve_mission
  {mission_id: 1879}: "I couldn't find a mission with ID 1879";
- "Send card 27.2 back …" became platform_get_social_post {post_id: "27.2"};
- "Card 1869 needs to go back. Correction: …" got no note at all, and Auto asked the
  owner a question (ask #1460) instead of sending the card back.

So a card named with a card word counts too: "card 1879", "ticket 0022", "card 27.2",
"step 27.2". A step ("27.2") or a leading zero ("0022") is the board's number
(#0027.2, #0022). Bare digits are read as the ticket tools read them
(``services.ticket_numbers.read_bare_refs``): the card with that number in this
workspace, or the card with that id when no card has that number (Gerard, 7 Oct). A number with no card word before it is never a card
("10 kg", "120 boxes"), and "step 2" with no card's number is a mission's step, not one.

The owner's words for a card are what follows their label: "Correction: …", "with this
note: …", "send #0347 back: …". The card note and the guard quote them, so the call
carries them word for word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sqlalchemy.orm import Session

# "card 1879", "ticket 0022", "card 27.2". Not "card #0022": that is owner_turn.CARD_REF's.
WORD_REF = re.compile(r"\b(?:card|ticket)\s+(?:no\.?\s*|number\s+)?(\d{1,6}(?:\.\d{1,3})?)\b(?![.,]\d)", re.I)
# "step 27.2": a mission's step, said by its card's number and its place.
STEP_REF = re.compile(r"\bstep\s+(\d{1,6}\.\d{1,3})\b(?![.,]\d)", re.I)
# What labels the owner's words for a card: "Correction: …", "with this note: …", "back: …".
LABELLED = re.compile(r"(?:\b(?:corrections?|notes?|feedback|comments?)|\bwith (?:this|the|my|that|these)"
                      r"(?: \w+){0,2}|\bback)\s*:\s*", re.I)
MIN_WORDS_CHARS = 3


@dataclass(frozen=True)
class SaidRef:
    """A card named in words: as written ("card 1879"), and the digits ("1879", "27.2")."""
    said: str
    digits: str


def word_refs(text: str) -> List[SaidRef]:
    """The cards ``text`` names with a card word, in the order written, each once."""
    found = sorted([*WORD_REF.finditer(text or ""), *STEP_REF.finditer(text or "")], key=lambda m: m.start())
    refs: List[SaidRef] = []
    for match in found:
        ref = SaidRef(said=match.group(0), digits=match.group(1))
        if all(ref.digits != seen.digits for seen in refs):
            refs.append(ref)
    return refs


def card_said(db: Session, workspace_id: Any, digits: str) -> Tuple[Optional[int], bool]:
    """The id of the card ``digits`` names in this workspace, and whether it was named
    by its id (no card has that number); (None, False) when no card fits."""
    from services.ticket_numbers import format_number, read_bare_refs, resolve_ticket_ref

    if "." in digits or (len(digits) > 1 and digits.startswith("0")):
        return resolve_ticket_ref(db, workspace_id, digits), False
    task_id = read_bare_refs(db, workspace_id, [digits]).get(int(digits))
    by_id = task_id is not None and resolve_ticket_ref(db, workspace_id, format_number(int(digits))) is None
    return task_id, by_id


def owners_card_words(text: str) -> str:
    """The words the owner gave for a card, after their label ("Correction: …"), as
    they wrote them; "" when they labelled none."""
    match = LABELLED.search(text or "")
    words = (text[match.end():] if match else "").strip()
    return words if len(words) >= MIN_WORDS_CHARS else ""


__all__ = ["SaidRef", "card_said", "owners_card_words", "word_refs"]
