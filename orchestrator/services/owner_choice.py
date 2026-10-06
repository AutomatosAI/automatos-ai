"""F365 (c) (night 10c): a result that asks the owner to choose gets a card to choose on.

Mission step #2145 (Brand Designer, "Present palette options") ended "a call to
action: the owner replies A, B, C or D, then gets an approval card … Still open: the
owner's choice", and the review it wrote said the same. Nobody was asked: F140's test
for an answer that asks the owner (``playbook_owner_ask.owner_question``, which a
ticket's result (F183) and a mission step's (F163) both go through) looks for a
question on the last line or a request for missing information, and this was neither.
The mission closed, and the owner had nowhere to reply.

A result that asks the owner to pick one of a list of lettered or numbered options
("reply A, B, C or D", "pick 1, 2 or 3", "choose one of A–D") is a question to the
owner, and its options are the card's answers. Pure functions only.
"""
from __future__ import annotations

import re
from typing import List, Optional

# A choice the owner is asked to make has a handful of options; more is a list, not a choice.
MAX_CHOICE_OPTIONS = 8
_LETTER, _NUMBER = r"[A-H]", r"[1-8]"
_RANGE_WORDS = r"(?:\s*[–—-]\s*|\s+(?:to|through)\s+)"


def _choices(item: str) -> str:
    """``X, Y or Z`` (or ``X or Y``), or the range ``X–Z``, of one kind of option."""
    return (rf"{item}(?:\s*,\s*{item})*,?\s+or\s+{item}"
            rf"|{item}{_RANGE_WORDS}{item}")


# The verb that hands the choice to the reader, then the options right after it.
_ASKS_TO_CHOOSE = re.compile(
    r"(?i:\b(?:reply|replies|replying|answer|answers|choose|chooses|pick|picks|select|selects"
    r"|tell\s+me|let\s+me\s+know)\b)"
    r"(?i:\s+(?:with|by|either|one\s+of|which\s+of|which|the|option|options))*[\s:,]+"
    rf"(?P<options>{_choices(_LETTER)}|{_choices(_NUMBER)})(?![\w])"
)
_OPTION = re.compile(rf"{_LETTER}|{_NUMBER}")
_IS_RANGE = re.compile(_RANGE_WORDS)


def _expanded(first: str, last: str) -> List[str]:
    """Every option from ``first`` to ``last`` (``A``–``D`` → A, B, C, D); none when backwards."""
    return [chr(code) for code in range(ord(first), ord(last) + 1)]


def choice_options(text: str) -> Optional[List[str]]:
    """The options a result asks the owner to choose between, in order, or ``None``
    when it asks no such choice. ``"the owner replies A, B, C or D"`` → A, B, C, D."""
    found = _ASKS_TO_CHOOSE.search(str(text or "").replace("\u2019", "'"))
    if found is None:
        return None
    named = _OPTION.findall(found.group("options"))
    if _IS_RANGE.search(found.group("options")) and len(named) == 2:
        named = _expanded(named[0], named[1])
    options = list(dict.fromkeys(named))
    return options if 2 <= len(options) <= MAX_CHOICE_OPTIONS else None


__all__ = ["MAX_CHOICE_OPTIONS", "choice_options"]
