"""PRD-256 FX-008: how an approval card writes what it approves, line by line.

A value is shown on one line (its whitespace and line breaks run together) and cut where
the card's digest cuts a value (``card_digest.shown_on_the_card``), so a long brief or
persona is a line, not a page. An empty value is said, never left blank: "description:
Runs the club desk → (empty)" is how the owner sees a wipe before they click.
"""
from __future__ import annotations

import json
from typing import Any, List

from modules.tools.formatting.card_digest import shown_on_the_card

EMPTY = "(empty)"
LIST_JOIN = ", "
LINE = "- {text}"
CHANGE = "{field}: {old} → {new}"
SAID = "{field}: {new}"


def shown(value: Any) -> str:
    """``value`` as one line of the card: a list joined, an object as JSON, empty said."""
    if isinstance(value, (list, tuple, set)):
        text = LIST_JOIN.join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = "" if value is None else str(value)
    text = " ".join(text.split())
    return shown_on_the_card(text) if text else EMPTY


def change_line(field: str, old: Any, new: Any) -> str:
    """"- description: Runs the club desk → (empty)"."""
    return LINE.format(text=CHANGE.format(field=field, old=shown(old), new=shown(new)))


def said_line(field: str, new: Any) -> str:
    """A value the call adds, with nothing before it: "- goal: Plan the spring menu"."""
    return LINE.format(text=SAID.format(field=field, new=shown(new)))


def value_line(text: str) -> str:
    """A line of the card as it is: "- mission: 'Spring menu' (#0188)"."""
    return LINE.format(text=text)


def question(act: str, lines: List[str]) -> str:
    """The card's question: what it asks ("Change an agent on 'CLUB DESK' (agent #12)"),
    then each line of the change."""
    head = f"{act[:1].upper()}{act[1:]}" if act else ""
    if not lines:
        return f"{head}."
    return "\n".join([f"{head}:", *lines])


__all__ = ["EMPTY", "change_line", "question", "said_line", "shown", "value_line"]
