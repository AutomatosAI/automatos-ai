"""A ``data.*`` chip's value as text, a list of plain items as a bulleted list (F356, 5 Oct).

F356: the Meeting Notes starter became a block template, and its data fields are
lists of names and lines (``attendees``, ``agenda``), as its schema has always
said. A chip printed a list as Python wrote it: "['Alice', 'Bob']". A list whose
items are all plain text or numbers now prints one "- item" line per item, which
the text block reads as a bulleted list (``blocks.text_body``, F347). Anything
else prints as before (``amounts.field_text``: an amount key's bare number with
two decimals, else ``str``). Pure.
"""
from __future__ import annotations

from typing import Any

from ..amounts import field_text

BULLET = "- "
PLAIN_ITEMS = (str, int, float)


def _plain_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(item, PLAIN_ITEMS) and not isinstance(item, bool) for item in value
    )


def chip_text(path: str, value: Any) -> str:
    """``value`` as a chip under ``path`` prints it."""
    if _plain_list(value):
        return "\n".join(f"{BULLET}{str(item).strip()}" for item in value)
    return field_text(path, value)


__all__ = ["chip_text"]
