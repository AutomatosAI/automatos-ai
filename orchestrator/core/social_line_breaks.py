"""``|`` breaks a line only where a template says so; anywhere else it never prints (F382, night 11).

F382 (7 Oct), B7: the IG story Announcement printed raw ``|`` in its feature text. A
template's display text (an element carrying ``data-dress``: a headline, a quote)
splits on ``|`` into lines, and its field's description tells the model so ("Use |
to break the line"). The model wrote ``|`` into the features too, whose text prints
as it is.

:func:`without_stray_breaks` is applied to a post's values when its bundle is built
(``core/media_render_bundle.build_bundle``): a field whose ``{{ placeholder }}`` the
template shows only as plain text, outside any ``data-dress`` element, has each ``|``
turned into a space (``"Fresh beans | roasted daily"`` prints "Fresh beans roasted
daily"). A field the template puts in an attribute or a script (a data story's rows
are ``|``-separated parts its script splits) is left as it is, and so is every
``data-dress`` field. Pure.
"""
from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Mapping

from core.social_templates import PLACEHOLDER

LINE_BREAK = "|"
# An element that carries data-dress, and what it holds (the templates' display text is one element of text).
_DRESSED = re.compile(r"<(\w+)\b[^>]*\sdata-dress\b[^>]*>(.*?)</\1>", re.S | re.I)
# A placeholder inside a tag (an attribute's value) or a script: that field is the template's own to parse.
_TAG = re.compile(r"<[^>]*>", re.S)
_SCRIPT = re.compile(r"<script\b.*?</script>", re.S | re.I)


def _names(text: str) -> FrozenSet[str]:
    return frozenset(match.group(1) for match in PLACEHOLDER.finditer(text))


def line_break_fields(html: str) -> FrozenSet[str]:
    """The fields whose ``|`` the template reads: its ``data-dress`` text, its attributes and its scripts."""
    dressed = frozenset().union(*(_names(match.group(2)) for match in _DRESSED.finditer(html)))
    in_tags = frozenset().union(*(_names(match.group(0)) for match in _TAG.finditer(html)))
    in_scripts = frozenset().union(*(_names(match.group(0)) for match in _SCRIPT.finditer(html)))
    return dressed | in_tags | in_scripts


def unbroken(text: str) -> str:
    """``text`` with each ``|`` a space, the words around it trimmed."""
    return " ".join(part.strip() for part in text.split(LINE_BREAK) if part.strip())


def without_stray_breaks(values: Mapping[str, Any], html: str) -> Dict[str, Any]:
    """A copy of ``values`` in which a field the template prints as plain text carries no ``|``."""
    kept = line_break_fields(html)
    return {
        name: unbroken(value) if isinstance(value, str) and LINE_BREAK in value and name not in kept else value
        for name, value in values.items()
    }


__all__ = ["LINE_BREAK", "line_break_fields", "unbroken", "without_stray_breaks"]
