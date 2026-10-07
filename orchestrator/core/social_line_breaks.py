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
``data-dress`` field. The template is read with the standard library's HTML parser, the way
a browser reads it (``</script >`` and ``</SCRIPT foo>`` end a script too), never with a
regex over the markup. Pure.
"""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from core.social_templates import PLACEHOLDER

LINE_BREAK = "|"
DRESS_ATTR = "data-dress"
# Elements whose text the template's own code reads, not the page: a placeholder there is the template's to parse.
RAW_TEXT_TAGS = frozenset({"script", "style"})
# Elements with no end tag: they never hold text, so they are never on the open-element stack.
VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
                       "track", "wbr"})


def _names(text: str) -> FrozenSet[str]:
    return frozenset(match.group(1) for match in PLACEHOLDER.finditer(text))


class _KeptFields(HTMLParser):
    """Collects the placeholders in attributes, in scripts and styles, and inside ``data-dress`` elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.kept: set = set()
        self._open: List[Tuple[str, bool]] = []  # (tag, whether it is dressed or raw text)

    def _inside(self) -> bool:
        return any(reads for _tag, reads in self._open)

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        for _name, value in attrs:
            self.kept.update(_names(value or ""))
        if tag not in VOID_TAGS:
            reads = tag in RAW_TEXT_TAGS or any(name == DRESS_ATTR for name, _value in attrs)
            self._open.append((tag, reads))

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        for _name, value in attrs:
            self.kept.update(_names(value or ""))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index][0] == tag:
                self._open = self._open[:index]
                return

    def handle_data(self, data: str) -> None:
        if self._inside():
            self.kept.update(_names(data))


def line_break_fields(html: str) -> FrozenSet[str]:
    """The fields whose ``|`` the template reads: its ``data-dress`` text, its attributes and its scripts."""
    parser = _KeptFields()
    parser.feed(html or "")
    parser.close()
    return frozenset(parser.kept)


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
