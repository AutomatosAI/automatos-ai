"""F311 (night 9): a Markdown document is chunked by its sections, each under its heading.

wholesale-terms-2026.md (document 1526, 1,201 bytes) was stored as 8 chunks of 105 to
225 characters by the topic-coherence chunker, which reads sentences and never sees a
heading. "## Delivery" and the van's days were one chunk; "Carriage is charged per drop:
under 12 kg £8.50, 12 kg and over £5.00" was the next, without its heading and glued to
"## Payment" and its 30 days. Asked what a café pays for delivery on 10 kg, Auto was
handed the Prices and Ordering chunks and said the terms name no delivery charge, twice
in fresh chats (ledger L1, L93).

Here a section is its heading line and everything under it, up to the next heading of
the same or a higher level. A section shorter than ``min_chars`` joins the one after it
(a heading with nothing under it always does), so a title never stands alone. A section
longer than ``max_chars`` is split at its paragraphs, then its lines, and every piece
carries the section's heading. Every chunk also starts with the headings above its
section, so "## Northfield Green Coffee Co." is stored under "# Green coffee — who we
buy from". No line of the document is dropped.

A text with no Markdown heading gives ``[]``: the caller's own chunker runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_HEADING = re.compile(r"^(#{1,6})\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    """One section: the heading lines above it (outermost first) and its own text."""

    above: Tuple[str, ...]
    text: str
    heading: str = ""


@dataclass(frozen=True)
class Piece:
    """A chunk's text and the heading of the section it starts in."""

    content: str
    heading: str


def _heading_level(line: str) -> int:
    """The heading's level (1-6), or 0 when the line is not a heading."""
    match = _HEADING.match(line)
    return len(match.group(1)) if match else 0


def _sections(text: str) -> List[Section]:
    """The document's sections in order; text before the first heading is one too."""
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []
    above: Tuple[str, ...] = ()
    heading, lines, fenced = "", [], False
    for line in text.splitlines():
        fenced = fenced != bool(_FENCE.match(line))
        level = 0 if fenced else _heading_level(line)
        if not level:
            lines.append(line)
            continue
        if "\n".join(lines).strip():
            sections.append(Section(above, "\n".join(lines).strip(), heading))
        stack = [entry for entry in stack if entry[0] < level]
        above, heading, lines = tuple(entry[1] for entry in stack), line.strip(), [line]
        stack.append((level, line.strip()))
    if "\n".join(lines).strip():
        sections.append(Section(above, "\n".join(lines).strip(), heading))
    return sections


def _join_short(sections: Sequence[Section], min_chars: int, max_chars: int) -> List[Section]:
    """A section under ``min_chars`` joins the next one while the two fit in ``max_chars``;
    a heading with nothing under it always does, as a heading above the next one."""
    joined: List[Section] = []
    carry: Optional[Section] = None
    for section in sections:
        if carry is not None and _body_is_empty(carry):
            section = _under(carry, section)
        elif carry is not None and len(carry.text) + len(section.text) + 2 <= max_chars:
            section = Section(carry.above, f"{carry.text}\n\n{section.text}", carry.heading or section.heading)
        elif carry is not None:
            joined.append(carry)
        carry = section if len(section.text) < min_chars else None
        if carry is None:
            joined.append(section)
    if carry is not None:
        joined.append(carry)
    return joined


def _body_is_empty(section: Section) -> bool:
    """A heading with nothing under it."""
    return len(section.text.splitlines()) == 1 and bool(_heading_level(section.text))


def _under(empty: Section, section: Section) -> Section:
    """``section`` with a heading that had nothing under it among the headings above it."""
    if empty.text in section.above:
        return section
    return Section((*section.above, empty.text), section.text, section.heading)


def _split_text(text: str, budget: int) -> List[str]:
    """``text`` in pieces of at most ``budget`` characters, cut at paragraphs, then
    lines, then spaces; a single word longer than the budget is its own piece."""
    if len(text) <= budget:
        return [text]
    for separator in ("\n\n", "\n", " "):
        parts = text.split(separator)
        if len(parts) > 1:
            return _pack(parts, separator, budget)
    return [text]


def _pack(parts: Sequence[str], separator: str, budget: int) -> List[str]:
    """Consecutive parts joined by ``separator`` while they fit in ``budget``."""
    packed: List[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            packed.append(current)
        split = _split_text(part, budget)
        packed.extend(split[:-1])
        current = split[-1]
    if current:
        packed.append(current)
    return packed


def _pieces(section: Section, max_chars: int) -> List[Piece]:
    """The section as chunks: the headings above it, then its text, split when too long."""
    above = "\n".join(section.above)
    prefix = f"{above}\n" if above else ""
    if len(prefix) + len(section.text) <= max_chars:
        return [Piece(prefix + section.text, section.heading)]
    lead = f"{prefix}{section.heading}\n" if section.heading else prefix
    budget = max(max_chars - len(lead), max_chars // 2)
    parts = _split_text(section.text, budget)
    first = [Piece(prefix + parts[0], section.heading)]
    return first + [Piece(lead + part, section.heading) for part in parts[1:]]


def section_chunks(text: str, min_chars: int, max_chars: int) -> List[Piece]:
    """The document's chunks by section, or ``[]`` when it has no Markdown heading."""
    sections = _sections(text or "")
    if not any(section.heading for section in sections):
        return []
    pieces: List[Piece] = []
    for section in _join_short(sections, min_chars, max_chars):
        pieces.extend(_pieces(section, max_chars))
    return pieces
