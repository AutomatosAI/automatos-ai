"""A text block's paragraphs, line breaks, lists and emphasis (F347, night 10b).

F347: a Branded template's body is a chip (``data.body``, ``data.findings``)
filled with what an agent or a person wrote, and both renderers printed it as
one run of text: its blank-line paragraphs ran into one, its ``**twice a
week**`` printed with the asterisks, and its "- " and "1. " lines ran inline.

A text block is now read into :class:`Group` s, the same for the PDF and the
Word file:

* a blank line ends a paragraph; a single line break stays a line break;
* in a chip's value, a line starting "- ", "* ", "+ ", "• " is a bulleted item and
  one starting "1. " or "1) " a numbered item; a run of items is one list;
* in a chip's value, ``**bold**`` / ``__bold__`` and ``*italic*`` / ``_italic_``.

Nothing else is read: no headings, links, images, code or HTML. The text the
template's author typed keeps its own marks and is never read as markdown. The
renderers escape every piece of text as they write it (``body_html`` here, the
DOCX writer adds runs), so a ``<script>`` in a value prints as text.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

TEXT, BREAK, MISSING, ITEM = "text", "break", "missing", "item"
PARAGRAPH, BULLETED, ORDERED = "p", "ul", "ol"
BOLD, ITALIC = "bold", "italic"
# The element a text block of several paragraphs or a list prints as.
TEXT_CLASS = "doc-text"

MARK_TAGS = {
    "bold": ("<strong>", "</strong>"),
    "italic": ("<em>", "</em>"),
    "underline": ("<u>", "</u>"),
    "strike": ("<s>", "</s>"),
    "code": ("<code>", "</code>"),
}

_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d{1,9}[.)])\s+")
_STRONG = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)")
# A text run that only joins its neighbours: spaces, with at most one mark between them.
_SEPARATOR = re.compile(r"^\s*[·•|/,\-–—]?\s*$")
_EMPHASIS = re.compile(r"\*(?=[^\s*])(.+?)(?<=[^\s*])\*|(?<!\w)_(?=[^\s_])(.+?)(?<=[^\s_])_(?!\w)")


@dataclass(frozen=True)
class Seg:
    """A piece of a line: text with its marks, a line break, a list marker, or a chip with no value."""

    text: str
    marks: Tuple[str, ...] = ()
    kind: str = TEXT


@dataclass(frozen=True)
class Group:
    """A paragraph (its lines joined by line breaks) or a list (one line per item)."""

    kind: str
    lines: Tuple[Tuple[Seg, ...], ...] = ()
    start: int = 1


LINE_BREAK = Seg("\n", kind=BREAK)


def _split_lines(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _plain(text: str, marks: Tuple[str, ...]) -> List[Seg]:
    return [Seg(text, marks)] if text else []


def _marked(text: str, pattern: re.Pattern, mark: str, marks: Tuple[str, ...], inner) -> List[Seg]:
    out: List[Seg] = []
    last = 0
    for match in pattern.finditer(text):
        out += inner(text[last:match.start()], marks)
        out += inner(match.group(1) or match.group(2), marks + (mark,))
        last = match.end()
    return out + inner(text[last:], marks)


def _italics(text: str, marks: Tuple[str, ...]) -> List[Seg]:
    return _marked(text, _EMPHASIS, ITALIC, marks, _plain)


def emphasis(text: str, marks: Tuple[str, ...] = ()) -> List[Seg]:
    """One line of a value with its ``**bold**`` and ``*italic*`` read as marks."""
    return _marked(text, _STRONG, BOLD, marks, _italics)


def value_segments(text: str) -> List[Seg]:
    """A chip's value: its lines, the list marker that starts one, and its emphasis."""
    out: List[Seg] = []
    for index, line in enumerate(_split_lines(text)):
        if index:
            out.append(LINE_BREAK)
        marker = _LIST_MARKER.match(line)
        if marker:
            out.append(Seg(marker.group(0), kind=ITEM))
            line = line[marker.end():]
        out += emphasis(line)
    return out


def _literal(text: str, marks: Tuple[str, ...]) -> List[Seg]:
    """Text the author typed: its line breaks kept, never read as markdown."""
    out: List[Seg] = []
    for index, line in enumerate(_split_lines(text)):
        if index:
            out.append(LINE_BREAK)
        out += _plain(line, marks)
    return out


def _chip(path: str, fallback, values: Dict[str, str], missing: List[str]) -> List[Seg]:
    if path in values:
        return value_segments(values[path])
    if fallback is not None:
        return value_segments(fallback)
    missing.append(path)
    return [Seg(path, kind=MISSING)]


def _run_segments(run, values: Dict[str, str], missing: List[str]) -> List[Seg]:
    if run.type == "text":
        return _literal(run.text, tuple(run.marks))
    if run.type == "variable":
        return _chip(run.path, run.fallback, values, missing)
    return []


def _is_separator(run) -> bool:
    return run.type == "text" and bool(_SEPARATOR.match(run.text))


def _has_content(segs: Sequence[Seg]) -> bool:
    return any(seg.kind != TEXT or seg.text.strip() for seg in segs)


def segments(content: Sequence, values: Dict[str, str], missing: List[str]) -> List[Seg]:
    """A block's inline runs as segments; a chip with no value is recorded in ``missing``.

    In a block with a chip, a separator run ("  ·  ", ", ", " | ") prints only between two
    pieces that printed something: "email · · " with no phone or website is "email".
    """
    if not any(run.type == "variable" for run in content):
        return [seg for run in content for seg in _run_segments(run, values, missing)]
    out: List[Seg] = []
    pending: List[Seg] = []
    for run in content:
        if _is_separator(run):
            pending = pending or _run_segments(run, values, missing)
            continue
        segs = _run_segments(run, values, missing)
        if not _has_content(segs):
            continue
        if _has_content(out):
            out += pending
        out += segs
        pending = []
    return out


def _lines(segs: Sequence[Seg]) -> List[Tuple[Seg, ...]]:
    lines: List[List[Seg]] = [[]]
    for seg in segs:
        if seg.kind == BREAK:
            lines.append([])
        else:
            lines[-1].append(seg)
    return [tuple(line) for line in lines]


def _is_blank(line: Tuple[Seg, ...]) -> bool:
    return all(seg.kind == TEXT and not seg.text.strip() for seg in line)


def _shape(line: Tuple[Seg, ...]) -> Tuple[str, int, Tuple[Seg, ...]]:
    """``(kind, number, body)``: a list item when the line starts with a value's list marker."""
    if line[0].kind != ITEM:
        return PARAGRAPH, 1, line
    number = line[0].text.strip()[:-1]
    if number.isdigit():
        return ORDERED, int(number), line[1:]
    return BULLETED, 1, line[1:]


def groups(segs: Sequence[Seg]) -> List[Group]:
    """The segments as paragraphs and lists: a blank line ends one, a change of kind starts the next."""
    built: List[Group] = []
    joining = False
    for line in _lines(segs):
        if _is_blank(line):
            joining = False
            continue
        kind, number, body = _shape(line)
        if joining and built[-1].kind == kind:
            built[-1] = replace(built[-1], lines=(*built[-1].lines, body))
        else:
            built.append(Group(kind=kind, lines=(body,), start=number))
            joining = True
    return built


def block_groups(block, values: Dict[str, str], missing: List[str]) -> List[Group]:
    """A ``text`` or ``variable`` block as paragraphs and lists."""
    if block.type == "variable":
        return groups(_chip(block.path, block.fallback, values, missing))
    return groups(segments(block.content, values, missing))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def unresolved_html(path: str) -> str:
    """The visible marker a chip with no value prints as (PRD-167 S3)."""
    return f'<span class="unresolved-var" data-path="{_esc(path)}">[[{_esc(path)}]]</span>'


def seg_html(seg: Seg) -> str:
    if seg.kind == MISSING:
        return unresolved_html(seg.text)
    text = _esc(seg.text)
    for mark in seg.marks:
        open_tag, close_tag = MARK_TAGS.get(mark, ("", ""))
        text = f"{open_tag}{text}{close_tag}"
    return text


def _line_html(line: Tuple[Seg, ...]) -> str:
    return "".join(seg_html(seg) for seg in line)


def _group_html(group: Group, attrs: str = "") -> str:
    if group.kind == PARAGRAPH:
        return f"<p{attrs}>{'<br />'.join(_line_html(line) for line in group.lines)}</p>"
    items = "".join(f"<li>{_line_html(line)}</li>" for line in group.lines)
    start = f' start="{group.start}"' if group.kind == ORDERED and group.start != 1 else ""
    return f"<{group.kind}{start}>{items}</{group.kind}>"


def body_html(built: Sequence[Group], attrs: str = "") -> str:
    """The groups as HTML, every piece of text escaped, ``attrs`` on the block's element.

    One paragraph (or none) is a ``<p>``, as a text block always printed; more than one
    paragraph or a list is a ``<div class="doc-text">`` holding them, so a style keyed on
    the block (F350's ``data-block``) still reaches all of it."""
    if not built:
        return f"<p{attrs}></p>"
    if len(built) == 1 and built[0].kind == PARAGRAPH:
        return _group_html(built[0], attrs)
    return f'<div class="{TEXT_CLASS}"{attrs}>{"".join(_group_html(group) for group in built)}</div>'


__all__ = [
    "BULLETED", "Group", "MARK_TAGS", "MISSING", "ORDERED", "PARAGRAPH", "Seg",
    "block_groups", "body_html", "emphasis", "groups", "seg_html", "segments", "unresolved_html", "value_segments",
]
