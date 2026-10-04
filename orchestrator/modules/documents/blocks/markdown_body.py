"""Markdown in a document's body, printed as formatting rather than as marks (F298).

F298 (night 8): generate_document PDFs printed the markdown an agent wrote. On
#0249 the owner asked for a printable checklist and got "## Before the
collection - [ ] Check Thursday's roast… - [ ] Count the boxes…" in one paragraph,
then headings with the "[ ]" jobs run together beneath them; #0394.2's price list
came out as one line, "* Harbour Blend: £19.50 per kilo + VAT * Christmas Blend…".
Both render lanes printed a section's text as it was: the seeded "Basic Report"
(a legacy Jinja template, ``{{ section.content }}``, autoescaped, its newlines
collapsed) and the block fallback a workspace without it uses
(``blocks_from_legacy``, one text block per section).

The text is now read as markdown (Python-Markdown, already a dependency):
headings, paragraphs whose single line breaks stay line breaks, bulleted and
numbered lists one item per line, ``- [ ]`` / ``- [x]`` as a box (☐ / ☑) at the
start of the job's own line, bold and italic without their asterisks, and
tables. Agents write lists straight under a line of text, which strict markdown
would fold into that paragraph, so a list, table or heading line that follows
text starts its own block first.

* :func:`markdown_html` is the HTML a legacy template prints (sanitised with
  bleach: no raw HTML, no images, links only to http(s) and mailto).
* :func:`legacy_render_data` hands a legacy template that HTML for each section.
* :func:`blocks_from_markdown` is the same text as blocks, for the block renderer.
"""
from __future__ import annotations

import re
from itertools import count
from typing import Any, Dict, Iterator, List, Optional, Tuple
from xml.etree.ElementTree import Element

import bleach
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from markupsafe import Markup

from .schema import HeadingBlock, TableBlock, TextBlock, TextRun

UNCHECKED_BOX = "☐"
CHECKED_BOX = "☑"
BULLET = "•"
TASK_ITEM_CLASS = "task-item"
TASK_LIST_CLASS = "task-list"
TASK_BOX_CLASS = "task-box"
# Section titles are h2 in both lanes: a heading inside a section sits below them.
HEADING_SHIFT = 2
MAX_HEADING_LEVEL = 6
# After inline processing (20) and before prettify (10); the capture runs after unescape (0).
TASK_PRIORITY = 15
CAPTURE_PRIORITY = -1

_TASK_PREFIX = re.compile(r"^\s*\[([ xX])\]\s+")
_LIST_LINE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")
_TABLE_LINE = re.compile(r"^\s*\|")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s")
_WRAPPING_P = re.compile(r"^<p>(.*)</p>$", re.S)

ALLOWED_TAGS = frozenset({
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "strong", "em", "b", "i",
    "code", "pre", "blockquote", "del", "s", "table", "thead", "tbody", "tr", "th", "td", "span", "a",
})
ALLOWED_ATTRIBUTES = {
    "a": ["href"], "ul": ["class"], "ol": ["class"], "li": ["class"], "span": ["class"],
}
# Trusted, fixed CSS for the markup above; a legacy template's own styles stay in charge of the rest.
BODY_CSS = (
    "<style>"
    f".doc-md ul.{TASK_LIST_CLASS}{{list-style:none;padding-left:0.2em}}"
    f".doc-md li{{margin:0.2em 0}}"
    f".doc-md .{TASK_BOX_CLASS}{{font-family:'DejaVu Sans','Noto Sans Symbols 2','Segoe UI Symbol',sans-serif;"
    "margin-right:0.2em}"
    ".doc-md table{border-collapse:collapse;margin:0.6em 0}"
    ".doc-md th,.doc-md td{border:1px solid #d0d0d0;padding:0.3em 0.6em}"
    "</style>"
)


# ---------------------------------------------------------------------------
# Reading the text
# ---------------------------------------------------------------------------


def _line_kind(line: str) -> str:
    if not line.strip():
        return "blank"
    if _HEADING_LINE.match(line):
        return "heading"
    if _TABLE_LINE.match(line):
        return "table"
    if _LIST_LINE.match(line):
        return "list"
    return "indented" if line[:1].isspace() else "text"


def _nested(line: str) -> str:
    """A list item indented by one to three spaces under another is nested (markdown wants four)."""
    indent = len(_LIST_LINE.match(line).group(1))
    return " " * 4 + line.lstrip() if 0 < indent < 4 else line


def as_agents_mean(text: str) -> str:
    """The text with each list, table and heading starting its own block, and two-space
    nesting made four. Pure."""
    out: List[str] = []
    previous = "blank"
    for line in text.replace("\r\n", "\n").split("\n"):
        kind = _line_kind(line)
        opens_block = kind in ("list", "table", "heading") and previous == "text"
        leaves_block = kind == "text" and previous in ("list", "table")
        if opens_block or leaves_block:
            out.append("")
        out.append(_nested(line) if kind == "list" and previous == "list" else line)
        previous = kind
    return "\n".join(out)


class _TaskItems(Treeprocessor):
    """``[ ]`` / ``[x]`` at the start of a list item: a box on the job's own line."""

    def run(self, root: Element) -> None:
        for lst in [el for el in root.iter() if el.tag in ("ul", "ol")]:
            if [_mark_task(li) for li in lst if li.tag == "li"].count(True):
                lst.set("class", TASK_LIST_CLASS)


def _mark_task(li: Element) -> bool:
    holder = li[0] if not (li.text or "").strip() and len(li) and li[0].tag == "p" else li
    match = _TASK_PREFIX.match(holder.text or "")
    if not match:
        return False
    box = Element("span", {"class": TASK_BOX_CLASS})
    box.text = CHECKED_BOX if match.group(1) in "xX" else UNCHECKED_BOX
    box.tail = " " + (holder.text or "")[match.end():]
    holder.text = None
    holder.insert(0, box)
    li.set("class", TASK_ITEM_CLASS)
    return True


class _Capture(Treeprocessor):
    """Keep the finished tree for :func:`blocks_from_markdown`."""

    def run(self, root: Element) -> None:
        self.md.captured_tree = root


class _DocumentMarkdown(Extension):
    def extendMarkdown(self, md: markdown.Markdown) -> None:  # noqa: N802 (Python-Markdown's API)
        # An agent's text is never HTML: a '<' is a character, not a tag.
        md.preprocessors.deregister("html_block", strict=False)
        md.inlinePatterns.deregister("html", strict=False)
        md.treeprocessors.register(_TaskItems(md), "f298_task_items", TASK_PRIORITY)
        md.treeprocessors.register(_Capture(md), "f298_capture", CAPTURE_PRIORITY)


def _converted(text: str) -> Tuple[str, Element]:
    md = markdown.Markdown(
        extensions=["tables", "sane_lists", "nl2br", _DocumentMarkdown()],
    )
    html = md.convert(as_agents_mean(text))
    # Python-Markdown skips its processors for blank text, so nothing was captured.
    tree = getattr(md, "captured_tree", None)
    return html, tree if tree is not None else Element("div")


def _clean(html: str) -> str:
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, strip=True)


def markdown_html(text: str) -> Markup:
    """The text as safe HTML for a legacy template to print as it is."""
    return Markup(f'{BODY_CSS}<div class="doc-md">{_clean(_converted(text)[0])}</div>')


def inline_html(text: str) -> Markup:
    """One line (a highlight, a recommendation) with its bold and italic, without a paragraph."""
    html = _clean(_converted(text)[0]).strip()
    match = _WRAPPING_P.match(html)
    return Markup(match.group(1) if match else html)


# ---------------------------------------------------------------------------
# The same text as blocks
# ---------------------------------------------------------------------------

_STASHED = re.compile("\x02[^\x03]*\x03")  # Python-Markdown's placeholders for stashed markup
_MARKS = {"strong": "bold", "b": "bold", "em": "italic", "i": "italic", "code": "code", "del": "strike", "s": "strike"}


def _runs(el: Element, marks: Tuple[str, ...] = ()) -> Iterator[Optional[TextRun]]:
    """The element's text as runs, ``None`` where a line break falls; nested lists are left out."""
    if el.text:
        yield TextRun(text=_STASHED.sub("", el.text), marks=list(marks))
    for child in el:
        if child.tag in ("br", "p"):
            yield None  # a line break, or a paragraph of a loose list item: a new line
        if child.tag not in ("br", "ul", "ol"):
            yield from _runs(child, marks + ((_MARKS[child.tag],) if child.tag in _MARKS else ()))
        if child.tail:
            yield TextRun(text=_STASHED.sub("", child.tail), marks=list(marks))


def _trimmed(line: List[TextRun]) -> List[TextRun]:
    """The line without the spaces a source line break left at its ends."""
    if not line:
        return line
    first = TextRun(text=line[0].text.lstrip(), marks=line[0].marks)
    line = [first, *line[1:]]
    last = TextRun(text=line[-1].text.rstrip(), marks=line[-1].marks)
    return [run for run in [*line[:-1], last] if run.text]


def _lines(el: Element) -> List[List[TextRun]]:
    """The element's text, one list of runs per line it prints on."""
    lines: List[List[TextRun]] = [[]]
    for run in _runs(el):
        if run is None:
            lines.append([])
        elif run.text.strip("\n"):
            lines[-1].append(TextRun(text=run.text.replace("\n", " "), marks=run.marks))
    return [_trimmed(line) for line in lines if "".join(r.text for r in line).strip()]


def _item_prefix(li: Element, ordered: bool, number: int) -> str:
    if li.get("class") == TASK_ITEM_CLASS:
        return ""  # the box is the line's first run
    return f"{number}. " if ordered else f"{BULLET} "


def _list_blocks(lst: Element, ids: Iterator[str]) -> Iterator[TextBlock]:
    for number, li in enumerate((c for c in lst if c.tag == "li"), start=1):
        lines = _lines(li) or [[]]
        prefix = _item_prefix(li, lst.tag == "ol", number)
        first = [TextRun(text=prefix)] if prefix else []
        yield TextBlock(id=next(ids), content=first + lines[0])
        for line in lines[1:]:
            yield TextBlock(id=next(ids), content=line)
        for nested in (c for c in li if c.tag in ("ul", "ol")):
            yield from _list_blocks(nested, ids)


def _table_block(table: Element, ids: Iterator[str]) -> TableBlock:
    rows = [[_cell(cell) for cell in tr if cell.tag in ("th", "td")] for tr in table.iter("tr")]
    return TableBlock(id=next(ids), header=table.find("thead") is not None, rows=rows)


def _cell(cell: Element) -> list:
    return [run for line in _lines(cell) for run in line]


def _element_blocks(el: Element, ids: Iterator[str]) -> Iterator[Any]:
    if re.fullmatch(r"h[1-6]", el.tag):
        level = min(MAX_HEADING_LEVEL, int(el.tag[1]) + HEADING_SHIFT)
        yield HeadingBlock(id=next(ids), level=level, content=[r for line in _lines(el) for r in line])
    elif el.tag in ("ul", "ol"):
        yield from _list_blocks(el, ids)
    elif el.tag == "table":
        yield _table_block(el, ids)
    else:  # a paragraph, a quote, code: each line its own line
        for line in _lines(el):
            yield TextBlock(id=next(ids), content=line)


def blocks_from_markdown(text: str, id_prefix: str = "md") -> List[Any]:
    """The text as heading, text and table blocks: each list item and each line its own block."""
    ids = (f"{id_prefix}-{n}" for n in count(1))
    root = _converted(text)[1]
    return [block for el in root for block in _element_blocks(el, ids)]


# ---------------------------------------------------------------------------
# A legacy template's data, its prose rendered
# ---------------------------------------------------------------------------

# One-line items the seeded legacy templates print in a list (Executive Summary).
LIST_TEXT_FIELDS = ("highlights", "recommendations")


def section_text(content: Any) -> Optional[str]:
    """A section's text: a string as it is, a list of lines as a list. Pure."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(isinstance(item, str) for item in content):
        return "\n".join(item if _LIST_LINE.match(item) else f"- {item}" for item in content)
    return None


def _section_for_template(section: Any) -> Any:
    if isinstance(section, str):
        return {"title": "", "content": markdown_html(section)}
    if not isinstance(section, dict):
        return section
    text = section_text(section.get("content"))
    return {**section, "content": markdown_html(text)} if text is not None else dict(section)


def legacy_render_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``data`` for a legacy Jinja template, its prose rendered from markdown.

    Each section's content becomes HTML the template prints as it is; a body sent
    as ``content`` alone (the seeded "Basic Report" prints only sections) becomes
    one untitled section; highlights and recommendations keep their bold and
    italic. ``data`` itself is left as it was: the widget's markdown is made from it.
    """
    out = dict(data)
    sections = data.get("sections")
    body = section_text(data.get("content"))
    if isinstance(sections, list) and sections:
        out["sections"] = [_section_for_template(section) for section in sections]
    elif body and body.strip():
        out["sections"] = [{"title": "", "content": markdown_html(body)}]
    for key in LIST_TEXT_FIELDS:
        if isinstance(data.get(key), list):
            out[key] = [inline_html(item) if isinstance(item, str) else item for item in data[key]]
    return out


__all__ = [
    "markdown_html", "inline_html", "blocks_from_markdown", "legacy_render_data", "section_text",
    "as_agents_mean", "UNCHECKED_BOX", "CHECKED_BOX",
]
