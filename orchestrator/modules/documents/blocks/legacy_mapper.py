"""Legacy report-data → blocks mapping (PRD-167 S2).

Converts the common legacy document-data shape (``title`` / ``author`` / ``date`` /
``sections`` / ``metrics`` / ``highlights`` / ``recommendations``) into a
:class:`BlockDocument`. This is the "legacy-JSON → blocks" direction the PRD calls for:
it lets a structural legacy template render through the canonical block path, and is the
basis for migrating the structural seed templates.

Note: blocks v1 models static + scalar-variable content. The array-driven seeds
(Invoice line-items, multi-section reports) are *generated* from their per-call data via
this mapper at render time; templates a user *authors* in the editor use variable chips
instead. See PRD-167-S1 memo for the editor-independence rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .data_details import extra_blocks
from .markdown_body import blocks_from_markdown, section_text
from .schema import (
    BlockDocument,
    HeadingBlock,
    TableBlock,
    TextBlock,
    TextRun,
)

_counter = {"n": 0}
# The byline prints these two; every other key the report shape leaves out prints as a detail.
BYLINE_KEYS = ("author", "date")


def _bid(prefix: str) -> str:
    _counter["n"] += 1
    return f"{prefix}-{_counter['n']}"


def _text_block(text: str) -> TextBlock:
    return TextBlock(id=_bid("t"), content=[TextRun(text=text)])


def _heading(text: str, level: int) -> HeadingBlock:
    return HeadingBlock(id=_bid("h"), level=level, content=[TextRun(text=text)])


def _list_section(heading: str, items: Any) -> List[Any]:
    """A titled list (highlights, recommendations): one line per item, its bold kept."""
    if not isinstance(items, list) or not items:
        return []
    text = section_text([str(item) for item in items])
    return [_heading(heading, 2), *blocks_from_markdown(text, _bid("md"))]


def _section_blocks(section: Any) -> List[Any]:
    """F298: a section's text is markdown, read into headings, lines and tables."""
    if isinstance(section, str):
        return blocks_from_markdown(section, _bid("md"))
    if not isinstance(section, dict):
        return []
    blocks = [_heading(str(section["title"]), 2)] if section.get("title") else []
    content = section.get("content")
    text = section_text(content) if section_text(content) is not None else str(content or "")
    return blocks + (blocks_from_markdown(text, _bid("md")) if text.strip() else [])


def _body(data: Dict[str, Any]) -> List[Any]:
    """The body sent as ``content``, then the sections (F331: one no longer hides the other)."""
    sections = data.get("sections") or []
    if not isinstance(sections, list):
        sections = [sections]
    body = section_text(data.get("content"))
    texts = ([body] if body and body.strip() else []) + sections
    return [block for section in texts for block in _section_blocks(section)]


def _metrics(data: Dict[str, Any]) -> List[Any]:
    metrics = data.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return []
    rows = [[[TextRun(text="Metric")], [TextRun(text="Value")]]]
    rows += [[[TextRun(text=str(key))], [TextRun(text=str(value))]] for key, value in metrics.items()]
    return [_heading("Key Metrics", 2), TableBlock(id=_bid("tbl"), header=True, rows=rows)]


def _byline(data: Dict[str, Any]) -> List[Any]:
    meta_bits = []
    if data.get("author"):
        meta_bits.append(f"Author: {data['author']}")
    if data.get("date"):
        meta_bits.append(f"Date: {data['date']}")
    return [_text_block(" · ".join(meta_bits))] if meta_bits else []


def blocks_from_legacy(data: Dict[str, Any]) -> BlockDocument:
    """Build a BlockDocument from the common legacy report-data shape, and every other key it carries.

    F331: a key the report shape has no place for (an invoice's number, customer
    and line items) is printed too (``data_details``), never dropped.
    """
    head = [_heading(str(data["title"]), 1)] if data.get("title") else []
    highlights = _list_section("Highlights", data.get("highlights"))
    metrics = _metrics(data)
    body = _body(data)
    recommendations = _list_section("Recommendations", data.get("recommendations"))
    parts = {"highlights": highlights, "metrics": metrics, "sections": body, "content": body,
             "recommendations": recommendations}
    printed = frozenset({"title", *BYLINE_KEYS} | {key for key, blocks in parts.items() if blocks})
    details, structured = extra_blocks(data, printed, _bid)
    blocks = [*head, *_byline(data), *details, *highlights, *metrics, *body, *structured, *recommendations]
    return BlockDocument(blocks=blocks)


__all__ = ["blocks_from_legacy"]
