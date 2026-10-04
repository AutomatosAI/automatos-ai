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

from .markdown_body import blocks_from_markdown, section_text
from .schema import (
    BlockDocument,
    HeadingBlock,
    TableBlock,
    TextBlock,
    TextRun,
)

_counter = {"n": 0}


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
    """The sections, or the body sent as ``content`` alone."""
    sections = data.get("sections") or []
    if isinstance(sections, str):
        sections = [sections]
    if not sections and section_text(data.get("content")):
        sections = [section_text(data.get("content"))]
    return [block for section in sections for block in _section_blocks(section)]


def _metrics(data: Dict[str, Any]) -> List[Any]:
    metrics = data.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return []
    rows = [[[TextRun(text="Metric")], [TextRun(text="Value")]]]
    rows += [[[TextRun(text=str(key))], [TextRun(text=str(value))]] for key, value in metrics.items()]
    return [_heading("Key Metrics", 2), TableBlock(id=_bid("tbl"), header=True, rows=rows)]


def blocks_from_legacy(data: Dict[str, Any]) -> BlockDocument:
    """Build a BlockDocument from the common legacy report-data shape."""
    blocks: List[Any] = []

    if data.get("title"):
        blocks.append(_heading(str(data["title"]), 1))

    meta_bits = []
    if data.get("author"):
        meta_bits.append(f"Author: {data['author']}")
    if data.get("date"):
        meta_bits.append(f"Date: {data['date']}")
    if meta_bits:
        blocks.append(_text_block(" · ".join(meta_bits)))

    blocks += _list_section("Highlights", data.get("highlights"))
    blocks += _metrics(data)
    blocks += _body(data)
    blocks += _list_section("Recommendations", data.get("recommendations"))
    return BlockDocument(blocks=blocks)


__all__ = ["blocks_from_legacy"]
