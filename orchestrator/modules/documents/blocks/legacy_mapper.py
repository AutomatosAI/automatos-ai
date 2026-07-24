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

    # Highlights (bulleted)
    highlights = data.get("highlights") or []
    if highlights:
        blocks.append(_heading("Highlights", 2))
        for item in highlights:
            blocks.append(_text_block(f"• {item}"))

    # Metrics table
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        blocks.append(_heading("Key Metrics", 2))
        rows = [[[TextRun(text="Metric")], [TextRun(text="Value")]]]
        for key, value in metrics.items():
            rows.append([[TextRun(text=str(key))], [TextRun(text=str(value))]])
        blocks.append(TableBlock(id=_bid("tbl"), header=True, rows=rows))

    # Sections (title + content)
    for section in data.get("sections") or []:
        if isinstance(section, dict):
            if section.get("title"):
                blocks.append(_heading(str(section["title"]), 2))
            if section.get("content"):
                blocks.append(_text_block(str(section["content"])))
        elif isinstance(section, str):
            blocks.append(_text_block(section))

    # Recommendations (bulleted)
    recs = data.get("recommendations") or []
    if recs:
        blocks.append(_heading("Recommendations", 2))
        for item in recs:
            blocks.append(_text_block(f"• {item}"))

    return BlockDocument(blocks=blocks)


__all__ = ["blocks_from_legacy"]
