"""The parts of a document that have nothing to print are left out (F356, 5 Oct).

F356: the owner asked for optional parts on the starters (a report's KPI tiles,
a proposal's pricing total, a meeting's decisions). A block template has no
conditions, so an optional part is one whose chips fall back to "" and whose
table may stay empty (``empty_text=""``). Without this, such a part still
printed its heading over nothing, or a "Total" row with no amount.

Both renderers ask these questions:

* a **section** whose every child has nothing to print is left out, heading
  and all;
* a **table row** that holds chips, every one of them empty, is left out (a row
  of literal text alone is kept).

A chip is empty only when it has no value and its fallback is "": a chip with
no value and no fallback is unresolved and still blocks the document (P2-09 S3).
Pure.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..variables.catalog import walk_dynamic


def _chip_empty(path: str, fallback: Optional[str], values: Dict[str, str]) -> bool:
    return path not in values and fallback == ""


def _runs_blank(content: Iterable[Any], values: Dict[str, str]) -> bool:
    """No literal text and no chip with a value."""
    for run in content:
        if run.type == "text" and run.text.strip():
            return False
        if run.type == "variable" and not _chip_empty(run.path, run.fallback, values):
            return False
    return True


def _table_blank(block: Any, data: Optional[Dict[str, Any]]) -> bool:
    rows = walk_dynamic(data or {}, block.path)
    return (not isinstance(rows, list) or not rows) and block.empty_text == ""


def block_is_blank(block: Any, values: Dict[str, str], data: Optional[Dict[str, Any]] = None) -> bool:
    """Whether ``block`` would print nothing: an empty optional line, table or section."""
    kind = block.type
    if kind in ("text", "heading"):
        return _runs_blank(block.content, values)
    if kind == "variable":
        return _chip_empty(block.path, block.fallback, values)
    if kind == "data_table":
        return _table_blank(block, data)
    if kind == "section":
        return bool(block.children) and all(block_is_blank(child, values, data) for child in block.children)
    return False


def row_is_blank(row: Iterable[Iterable[Any]], values: Dict[str, str]) -> bool:
    """Whether a table row holds chips and every one of them is empty."""
    chips = [run for cell in row for run in cell if run.type == "variable"]
    return bool(chips) and all(_chip_empty(run.path, run.fallback, values) for run in chips)


__all__ = ["block_is_blank", "row_is_blank"]
