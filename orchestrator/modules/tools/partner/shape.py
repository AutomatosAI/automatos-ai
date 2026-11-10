"""Partner tools: the part of the partner's answer the model is given (PRD-275).

An operation's ``response`` says where the records are (``root``), which fields to keep
(``keep``, dotted paths) and two caps. Only the listed fields reach the model, a list is cut
to ``max_items``, and what is kept never serialises to more than ``max_chars``: a list loses
records from its end, and a single record that is still too large is refused whole, so the
model never reads half a record.

Pure: no I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from modules.tools.partner.pack import ResponseShape
from modules.tools.partner.request import PartnerRefusal

UNEXPECTED_ANSWER = "partner_unexpected_answer"
ANSWER_TOO_LARGE = "partner_answer_too_large"

# A keep tree: a field maps to the fields kept under it, or to None when it is kept whole.
Tree = Dict[str, Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class Shaped:
    """What the model is given, and what was left out to fit the caps."""

    data: Any
    items: Optional[int] = None        # records kept, when the answer is a list
    total_items: Optional[int] = None  # records the partner sent
    truncated: bool = False


def keep_tree(paths: Sequence[str]) -> Tree:
    """The dotted paths as a tree. A field listed whole wins over a path beneath it."""
    tree: Tree = {}
    for path in sorted(paths, key=len):  # a parent is shorter than its children: it is placed first
        node: Optional[Dict[str, Any]] = tree
        *parents, leaf = path.split(".")
        for part in parents:
            if node is None:
                break
            node = node.setdefault(part, {})
        if node is not None:
            node[leaf] = None
    return tree


def pruned(value: Any, tree: Optional[Tree]) -> Any:
    """``value`` with only the fields ``tree`` keeps; a list is pruned record by record."""
    if tree is None:
        return value
    if isinstance(value, list):
        return [pruned(item, tree) for item in value]
    if isinstance(value, dict):
        return {key: pruned(value[key], below) for key, below in tree.items() if key in value}
    return value


def _dig(body: Any, root: str) -> Any:
    found = body
    for part in (root.split(".") if root else []):
        if not isinstance(found, dict) or part not in found:
            raise PartnerRefusal(UNEXPECTED_ANSWER, f"The partner's answer has no '{root}' field to read.")
        found = found[part]
    return found


def _size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False))


def _fitting(records: List[Any], max_chars: int) -> List[Any]:
    """The longest run of records from the start that serialises within ``max_chars``."""
    kept: List[Any] = []
    used = 2  # the brackets
    for record in records:
        used += _size(record) + 2  # and the separator
        if used > max_chars:
            break
        kept.append(record)
    return kept


def shaped(body: Any, response: ResponseShape) -> Shaped:
    """The answer the model reads: the kept fields of what is at ``root``, within the caps."""
    found = _dig(body, response.root)
    tree = keep_tree(response.keep)
    if isinstance(found, list):
        kept = _fitting([pruned(record, tree) for record in found[:response.max_items]], response.max_chars)
        if found and not kept:
            raise PartnerRefusal(ANSWER_TOO_LARGE, "One record of the partner's answer is larger than this tool "
                                                   "may return.")
        return Shaped(kept, items=len(kept), total_items=len(found), truncated=len(kept) < len(found))
    data = pruned(found, tree)
    if _size(data) > response.max_chars:
        raise PartnerRefusal(ANSWER_TOO_LARGE, "The partner's answer is larger than this tool may return.")
    return Shaped(data)


__all__ = ["ANSWER_TOO_LARGE", "Shaped", "UNEXPECTED_ANSWER", "keep_tree", "pruned", "shaped"]
