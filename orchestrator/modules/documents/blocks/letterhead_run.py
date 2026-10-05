"""Which blocks open a document as its letterhead (F356, 5 Oct).

F356: the owner asked for a smaller letterhead logo beside the company block,
not above it. The letterhead is ordinary blocks (``presets.letterhead``: the
logo, then the company's name, address and contact line), so both renderers
first take the run of letterhead blocks a document opens with and lay it out
as one unit: the PDF as a row (logo left, company block right, a rule under
both), the Word file as its first page's header.

A run counts only when it holds a company block (``lh-*``); a document that
opens with a logo alone keeps the logo where it was. Pure.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

LOGO_ID = "logo"
COMPANY_IDS = frozenset({"lh-name", "lh-address", "lh-contact"})
LETTERHEAD_IDS = COMPANY_IDS | {LOGO_ID}


def split_letterhead(blocks: Sequence) -> Tuple[List, List]:
    """``(letterhead, rest)``: the leading run of letterhead blocks, when it holds a company block."""
    count = 0
    while count < len(blocks) and getattr(blocks[count], "id", None) in LETTERHEAD_IDS:
        count += 1
    head = list(blocks[:count])
    if not any(block.id in COMPANY_IDS for block in head):
        return [], list(blocks)
    return head, list(blocks[count:])


def logo_of(head: Sequence):
    """The letterhead's logo block, or None."""
    return next((block for block in head if block.id == LOGO_ID), None)


def company_of(head: Sequence) -> List:
    """The letterhead's company blocks, in order."""
    return [block for block in head if block.id in COMPANY_IDS]


__all__ = ["COMPANY_IDS", "LETTERHEAD_IDS", "LOGO_ID", "company_of", "logo_of", "split_letterhead"]
