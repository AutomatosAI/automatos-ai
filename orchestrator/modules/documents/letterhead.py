"""A PDF made with no template is under the brand kit's letterhead (F331, night 10).

F331 retest (5 Oct): deliverable 837dd5fa, a session agent's ``generate_document``
with no template, came out in the kit's colours but with no logo, no company name
and no contact line, though the workspace's kit has an uploaded logo, and agents
are told the tool puts the logo on the document. The no-template block render now
starts with the Branded Letter's letterhead (``presets.letterhead``: logo, company
name, contact line) whenever the workspace has a kit.

The letterhead's logo and company name have no fallback, so a kit without a logo
leaves the logo out, and one without a name leaves the name out: an empty chip
would block the whole document (P2-09 S3). A workspace that never set a kit gets
the page it always got.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.documents.blocks import BlockDocument, blocks_from_legacy, validate_blocks
from modules.documents.presets import letterhead

LOGO_BLOCK_ID = "logo"
NAME_BLOCK_ID = "lh-name"


def letterhead_blocks(kit: Optional[Dict[str, Any]], render_kit: Dict[str, Any]) -> List[Any]:
    """The letterhead's blocks for a workspace with ``kit`` (none without one). Pure.

    ``render_kit`` is the kit as the renderer reads it: its ``logo_url`` is the
    uploaded logo inlined, or empty when there is no logo to show.
    """
    from services.brand_rules import document_author

    if not kit:
        return []
    left_out = set()
    if not (render_kit or {}).get("logo_url"):
        left_out.add(LOGO_BLOCK_ID)
    if not document_author(kit):
        left_out.add(NAME_BLOCK_ID)
    return [block for block in validate_blocks(letterhead()).blocks if block.id not in left_out]


async def fallback_blocks(data: Dict[str, Any], db: Any, workspace_id: Any,
                          render_kit: Dict[str, Any]) -> BlockDocument:
    """The no-template document: every key ``data`` carries, under the kit's letterhead."""
    from services.brand_rules import kit_off_loop

    body = blocks_from_legacy(data)
    head = letterhead_blocks(await kit_off_loop(db, workspace_id), render_kit)
    return BlockDocument(blocks=[*head, *body.blocks])


__all__ = ["fallback_blocks", "letterhead_blocks"]
