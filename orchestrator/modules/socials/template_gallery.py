"""PRD-251B Wave 1, US-B102 — the templates the editor's Look gallery shows (B5).

``gallery`` lists the workspace's social templates, of one post format's kind when a
format is given, each with the lengths it declares (``blocks.durations``, US-B104; a
video without a declared list has its root ``data-duration`` as its one length) and
a link to its thumbnail, and the slots a generation toolkit may fill (``footage_slots``,
S1.8: what the editor's AI footage switch asks for, US-B109). A thumbnail is stored like a preview file, under the
template's id, and ``thumbnail_url`` holds its storage KEY; the list presigns it
inline for the configured TTL (D9), so a link never outlives the storage policy.
``missing_thumbnails`` names the templates the backfill still has to render
(``template_thumbnails.py``).

Columns are selected one by one (never the whole row), as the composer's own
selection does: a gallery entry needs no sample data and no tags.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from config import config
from core.models.core import DocumentTemplate
from core.social_templates import SOCIAL_TEMPLATE_FORMATS, SOCIAL_VIDEO, root_duration, slot_generatable
from modules.socials.compose_checks import template_kind
from modules.socials.media_store import MediaStore

logger = logging.getLogger(__name__)

# modules/documents/template_summary.py marks starters with this creator; that module
# is off limits to modules/socials, so the value is pinned here (a test holds them together).
STARTER_CREATOR = "system"
TEXT_FORMAT = "text"
GALLERY_COLUMNS = (
    DocumentTemplate.id, DocumentTemplate.name, DocumentTemplate.description, DocumentTemplate.format,
    DocumentTemplate.blocks, DocumentTemplate.thumbnail_url, DocumentTemplate.created_by, DocumentTemplate.updated_at,
)


def durations_of(blocks: Any, fmt: Optional[str]) -> List[int]:
    """The lengths a video template offers, in seconds: ``blocks.durations`` when it
    declares them (US-B104), else its root ``data-duration``; an image offers none."""
    if fmt != SOCIAL_VIDEO or not isinstance(blocks, dict):
        return []
    declared = blocks.get("durations")
    if isinstance(declared, list):
        clean = sorted({int(v) for v in declared if isinstance(v, int) and not isinstance(v, bool) and v > 0})
        if clean:
            return clean
    root = root_duration(blocks.get("html") or "")
    return [int(round(root))] if root and root > 0 else []


def footage_slots(blocks: Any) -> List[str]:
    """The template's slots a generation toolkit may fill (S1.8), by name: every slot but
    one marked ``\"generate\": false``, which only the workspace's own file fills."""
    slots = blocks.get("slots") if isinstance(blocks, dict) and isinstance(blocks.get("slots"), dict) else {}
    return sorted(name for name, spec in slots.items() if isinstance(spec, dict) and slot_generatable(spec))


def thumbnail_link(store: MediaStore, raw: Optional[str]) -> Optional[str]:
    """A presigned inline link to the stored thumbnail; an http(s) value as it is;
    ``None`` without one or without storage."""
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        return raw
    if not store.configured():
        return None
    try:
        return store.presigned_get(raw, config.SOCIALS_MEDIA_URL_TTL_SECONDS)
    except Exception:  # noqa: BLE001 — a link that cannot be minted hides the thumbnail, never the template
        logger.warning("[Socials] could not presign the thumbnail %s", raw, exc_info=True)
        return None


def entry(row: Any, store: MediaStore) -> Dict[str, Any]:
    blocks = row.blocks if isinstance(row.blocks, dict) else {}
    return {
        "id": str(row.id),
        "name": row.name,
        "description": row.description,
        "format": row.format,
        "kind": "video" if row.format == SOCIAL_VIDEO else "image",
        "sizes": [str(size) for size in (blocks.get("sizes") or [])],
        "durations": durations_of(blocks, row.format),
        "footage_slots": footage_slots(blocks),
        # The template's fields: the editor's Claims and sources card fills them (US-B109).
        "variables_schema": blocks.get("variables_schema") if isinstance(blocks.get("variables_schema"), dict) else {},
        "thumbnail_url": thumbnail_link(store, row.thumbnail_url),
        "is_starter": (row.created_by or "") == STARTER_CREATOR,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _kinds(post_format: Optional[str]) -> List[str]:
    if post_format is None:
        return list(SOCIAL_TEMPLATE_FORMATS)
    kind = template_kind(post_format)
    return [kind] if kind else []


def gallery(db: Any, workspace_id: UUID, post_format: Optional[str] = None, *, store: Optional[MediaStore] = None) -> List[Dict[str, Any]]:
    """The workspace's active social templates for the gallery, by name. A text
    post has no template, so ``text`` lists none."""
    if post_format == TEXT_FORMAT:
        return []
    rows = (
        db.query(*GALLERY_COLUMNS)
        .filter(
            DocumentTemplate.workspace_id == workspace_id,
            DocumentTemplate.format.in_(_kinds(post_format)),
            DocumentTemplate.is_active.isnot(False),
        )
        .order_by(DocumentTemplate.name, DocumentTemplate.id)
        .all()
    )
    store = store or MediaStore()
    return [entry(row, store) for row in rows]


def missing_thumbnails(db: Any, workspace_id: UUID) -> List[UUID]:
    """The active social templates without a thumbnail yet."""
    rows = (
        db.query(DocumentTemplate.id)
        .filter(
            DocumentTemplate.workspace_id == workspace_id,
            DocumentTemplate.format.in_(list(SOCIAL_TEMPLATE_FORMATS)),
            DocumentTemplate.is_active.isnot(False),
            DocumentTemplate.thumbnail_url.is_(None),
        )
        .order_by(DocumentTemplate.name)
        .all()
    )
    return [row.id for row in rows]
