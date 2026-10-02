"""PRD-251B Wave 1, US-B102 — ``GET /api/socials/templates``: the editor's gallery.

Each of the workspace's social templates with its kind, sizes, the lengths it
declares and a presigned link to its thumbnail (``modules/socials/template_gallery``).
``?format=`` narrows the list to one post format's kind; ``text`` lists none.

When a renderer and storage are configured, a list whose templates lack thumbnails
starts the backfill off the request (``template_thumbnails.start_backfill``); the
answer never waits for it. The brand kit the renders need lives in
``modules/documents``, which ``modules/socials`` may not import, so it is handed in
from here, as the post render's is.

Included into the Socials router (``api/socials.py``), so it takes the ``/api/socials``
prefix and the Socials gate; a plain ``def`` (F105).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.socials import SOCIAL_POST_FORMATS
from modules.documents.brand_fonts import brand_kit_for_media_render
from modules.documents.brand_kit import get_brand_kit
from modules.socials import template_gallery, template_thumbnails
from modules.socials.media_store import MediaStore

router = APIRouter()


def _brand_kit_of(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The workspace brand kit, render-ready (the API layer's job, as api/socials.py does)."""
    return brand_kit_for_media_render(get_brand_kit(settings))


def backfill_possible(store: MediaStore) -> bool:
    return bool(config.SOCIALS_RENDER_URL) and store.configured()


@router.get("/templates")
def list_social_templates(
    format: Optional[str] = Query(None, description="A post format: only the templates of its kind"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> List[Dict[str, Any]]:
    """The workspace's social templates for the gallery (US-B102)."""
    if format is not None and format not in SOCIAL_POST_FORMATS:
        raise HTTPException(status_code=422, detail=f"format must be one of {list(SOCIAL_POST_FORMATS)}")
    store = MediaStore()
    rows = template_gallery.gallery(db, ctx.workspace_id, format, store=store)
    if backfill_possible(store):
        missing = template_gallery.missing_thumbnails(db, ctx.workspace_id)
        template_thumbnails.start_backfill(ctx.workspace_id, missing, brand_kit_of=_brand_kit_of)
    return rows
