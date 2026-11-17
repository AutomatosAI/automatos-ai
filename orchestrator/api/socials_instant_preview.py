"""PRD-251D US-D201 (E8): the instant preview of an image post.

``POST /api/socials/templates/{template_id}/preview-page`` takes what the editor has on
screen (the fields' values, one of the template's sizes, whether the post is a story and the
photo spots it holds a picture for) and answers ``{"html", "fonts_version"}``: the template as
one self-contained page, filled and escaped by the renderer's rules
(``modules/socials/instant_preview.py``). ``GET /api/socials/brand-kit/preview-fonts`` answers
``{"css"}``: the kit's font faces as data: URIs, which the editor fetches once per
``fonts_version`` and puts where the page marks them, so they never travel with a keystroke.

Both answer JSON, never a page of HTML: the page runs only inside the editor's sandboxed frame,
never as a document of the API's origin.

Only a starter the platform ships, unchanged, previews this way (:func:`ships_with_the_platform`).
A template runs its own script to lay itself out, and a frame's sandbox keeps a script from the
app but cannot keep it from sending out what its own page holds (the words on screen, the kit's
logos). So a script that is not the platform's never runs in the editor: a workspace's own
template, or a starter someone edited, is refused and keeps its rendered preview.

Included into the Socials router (``api/socials.py``), so both take the ``/api/socials``
prefix and the Socials gate. Every read is of the caller's workspace: another workspace's
template is 404. The page is for a person who drafts posts (``documents:create``, as the
composer is); the fonts are the kit's, read as the kit itself is. The page reads the kit's
logos and none of its fonts, uploaded or bundled. Plain ``def`` routes (F105): nothing here awaits.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Annotated, Any, Dict, FrozenSet, List, Mapping, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate
from core.media_render_bundle import BUNDLED_FONT_FILES
from core.models.workspaces import Workspace
from core.social_templates import MAX_SLOTS, MAX_TEXT_CHARS, MAX_VARIABLES
from modules.documents.brand_fonts import brand_kit_for_media_render
from modules.documents.brand_kit import get_brand_kit
from modules.documents.social_starters import social_starters
from modules.socials import instant_preview

router = APIRouter()

CAN_DRAFT = Depends(require_workspace_permission("documents:create"))
TEMPLATE_NOT_FOUND = "Template not found"
# The refusal for a template whose script is not the platform's own; the editor shows it under the frame.
STARTERS_ONLY = "only a starter template previews as you type; a template of your own shows after Render preview"
WORKSPACE_NOT_FOUND = "Workspace not found"
# A template field's or slot's name: letters, digits and _, 64 at most (core/social_templates.py).
NAME_MAX_CHARS = 64
# A size is WIDTHxHEIGHT, each four digits at most.
SIZE_MAX_CHARS = 9
TEMPLATE_COLUMNS = (DocumentTemplate.id, DocumentTemplate.format, DocumentTemplate.blocks)

FieldName = Annotated[str, StringConstraints(min_length=1, max_length=NAME_MAX_CHARS)]
FieldValue = Union[bool, int, float, Annotated[str, StringConstraints(max_length=MAX_TEXT_CHARS)]]
SlotName = Annotated[str, StringConstraints(min_length=1, max_length=NAME_MAX_CHARS)]
SizeName = Annotated[str, StringConstraints(min_length=1, max_length=SIZE_MAX_CHARS)]


class PreviewPageRequest(BaseModel):
    """What the editor has on screen. A field it leaves out is empty; unknown keys are refused."""

    model_config = ConfigDict(extra="forbid")

    values: Dict[FieldName, FieldValue] = Field(default_factory=dict, max_length=MAX_VARIABLES)
    size: Optional[SizeName] = None
    story: bool = False
    photos: List[SlotName] = Field(default_factory=list, max_length=MAX_SLOTS)


def _markup_digest(blocks: Any) -> str:
    """sha256 of a composition's markup, its inline script included."""
    html = blocks.get("html") if isinstance(blocks, Mapping) else None
    return hashlib.sha256(str(html or "").encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def starter_markups() -> FrozenSet[str]:
    """The digest of each starter's markup as the platform ships it, read once."""
    return frozenset(_markup_digest(starter["blocks"]) for starter in social_starters())


def ships_with_the_platform(template: Any) -> bool:
    """Whether the template's markup is a starter's, unchanged, so the script it runs is the
    platform's own. A workspace's own template is not, nor is a starter someone edited (the
    seeder puts a platform-owned row back to its seed)."""
    return _markup_digest(getattr(template, "blocks", None)) in starter_markups()


def _template(db: Session, ctx: RequestContext, template_id: UUID) -> Any:
    """The template's format and blocks, from the caller's workspace only: 404 otherwise."""
    template = (
        db.query(*TEMPLATE_COLUMNS)
        .filter(DocumentTemplate.id == template_id, DocumentTemplate.workspace_id == ctx.workspace_id)
        .first()
    )
    if template is None:
        raise HTTPException(status_code=404, detail=TEMPLATE_NOT_FOUND)
    return template


def _workspace(db: Session, ctx: RequestContext) -> Workspace:
    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail=WORKSPACE_NOT_FOUND)
    return workspace


@router.post("/templates/{template_id}/preview-page", dependencies=[CAN_DRAFT])
def social_template_preview_page(
    template_id: UUID,
    body: PreviewPageRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, str]:
    """The template as one self-contained page for the words on screen (US-D201).

    ``html`` is the page and ``fonts_version`` names the kit's faces it expects at its font
    marker (``GET /brand-kit/preview-fonts``). 404 for a template of another workspace; 422,
    saying why, for a template that is not a starter the platform ships, or not an image of one
    still, a value that does not fit its field, a size or a photo spot the template does not
    have, or a composition a render refuses.
    """
    template = _template(db, ctx, template_id)
    if not ships_with_the_platform(template):
        raise HTTPException(status_code=422, detail=STARTERS_ONLY)
    workspace = _workspace(db, ctx)
    stored_kit = get_brand_kit(workspace.settings)
    # The logos and tokens a render reads. No font goes into the page's bundle: the uploaded ones are
    # left unread and the bundled ones are dropped (the editor has both, from the fonts route).
    brand_kit = {**brand_kit_for_media_render({**stored_kit, "font_files": []}), BUNDLED_FONT_FILES: []}
    request = instant_preview.PreviewRequest(values=body.values, size=body.size, story=body.story, photos=tuple(body.photos))
    try:
        page = instant_preview.preview_page(
            template, brand_kit, request, workspace_id=workspace.id, fallback_name=workspace.name or "",
        )
    except instant_preview.PreviewRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"html": page, "fonts_version": instant_preview.fonts_version(stored_kit)}


@router.get("/brand-kit/preview-fonts")
def social_preview_fonts(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, str]:
    """The kit's font faces as ``@font-face`` rules with data: URIs (US-D201): the uploaded
    faces and the bundled ones a render stages, for the instant preview's font marker."""
    workspace = _workspace(db, ctx)
    brand_kit = brand_kit_for_media_render(get_brand_kit(workspace.settings))
    return {"css": instant_preview.font_faces(brand_kit)}
