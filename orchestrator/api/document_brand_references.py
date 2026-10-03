"""Brand kit style references and the style profile (PRD-251B B9; US-B302, US-B303).

Included by ``api/document_brand_kit.py`` (so under ``/api/documents``):

* ``GET /brand-kit/references``: the references (each with its note, stance and image
  route), the style profile Auto read from them, and whether liked images go to AI tools;
* ``POST /brand-kit/references`` (a PNG, JPEG or WebP file, its note and stance): 201;
  413 over the size, 415 not one of the three, 422 when the kit holds the most it may;
* ``PUT /brand-kit/references/{ref_id}``: a new note or stance; ``DELETE``: removed;
* ``GET /brand-kit/references/{ref_id}/image``: the image itself;
* ``POST /brand-kit/style/read``: **Read the references again** (502 when the model's
  answer is not a profile, 504 when it takes too long);
* ``PUT /brand-kit/style``: whether liked images go to AI tools as references.

Any change to the references reads the profile again in the background. Writes need
``workspace:manage``, as the kit's do; every route is the caller's workspace's own, so
another workspace's reference is a 404. Plain ``def`` routes (F105): their database work
runs in the threadpool, the model call and the background read on the event loop.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Dict, Optional

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.workspaces import Workspace
from modules.documents import brand_references as refs
from modules.documents import brand_style

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-generation"])

_MANAGE = Depends(require_workspace_permission("workspace:manage"))
NOT_FOUND = "No such style reference"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceChange(_Strict):
    note: Optional[str] = Field(None, max_length=refs.MAX_NOTE_CHARS)
    stance: Optional[str] = None


class StyleSettings(_Strict):
    send_liked: bool


def _workspace(db: Session, ctx: RequestContext) -> Workspace:
    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def _answer(style: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "references": [refs.listed(ref) for ref in style["references"]],
        "profile": style["profile"],
        "send_liked": style["send_liked"],
        "limits": {"references": refs.MAX_REFERENCES, "bytes": refs.MAX_REFERENCE_BYTES},
    }


def _changed(db: Session, workspace: Workspace, style: Dict[str, Any]) -> Dict[str, Any]:
    """Save the style, then read the profile again in the background."""
    saved = refs.save_style(db, workspace, style)
    anyio.from_thread.run_sync(brand_style.launch_refresh, workspace.id)
    return _answer({**refs.style_of(workspace.settings), **saved})


@router.get("/brand-kit/references")
def list_style_references(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    return _answer(refs.style_of(_workspace(db, ctx).settings))


@router.post("/brand-kit/references", status_code=201, dependencies=[_MANAGE])
def upload_style_reference(
    file: UploadFile = File(...),
    note: str = Form(""),
    stance: str = Form(refs.LIKE),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Store an image as a style reference, with its note and stance (the module docstring)."""
    workspace = _workspace(db, ctx)
    style = refs.style_of(workspace.settings)
    data = file.file.read(refs.MAX_REFERENCE_BYTES + 1)
    try:
        references = refs.add_reference(workspace.id, style["references"], data, note=note, stance=stance)
    except refs.BrandReferenceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    logger.info("[BrandKit] a style reference was added for workspace %s (%d bytes)", workspace.id, len(data))
    return _changed(db, workspace, {**style, "references": references})


@router.put("/brand-kit/references/{ref_id}", dependencies=[_MANAGE])
def update_style_reference(
    ref_id: str, body: ReferenceChange, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    workspace = _workspace(db, ctx)
    style = refs.style_of(workspace.settings)
    try:
        references = refs.update_reference(style["references"], ref_id, body.model_dump(exclude_unset=True))
    except refs.BrandReferenceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    if references is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return _changed(db, workspace, {**style, "references": references})


@router.delete("/brand-kit/references/{ref_id}", dependencies=[_MANAGE])
def delete_style_reference(ref_id: str, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    workspace = _workspace(db, ctx)
    style = refs.style_of(workspace.settings)
    references = refs.remove_reference(style["references"], ref_id)
    if references is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return _changed(db, workspace, {**style, "references": references})


@router.get("/brand-kit/references/{ref_id}/image")
def stream_style_reference(ref_id: str, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Response:
    found = refs.find_reference(refs.style_of(_workspace(db, ctx).settings)["references"], ref_id)
    data = refs.load_reference(found) if found else None
    if not data:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return Response(content=data, media_type=found.get("content_type") or "image/png", headers={"Cache-Control": "private, max-age=300"})


@router.post("/brand-kit/style/read", dependencies=[_MANAGE])
def read_style_again(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """Read the references again now; the profile as stored."""
    workspace = _workspace(db, ctx)
    try:
        anyio.from_thread.run(functools.partial(brand_style.refresh_profile, workspace.id))
    except brand_style.StyleReadFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The model did not answer in time. Try again.") from exc
    db.refresh(workspace)
    return _answer(refs.style_of(workspace.settings))


@router.put("/brand-kit/style", dependencies=[_MANAGE])
def update_style_settings(body: StyleSettings, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    workspace = _workspace(db, ctx)
    saved = refs.save_style(db, workspace, {**refs.style_of(workspace.settings), "send_liked": body.send_liked})
    return _answer(saved)
