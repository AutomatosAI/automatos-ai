"""F353 (issue #947): GET /api/deliverables/{id}/thumbnail — a document card's first-page picture.

The picture is served only to the Deliverable's own workspace: the caller's
workspace comes from ``get_request_context_hybrid`` (the same dependency the
Deliverable and its file are served behind), and the Deliverable must be a live
row of THAT workspace in ``v_workspace_outputs``, or the answer is 404. The id is
checked to be a UUID before anything is read, and the stored file's name is
built from it (``store.thumbnail_name``), never taken from the request.

The bytes are streamed from this container's disk or object storage
(``store.load_thumbnail``), never a redirect to a presigned URL, so the browser's
authenticated fetch (``useAuthenticatedBlobUrl``) works the same in both editions.

A plain ``def``: the database read and the storage read are synchronous, so
FastAPI runs it in its threadpool.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from modules.documents.thumbnails.job import load_output
from modules.documents.thumbnails.store import PNG_CONTENT_TYPE, load_thumbnail

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])

NOT_FOUND = "Preview not found"
CACHE_CONTROL = "private, max-age=300"


@router.get("/{deliverable_id}/thumbnail")
def get_deliverable_thumbnail(
    deliverable_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Response:
    """The PNG of the Deliverable's first page; 404 when it has none or is not the caller's."""
    try:
        output_id = UUID(deliverable_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=NOT_FOUND) from None
    if not ctx.workspace_id or load_output(db, ctx.workspace_id, output_id) is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    png = load_thumbnail(ctx.workspace_id, output_id)
    if png is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return Response(content=png, media_type=PNG_CONTENT_TYPE, headers={"Cache-Control": CACHE_CONTROL})
