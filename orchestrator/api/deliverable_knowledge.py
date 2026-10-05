"""F354 (5 Oct): "Add to Knowledge" on a Deliverable, and taking it back out.

The owner picks the document (an invoice, a letter); nothing is added on its own
(F305). ``POST`` files the document as the owner's document, once; ``DELETE`` removes
that copy and leaves the Deliverable. Both behind the Documents page's own locks
(upload: ``documents:create``, delete: ``documents:delete``) and scoped to the
caller's workspace: another workspace's Deliverable is 404. Whether one was added
rides the Deliverables list and detail answers (``knowledge_document_id``). The work
is services/owner_knowledge.py; its own module so api/deliverables.py stays the
Outputs feed's reads and each mounted router is a module's ``router``.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.add_to_knowledge import by_whom, refused_as_http
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.owner_knowledge import NotFound, Refused, add_deliverable, remove_deliverable

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])


@router.post("/{deliverable_id}/knowledge", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def add_deliverable_to_knowledge(
    deliverable_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The Deliverable's document becomes the owner's document. 409 for a file that is
    not a document or cannot be read, 404 for one outside the workspace."""
    try:
        return await add_deliverable(db, ctx.workspace_id, deliverable_id, by=by_whom(ctx))
    except (NotFound, Refused) as exc:
        raise refused_as_http(exc) from exc


@router.delete("/{deliverable_id}/knowledge", dependencies=[Depends(require_workspace_permission("documents:delete"))])
def remove_deliverable_from_knowledge(
    deliverable_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Remove the owner's document the Deliverable was added as; the Deliverable stays.
    404 for one outside the workspace."""
    try:
        return remove_deliverable(db, ctx.workspace_id, deliverable_id)
    except NotFound as exc:
        raise refused_as_http(exc) from exc
