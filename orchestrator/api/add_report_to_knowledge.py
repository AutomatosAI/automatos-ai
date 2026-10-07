"""F305 (night 9): "Add to knowledge" on a report (the card's routes are
api/add_to_knowledge.py); PRE-11 (7 Oct): and taking it back out.

Its own module so that each mounted router is a module's ``router`` (the route
manifest's rule). It no longer sits behind the Reports page's super-admin lock:
Gerard (7 Oct) lets a workspace owner or admin add a report, and the platform
super-admin; in the local edition the operator (the super-admin session). Both
verbs sit behind ``require_workspace_admin`` (fail-closed). Another workspace's
report is 404.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.add_to_knowledge import by_whom, refused_as_http
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from services.owner_knowledge import NotFound, Refused, add_report, remove_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/{report_id}/add-to-knowledge", dependencies=[Depends(require_workspace_admin)])
async def add_report_to_knowledge(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The report's text (without its Execution Metrics) becomes the owner's document,
    once. 409 while its ticket is not approved, 404 for one outside the workspace."""
    try:
        return await add_report(db, ctx.workspace_id, report_id, by=by_whom(ctx))
    except (NotFound, Refused) as exc:
        raise refused_as_http(exc) from exc


@router.delete("/{report_id}/add-to-knowledge", dependencies=[Depends(require_workspace_admin)])
def remove_report_from_knowledge(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Remove the owner's copy of the report; the report stays. 404 for one outside
    the workspace."""
    try:
        return remove_report(db, ctx.workspace_id, report_id)
    except NotFound as exc:
        raise refused_as_http(exc) from exc
