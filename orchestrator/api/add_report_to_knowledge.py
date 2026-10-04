"""F305 (night 9): "Add to knowledge" on a report (the card's route is api/add_to_knowledge.py).

Its own module so that each mounted router is a module's ``router`` (the route
manifest's rule); behind the Reports page's lock, like its neighbours.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.add_to_knowledge import by_whom, refused_as_http
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import require_super_admin
from core.database.database import get_db
from services.owner_knowledge import NotFound, Refused, add_report

router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Depends(require_super_admin)])


@router.post("/{report_id}/add-to-knowledge")
async def add_report_to_knowledge(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The report's text (without its Execution Metrics) becomes the owner's document.
    409 while its ticket is not approved, 404 for one outside the workspace."""
    try:
        return await add_report(db, ctx.workspace_id, report_id, by=by_whom(ctx))
    except (NotFound, Refused) as exc:
        raise refused_as_http(exc) from exc
