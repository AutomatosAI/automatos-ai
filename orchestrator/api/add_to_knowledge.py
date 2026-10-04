"""F305 (night 9): "Add to knowledge" on an approved card or a report.

Agent outputs stay out of the owner's documents unless the owner adds them; these two
routes are that one action (services/owner_knowledge.py). Their own module:
api/board_tasks.py is over 800 lines and does not grow. The card's route sits beside
the board's (``missions:update``, like approve); the report's sits beside the Reports
page's, behind the same router-wide lock as its neighbours.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import require_super_admin
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.owner_knowledge import NotFound, Refused, add_card, add_report

router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])
reports_router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Depends(require_super_admin)])
OWNER = "owner"


def _by(ctx: RequestContext) -> str:
    user = getattr(ctx, "user", None)
    return str(getattr(user, "clerk_user_id", None) or getattr(user, "id", None) or OWNER)


def _refused(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/{task_id}/add-to-knowledge", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def add_card_to_knowledge(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The approved card's answer becomes the owner's document. 409 for a card that is
    not approved (or has no answer), 404 for one outside the workspace."""
    try:
        return await add_card(db, ctx.workspace_id, task_id, by=_by(ctx))
    except (NotFound, Refused) as exc:
        raise _refused(exc) from exc


@reports_router.post("/{report_id}/add-to-knowledge")
async def add_report_to_knowledge(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The report's text (without its Execution Metrics) becomes the owner's document.
    409 while its ticket is not approved, 404 for one outside the workspace."""
    try:
        return await add_report(db, ctx.workspace_id, report_id, by=_by(ctx))
    except (NotFound, Refused) as exc:
        raise _refused(exc) from exc
