"""F305 (night 9): "Add to knowledge" on an approved card or a report; PRE-11 (7 Oct):
and taking it back out.

Agent outputs stay out of the owner's documents unless the owner adds them; these
routes are that one action (services/owner_knowledge.py). Their own modules:
api/board_tasks.py is over 800 lines and does not grow. The card's routes are here;
the report's in api/add_report_to_knowledge.py.

Who may (Gerard, 7 Oct): a workspace owner or admin, and the platform super-admin;
in the local edition the operator, whose session is the super-admin
(core/auth/hybrid.py). An editor, viewer or member is refused, so both verbs sit
behind ``require_workspace_admin`` (core/auth/workspace_admin.py, fail-closed).
Another workspace's card is 404. Whether a card was added rides the board's own
answers (``knowledge_document_id``, services/board_task_view.py).
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from services.owner_knowledge import NotFound, Refused, add_card, remove_card

router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])
OWNER = "owner"


def by_whom(ctx: RequestContext) -> str:
    user = getattr(ctx, "user", None)
    return str(getattr(user, "clerk_user_id", None) or getattr(user, "id", None) or OWNER)


def refused_as_http(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/{task_id}/add-to-knowledge", dependencies=[Depends(require_workspace_admin)])
async def add_card_to_knowledge(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The approved card's answer becomes the owner's document, once (adding again
    answers ``already_added``). 409 for a card that is not approved (or has no
    answer), 404 for one outside the workspace."""
    try:
        return await add_card(db, ctx.workspace_id, task_id, by=by_whom(ctx))
    except (NotFound, Refused) as exc:
        raise refused_as_http(exc) from exc


@router.delete("/{task_id}/add-to-knowledge", dependencies=[Depends(require_workspace_admin)])
def remove_card_from_knowledge(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Remove the owner's copy of the card's answer; the card stays. 404 for one
    outside the workspace."""
    try:
        return remove_card(db, ctx.workspace_id, task_id)
    except NotFound as exc:
        raise refused_as_http(exc) from exc
