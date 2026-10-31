"""PRD-268 §6.11.4, US-C20a, US-C24, K21 — ``GET /api/v1/approval-grants/{grant_id}/view``: the ``ApprovalView@1``
its id names.

The id is the one the view carries as ``grant_id``: an approval grant's number, or a Mission
plan's id (a UUID: a plan has no grant row), so a phone opens either view from a card or a push
with the id it holds (discovery ``approval_view``). Who may read it is who may answer it (PRD-266
D16): an owner, admin or editor of the workspace, or the super admin
(``require_workspace_answerer``; the same roles hold ``missions:update``, which approves a plan);
a viewer is 403. The view is read in the caller's workspace only, so another workspace's grant or
Mission is 404, never 403, and so are a question (it is answered, never allowed) and an id that
names nothing. The view's words are in the caller's language (``Accept-Language``, W1's K13
chooser). A database that cannot be read is 503, logged with its stack.

A plain ``def`` on the sync session (a read; nothing is awaited), registered through
``RouterSpec("api.approval_views")`` in ``router_manifest.py``. The grant verbs stay in
``api/approval_grants.py``, untouched; ``services.approval_views`` builds the view.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_answerer
from core.database.database import get_db
from modules.proactive.locale import request_locale
from services.approval_views import view_of

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/approval-grants", tags=["approval-grants"])

NOT_FOUND = "Approval not found"
UNAVAILABLE = "The approval can't be read right now. Try again in a minute."
HTTP_NOT_FOUND, HTTP_UNAVAILABLE = 404, 503


@router.get("/{grant_id}/view")
def get_approval_view(
    grant_id: str,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_answerer),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The ApprovalView@1 of the grant, or the Mission plan, ``grant_id`` names, in the caller's language."""
    try:
        view = view_of(db, ctx.workspace_id, grant_id, request_locale(accept_language))
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("approval view %r failed", grant_id, exc_info=failure)
        raise HTTPException(status_code=HTTP_UNAVAILABLE, detail=UNAVAILABLE) from failure
    if view is None:
        raise HTTPException(status_code=HTTP_NOT_FOUND, detail=NOT_FOUND)
    return view
