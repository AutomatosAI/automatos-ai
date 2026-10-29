"""PRD-268 §6.11.4, US-C20a, K21 — ``GET /api/v1/approval-grants/{grant_id}/view``: a grant's ``ApprovalView@1``.

Who may read it is who may answer it (PRD-266 D16): an owner, admin or editor of the workspace,
or the super admin (``require_workspace_answerer``); a viewer is 403. The grant is read in the
caller's workspace only, so another workspace's id is 404, never 403, and so is a question (it is
answered, never allowed). The view's words are in the caller's language (``Accept-Language``,
W1's K13 chooser). A database that cannot be read is 503, logged with its stack.

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
from services.approval_views import grant_view

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/approval-grants", tags=["approval-grants"])

NOT_FOUND = "Approval grant not found"
UNAVAILABLE = "The approval can't be read right now. Try again in a minute."
HTTP_NOT_FOUND, HTTP_UNAVAILABLE = 404, 503


@router.get("/{grant_id}/view")
def get_approval_view(
    grant_id: int,
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_answerer),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The grant's ApprovalView@1, in the caller's language."""
    try:
        view = grant_view(db, ctx.workspace_id, grant_id, request_locale(accept_language))
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("approval view %s failed", grant_id, exc_info=failure)
        raise HTTPException(status_code=HTTP_UNAVAILABLE, detail=UNAVAILABLE) from failure
    if view is None:
        raise HTTPException(status_code=HTTP_NOT_FOUND, detail=NOT_FOUND)
    return view
