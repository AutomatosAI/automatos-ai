"""Ticket PR fields: the owner takes a wrong pull request off a ticket.

``DELETE /api/v1/tasks/{task_id}/pull-request`` removes ``runtime_ref.pull_request`` with one
statement (``services.ticket_pull_request.clear_pull_request``) and notes it on the ticket, signed
by the owner. After that a person's Approve merges nothing for the ticket, and no later write puts
the pull request back: the ``kept_runtime_keys`` listener carries the row's absence into any
whole-document write too. A running session may report a new one; the owner's removal is noted.

Who may: a person (``done_needs_an_artifact.a_persons_request``, the Done gate's and the merge's
own rule: a Clerk sign-in or a paired phone or browser) with ``missions:update``, on a ticket of
their own workspace. Its own module: api/board_tasks.py is over 800 lines and does not grow.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.done_needs_an_artifact import a_persons_request
from services.ticket_pull_request import clear_pull_request

router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])

ONLY_A_PERSON = "Only a person signed in, or on a paired phone or browser, can take a pull request off a ticket."
NO_PULL_REQUEST = "This ticket has no pull request to remove."
REMOVED = "The pull request is off this ticket. Approving it merges nothing."


@router.delete("/{task_id}/pull-request", dependencies=[Depends(require_workspace_permission("missions:update"))])
def remove_ticket_pull_request(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Take the ticket's pull request off it. A plain ``def``: the session is synchronous (F105)."""
    if not a_persons_request(ctx):
        raise HTTPException(status_code=403, detail=ONLY_A_PERSON)
    url = clear_pull_request(db, task_id=task_id, workspace_id=ctx.workspace_id)
    if url is None:  # nothing was written; the request's session ends with it (get_db)
        raise HTTPException(status_code=404, detail=NO_PULL_REQUEST)
    db.commit()
    return {"task_id": task_id, "removed": True, "url": url or None, "message": REMOVED}
