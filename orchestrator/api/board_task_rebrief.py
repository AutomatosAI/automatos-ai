"""PRD-252 R2 (Discuss): "Update ticket and re-queue".

A discussion with Auto ends with a brief the owner agreed. This writes it onto the
ticket as its description, records the owner's correction (the redo leads with
it), keeps the old brief and the last run on record, and hands the ticket back
to its agent, the way Reject does. Its own router: api/board_tasks.py is over
800 lines and does not grow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy.orm import Session

from api.board_tasks import (
    _decide, _operator_ref, _redo, _redo_status, _refreshed, _ticket_for_verdict, already_decided,
    keep_previous_run,
)
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.run_redo import redo_refusal, takes_its_own_redo
from services.ticket_numbers import ticket_label
from services.ticket_redo import BRIEF_AGREED, MAX_BRIEF_CHARS, REBRIEFED, with_correction, with_new_brief

router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])


class RebriefBody(BaseModel):
    # Stripped before the bounds, so a blank brief never empties a ticket.
    brief: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_BRIEF_CHARS)]


@router.post("/{task_id}/rebrief", dependencies=[Depends(require_workspace_permission("missions:update"))])
def rebrief_task(
    task_id: int,
    body: RebriefBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-252 R2 (Discuss): "Update ticket and re-queue". The brief the owner and
    Auto agreed becomes the ticket's description, the owner's correction says so,
    the old brief and the last run stay on record, and the ticket goes back to
    its agent. F243: a playbook's card runs its playbook again with the agreed
    brief, a step of a running mission goes back to its mission with it, and any
    other mission ticket is refused up front (409), saying why and what to do; a
    running ticket is not re-briefed under its run (409)."""
    task = _ticket_for_verdict(db, ctx, task_id)
    rebrief(db, ctx, task, body.brief, by=_operator_ref(ctx))
    _refreshed(db, task, task_id)
    return {"success": True, "task_id": task.id, "status": task.status}


def rebrief(db: Session, ctx: Any, task: Any, brief: str, *, by: str) -> None:
    """The agreed brief becomes ``task``'s description, and the ticket goes back to
    its agent (or its playbook or mission) to work from it. Refused with an
    HTTPException before anything changes. ``ctx`` carries the workspace, and the
    person for a playbook's or a mission's redo. Committed. The board's "Update
    ticket and re-queue" and Auto's platform_update_task (send_back, F241 night 7b)
    both come here."""
    refused = redo_refusal(db, task)
    if refused:
        raise HTTPException(status_code=409, detail=refused)
    if task.status == "in_progress":
        raise HTTPException(status_code=409, detail=f"{ticket_label(task, capital=True)} is running; "
                                                    "re-brief it once it stops.")
    if not task.assigned_agent_id and not takes_its_own_redo(task):
        raise HTTPException(status_code=422, detail="Assign an agent before re-queueing the ticket.")
    seen, at = task.status, datetime.now(timezone.utc).isoformat()
    keep_previous_run(task, why=REBRIEFED, by=by)
    data = with_correction(task.planning_data, BRIEF_AGREED, by=by, at=at)
    # A claim works from raw_prompt, else the description: both carry the new brief.
    task.planning_data = with_new_brief(data, task.raw_prompt or task.description, by=by, at=at)
    task.raw_prompt = task.description = brief
    if not _decide(db, task, seen=seen, values={"status": _redo_status(task)}):
        raise already_decided(task)
    _redo(db, ctx, task, BRIEF_AGREED)
