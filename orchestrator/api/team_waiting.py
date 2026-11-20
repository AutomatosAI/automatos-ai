"""PRD-277 US-504: the Team page's people waiting for a workspace (``/api/workspaces/{id}/team/waiting``).

A person who signs in with the organisation's identity provider and is given no workspace
(``ENTERPRISE_DEFAULT_ROLE=none``, or no access rule that places them) can open nothing: the
session lane answers ``workspace_required`` (US-404). These two routes let an install admin see
them and place one in the workspace whose Team page they are on.

Unless the edition issues sessions (``issues_sessions``: enterprise), each route answers its
declared method with 404 before any authentication or database work, so ``local`` and ``saas``
serve exactly what they served before.

* ``GET /waiting?limit=&offset=`` — a page of the people waiting, oldest first, so whoever has
  waited longest is never pushed off the list (``core.auth.people.waiting_people``): ``user_id``,
  ``email`` (none for an identity with no verified address), ``name`` and when they first signed
  in; with ``total`` (everyone waiting), ``limit`` (1 to ``WAITING_PAGE_MAX``, default
  ``WAITING_PAGE_DEFAULT``) and ``offset``. Never a token, an identity's subject or its issuer.
* ``POST /waiting/{user_id}`` with ``{"role": "admin" | "editor" | "viewer"}`` — makes the person
  an active member of this workspace in that role (``people.give_workspace``), audited
  ``member:added_from_waiting`` in this workspace. 409 when they are no longer waiting, so an admin's
  later change is never overwritten; 404 for no such person or workspace; 400 for any other role.

Both take the Team routes' own checks: the caller's role here must grant ``members:invite``
(``require_workspace_permission``, the invitation's permission, since this grants a person rights in
this workspace) and the caller must be an install admin on the session lane (``_require_admin``,
the check the session routes use): listing people install-wide is an install admin's view. The
POST also needs step-up, as an invitation and a role change do (``STEP_UP``). Plain ``def``.
Registered through ``RouterSpec("api.team_waiting")`` in ``router_manifest.py``.
"""
from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# The session routes' install-admin check and their 404 where no session is issued, shared.
from api.auth_sessions import _require_admin, _sessions_issued
from api.credentials import get_client_ip
from api.team import STEP_UP
from core.auth import people
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.system_role import UnknownPerson
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.workspaces import Workspace

WAITING_PERMISSION = "members:invite"
WAITING_PAGE_DEFAULT = 50
WAITING_OFFSET_MAX = 1_000_000
ROLE_MAX = 32
NO_SUCH_WORKSPACE = "Workspace not found"
NO_SUCH_PERSON = "No such person."

router = APIRouter(prefix="/api/workspaces/{workspace_id}/team", tags=["team"],
                   dependencies=[Depends(_sessions_issued)])


class GiveWorkspace(BaseModel):
    """The role the waiting person is given in this workspace (never owner)."""

    role: str = Field(max_length=ROLE_MAX)


def _usable_workspace(db: Session, workspace_id: str) -> UUID:
    """The path's workspace when it exists and is usable; 404 otherwise."""
    try:
        parsed = UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_WORKSPACE) from None
    found = db.query(Workspace.id).filter(
        Workspace.id == parsed, Workspace.is_active.isnot(False), Workspace.deleted_at.is_(None),
    ).first()
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_WORKSPACE)
    return parsed


def _view(person: people.WaitingPerson) -> Dict[str, Any]:
    first = person.signed_in_first_at
    return {"user_id": person.user_id, "email": person.email, "name": person.name,
            "signed_in_first_at": first.isoformat() if first else None}


@router.get("/waiting", dependencies=[Depends(require_workspace_permission(WAITING_PERMISSION))])
def list_waiting(workspace_id: str,
                 limit: int = Query(WAITING_PAGE_DEFAULT, ge=1, le=people.WAITING_PAGE_MAX),
                 offset: int = Query(0, ge=0, le=WAITING_OFFSET_MAX),
                 ctx: RequestContext = Depends(get_request_context_hybrid),
                 db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A page of the people waiting for a workspace, oldest first, for an install admin on this Team page."""
    _require_admin(ctx)
    _usable_workspace(db, workspace_id)
    page = people.waiting_people(db, limit=limit, offset=offset)
    return {"people": [_view(person) for person in page.people], "total": page.total,
            "limit": limit, "offset": offset}


@router.post("/waiting/{user_id}",
             dependencies=[Depends(require_workspace_permission(WAITING_PERMISSION)), Depends(STEP_UP)])
def give_workspace(workspace_id: str, user_id: int, change: GiveWorkspace, request: Request,
                   ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Make a waiting person a member of this workspace in ``role`` (US-504), audited."""
    actor = _require_admin(ctx)
    workspace = _usable_workspace(db, workspace_id)
    requester = people.Requester(user_id=actor, workspace_id=workspace, ip_address=get_client_ip(request),
                                 user_agent=request.headers.get("user-agent"))
    try:
        people.give_workspace(db, user_id, workspace, change.role, requester)
    except people.MembershipRefused as refused:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(refused)) from None
    except UnknownPerson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_PERSON) from None
    except people.NotWaiting:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=people.NOT_WAITING_MESSAGE) from None
    except people.WorkspaceNotGiven:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=people.WORKSPACE_NOT_GIVEN_MESSAGE) from None
    return {"user_id": user_id, "workspace_id": str(workspace), "role": change.role}


__all__ = ["router"]
