"""PRD-261 A-09a — ``GET/PUT /api/themes/looks/{themeId}``: a member's own looks for one theme.

Plan §2.2: gate "member" (``members:read``, the gate the authZ sweep classifies for per-user
writes), sweep bucket (a). Every read and write is scoped to the caller's workspace and to the
caller's own ``users.id`` (``resolve_user_pk``, never ``ctx.user.id``), so another member's
looks are never reachable, and a theme that isn't installed in the caller's workspace is 404
(also when another workspace has it). Both editions: the local edition's anonymous owner
resolves to the operator's row; SaaS resolves the Clerk user.

``themeId`` is the theme's ``publisher/slug`` (``automatos/play`` for builtin Play), so the
path parameter takes the slash. Plain ``def`` routes: the schema read blocks on storage.

Errors are ``{"detail": "<one plain sentence>", "errors": [{code, path, message}]}``; ``path``
names the field (``looks[2].glow``, ``active.name``). A body that isn't
``{looks: [objects], active: object | null}`` is FastAPI's 422 before any look is read.

Registered by B2-wire (``main.py``, ``reports/route-manifest.json``), not here (PA-5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.themes.looks import LooksError, read_looks, resolve_look_source, save_looks

router = APIRouter(prefix="/api/themes/looks", tags=["themes"])

MEMBER_PERMISSION = "members:read"
NO_PERSON = "Looks are kept for a signed-in person, and this request has none. Sign in and try again."
NO_WORKSPACE = "Open a workspace first: looks are kept per workspace."


class LooksUpdate(BaseModel):
    """The body's shape; each look is checked against the theme's own schema in the service.

    Both fields are required (``active`` may be null): a save replaces the whole list, so a
    body without ``looks`` must never empty it.
    """

    model_config = ConfigDict(extra="forbid")

    looks: List[Dict[str, Any]]
    active: Optional[Dict[str, Any]]


def _refusal(error: LooksError) -> JSONResponse:
    content: dict = {"detail": error.detail}
    if error.errors:
        content["errors"] = error.errors
    return JSONResponse(status_code=error.status, content=content)


def _caller(db: Session, ctx: RequestContext) -> Tuple[UUID, int]:
    """The caller's workspace and own ``users.id``; LooksError when either is missing."""
    workspace_id = getattr(ctx, "workspace_id", None)
    if workspace_id is None:
        raise LooksError(400, NO_WORKSPACE)
    user_id = resolve_user_pk(db, ctx)
    if user_id is None:
        raise LooksError(403, NO_PERSON)
    return UUID(str(workspace_id)), user_id


@router.get("/{theme_id:path}", dependencies=[Depends(require_workspace_permission(MEMBER_PERMISSION))])
def get_theme_looks(
    theme_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Any:
    """The caller's saved looks and active look for ``theme_id``, fitted to its current schema."""
    try:
        workspace_id, user_id = _caller(db, ctx)
        source = resolve_look_source(db, workspace_id, theme_id)
        return read_looks(db, workspace_id, user_id, source)
    except LooksError as error:
        return _refusal(error)


@router.put("/{theme_id:path}", dependencies=[Depends(require_workspace_permission(MEMBER_PERMISSION))])
def put_theme_looks(
    theme_id: str,
    body: LooksUpdate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Any:
    """Replace the caller's looks for ``theme_id``: ``{looks: [...], active: {...} | null}``."""
    try:
        workspace_id, user_id = _caller(db, ctx)
        source = resolve_look_source(db, workspace_id, theme_id)
        return save_looks(db, workspace_id, user_id, source, body.model_dump())
    except LooksError as error:
        return _refusal(error)
