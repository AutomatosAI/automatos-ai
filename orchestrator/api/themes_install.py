"""PRD-261 A-07b — installed themes, the active theme, the member's choice and the install's versions.

Routes (plan §2.2; B2-wire registers the router, adds the step-up dependency to opt-in,
default and update, and lists the routes in ``reports/route-manifest.json``):

==========================================================  ==================
``GET    /api/themes/installed``                            member
``GET    /api/themes/active``                               member
``PUT    /api/themes/choice``          ``{choice}``         member
``POST   /api/themes/installs/{id}/opt-in``  ``{packageId}``  member (+ step-up)
``PUT    /api/themes/default``         ``{installId|null}``  manage (+ step-up)
``GET    /api/themes/installs/{id}/compare/{packageId}``    manage
``POST   /api/themes/installs/{id}/update``  ``{packageId}``  manage (+ step-up)
``POST   /api/themes/installs/{id}/rollback``               manage
``DELETE /api/themes/installs/{id}``                        manage
==========================================================  ==================

"Member" is ``require_workspace_permission("members:read")``, "manage" is
``"workspace:manage"``. Every route is scoped to the caller's workspace and answers 404,
never 403, for another workspace's install or package. Users are resolved with
``resolve_user_pk``, so the local edition's anonymous owner and a Clerk member both work.
Every handler is a plain ``def`` over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.themes import THEME_CHOICE_LENGTH
from services.themes import updates
from services.themes.choices import NOT_FOUND, ThemeInstallError, set_choice
from services.themes.defaults import resolve_active, set_default
from services.themes.optin import record_opt_in

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/themes", tags=["themes"])

MEMBER = require_workspace_permission("members:read")
MANAGE = require_workspace_permission("workspace:manage")
NO_WORKSPACE = "Choose a workspace first."

T = TypeVar("T")


class ChoiceBody(BaseModel):
    """``PUT /api/themes/choice``: 'classic' | 'studio' | 'builtin:play' | an install id."""

    model_config = ConfigDict(extra="forbid")
    choice: str = Field(min_length=1, max_length=THEME_CHOICE_LENGTH)


class PackageBody(BaseModel):
    """The package an opt-in or an update is for."""

    model_config = ConfigDict(extra="forbid")
    packageId: UUID  # noqa: N815 — the wire name of plan §2.2


class DefaultBody(BaseModel):
    """``PUT /api/themes/default``: an install id, or null to clear the default."""

    model_config = ConfigDict(extra="forbid")
    installId: Optional[UUID]  # noqa: N815 — the wire name of plan §2.2


def _workspace_of(ctx: RequestContext) -> UUID:
    workspace_id = getattr(ctx, "workspace_id", None)
    if workspace_id is None:
        raise HTTPException(status_code=NOT_FOUND, detail=NO_WORKSPACE)
    return workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))


def _run(db: Session, work: Callable[[], T], commit: bool) -> T:
    """Run one service call; commit a write once, and answer a refusal with its own status."""
    try:
        result = work()
        if commit:
            db.commit()
        return result
    except ThemeInstallError as refusal:
        db.rollback()
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal


def _install_answer(db: Session, ctx: RequestContext, install: Any) -> Dict[str, Any]:
    return updates.install_row(db, _workspace_of(ctx), resolve_user_pk(db, ctx), install)


@router.get("/installed")
def list_installed_themes(ctx: RequestContext = Depends(MEMBER), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The workspace's installed themes, each with its opt-in, revocation and drift state."""
    workspace_id = _workspace_of(ctx)
    return {"installs": updates.list_installed(db, workspace_id, resolve_user_pk(db, ctx))}


@router.get("/active")
def get_active_theme(ctx: RequestContext = Depends(MEMBER), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The theme this member sees: their choice, else the workspace default, else Studio."""
    return resolve_active(db, _workspace_of(ctx), resolve_user_pk(db, ctx))


@router.put("/choice")
def put_theme_choice(
    body: ChoiceBody, ctx: RequestContext = Depends(MEMBER), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Store the member's choice in this workspace and answer the active theme it gives."""
    workspace_id, user_pk = _workspace_of(ctx), resolve_user_pk(db, ctx)
    _run(db, lambda: set_choice(db, workspace_id, user_pk, body.choice), commit=True)
    return resolve_active(db, workspace_id, user_pk)


@router.post("/installs/{install_id}/opt-in")
def opt_in_to_theme(
    install_id: UUID, body: PackageBody, ctx: RequestContext = Depends(MEMBER), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Opt the member in to the install's current private package (409 for any other)."""
    workspace_id, user_pk = _workspace_of(ctx), resolve_user_pk(db, ctx)
    state = _run(db, lambda: record_opt_in(db, workspace_id, user_pk, install_id, body.packageId), commit=True)
    return {"installId": str(install_id), "optIn": state}


@router.put("/default")
def put_default_theme(
    body: DefaultBody, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Set or clear the workspace default; an unreviewed theme answers 409 (Q6)."""
    workspace_id = _workspace_of(ctx)
    target = _run(db, lambda: set_default(db, workspace_id, body.installId), commit=True)
    return {"defaultInstallId": None if target is None else str(target.id)}


@router.get("/installs/{install_id}/compare/{package_id}")
def compare_theme_update(
    install_id: UUID, package_id: UUID, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """What moving the install to another version changes: versions, files, scan."""
    workspace_id = _workspace_of(ctx)
    return _run(db, lambda: updates.compare(db, workspace_id, install_id, package_id), commit=False)


@router.post("/installs/{install_id}/update")
def update_theme_install(
    install_id: UUID, body: PackageBody, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Move the install to a newer package; the old one becomes the rollback target."""
    workspace_id = _workspace_of(ctx)
    install = _run(db, lambda: updates.apply_update(db, workspace_id, install_id, body.packageId), commit=True)
    return _install_answer(db, ctx, install)


@router.post("/installs/{install_id}/rollback")
def rollback_theme_install(
    install_id: UUID, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Swap the install back to its previous package."""
    workspace_id = _workspace_of(ctx)
    install = _run(db, lambda: updates.rollback(db, workspace_id, install_id), commit=True)
    return _install_answer(db, ctx, install)


@router.delete("/installs/{install_id}")
def uninstall_theme(
    install_id: UUID, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Soft-uninstall: looks stay, and members who chose it see the workspace default."""
    workspace_id = _workspace_of(ctx)
    install = _run(db, lambda: updates.uninstall(db, workspace_id, install_id), commit=True)
    return {"uninstalled": str(install.id)}
