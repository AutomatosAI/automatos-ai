"""PRD-261 A-07e — a workspace's theme settings (PRD §5 admin controls, D1-b).

Routes (plan §2.2; B2-wire registers the router, adds the step-up dependency to the PUT and
lists both routes in ``reports/route-manifest.json``):

==============================  ==========================================================
``GET /api/themes/settings``    member: every setting with its default, the platform
                                switch, the caller's install right, the default install
                                and each install's pin
``PUT /api/themes/settings``    manage (+ step-up): ``privateUploads`` ``on|off|null``,
                                ``membersMayChoose`` ``true|false|null``, ``installRole``
                                ``admins|owner|null`` and ``pins``
                                ``[{installId, version|null}]``; null puts a setting back to
                                its default, an absent key leaves it alone
==============================  ==========================================================

"Member" is ``require_workspace_permission("members:read")``, "manage" is
``"workspace:manage"``. Changing who may install also needs the old and the new install
permission, and a pin needs the install permission (``services/themes/settings.py``). An
install of another workspace answers 404. The workspace default is set by
``PUT /api/themes/default`` (A-07b), never here. Users are resolved with
``resolve_user_pk``, so the local edition's anonymous owner and a Clerk member both work.
Both handlers are plain ``def`` over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.themes import THEME_ROLE_LENGTH, THEME_STATUS_LENGTH, THEME_VERSION_LENGTH
from services.themes.choices import NOT_FOUND, ThemeInstallError
from services.themes.settings import (
    CHANGE_INSTALL_ROLE,
    CHANGE_MEMBERS_MAY_CHOOSE,
    CHANGE_PINS,
    CHANGE_PRIVATE_UPLOADS,
    MAX_PINS,
    NO_WORKSPACE,
    apply_settings,
    read_settings,
)

router = APIRouter(prefix="/api/themes", tags=["themes"])

MEMBER = require_workspace_permission("members:read")
MANAGE = require_workspace_permission("workspace:manage")

# The wire names of plan §2.2 -> the keys ``apply_settings`` reads.
SCALAR_FIELDS = {
    "privateUploads": CHANGE_PRIVATE_UPLOADS,
    "membersMayChoose": CHANGE_MEMBERS_MAY_CHOOSE,
    "installRole": CHANGE_INSTALL_ROLE,
}


class PinBody(BaseModel):
    """One install's pin: a version it holds, or null to unpin."""

    model_config = ConfigDict(extra="forbid")
    installId: UUID  # noqa: N815 — the wire name of plan §2.2
    version: Optional[str] = Field(max_length=THEME_VERSION_LENGTH)


class SettingsBody(BaseModel):
    """``PUT /api/themes/settings``: only the keys sent change; null means the default.

    The values are checked by ``services/themes/settings.py``, which answers a bad one with a
    plain 400 message.
    """

    model_config = ConfigDict(extra="forbid")
    privateUploads: Optional[str] = Field(default=None, max_length=THEME_STATUS_LENGTH)  # noqa: N815
    membersMayChoose: Optional[StrictBool] = None  # noqa: N815
    installRole: Optional[str] = Field(default=None, max_length=THEME_ROLE_LENGTH)  # noqa: N815
    pins: Optional[List[PinBody]] = Field(default=None, max_length=MAX_PINS)


def _workspace_of(ctx: RequestContext) -> UUID:
    workspace_id = getattr(ctx, "workspace_id", None)
    if workspace_id is None:
        raise HTTPException(status_code=NOT_FOUND, detail=NO_WORKSPACE)
    return workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))


def _changes_of(body: SettingsBody) -> Dict[str, Any]:
    """The keys the caller sent, as ``apply_settings`` reads them."""
    sent = body.model_fields_set
    changes: Dict[str, Any] = {key: getattr(body, name) for name, key in SCALAR_FIELDS.items() if name in sent}
    if "pins" in sent and body.pins is not None:
        changes[CHANGE_PINS] = tuple((pin.installId, pin.version) for pin in body.pins)
    return changes


@router.get("/settings")
def get_theme_settings(ctx: RequestContext = Depends(MEMBER), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The workspace's theme settings, each with its default, and what this caller may do."""
    return read_settings(db, _workspace_of(ctx), ctx)


@router.put("/settings")
def put_theme_settings(
    body: SettingsBody, ctx: RequestContext = Depends(MANAGE), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Change the workspace's theme settings, all or none, and answer the new settings."""
    workspace_id = _workspace_of(ctx)
    try:
        apply_settings(db, ctx, workspace_id, resolve_user_pk(db, ctx), _changes_of(body))
        db.commit()
    except ThemeInstallError as refusal:
        db.rollback()
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal
    return read_settings(db, workspace_id, ctx)
