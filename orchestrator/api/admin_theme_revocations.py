"""PRD-261 C-05e — the admin's theme revocations: list, create, lift.

Routes (C3-wire registers the router, lists the routes in ``reports/route-manifest.json`` and
adds the two mutating ones to the authZ sweep's ``ADMIN_GATED_IN_HANDLER`` bucket):

============================================================  ==========================================
``GET  /api/admin/themes/revocations``                         every row, live first, then the newest
``POST /api/admin/themes/revocations``                         ``{digest | themeId [+ versionRange],
                                                               level, reason}``; 201, or 200 for a repeat
``POST /api/admin/themes/revocations/{revocationId}/lift``     sets ``liftedAt`` on a ``local`` row (a
                                                               ``feed`` one while sync is off or stale)
============================================================  ==========================================

Every handler checks ``caller_is_admin`` (``core/auth/roles.py``) first, through
``_require_admin(ctx)`` (the authZ sweep reads it in the handler); the local edition's operator
is a super admin, so the panel works there too. Create and lift also need a fresh step-up after
the admin check (A-07c, ``STEP_UP_MAX_AGE_MINUTES``; the ``admin_themes.py`` ``ADMIN_STEP_UP``
pattern): ``require_revocation_admin`` runs first as a dependency, so a non-admin gets the plain
403 and is never asked to reverify. The local edition has no step-up (``core/auth/step_up.py``).
Every check and refusal is ``services/themes/revocation_admin.py``'s; a create writes only an
``origin='local'`` row (the body has no ``origin``: an unknown field is a 422). A lift of a ``feed``
row (only while revocation sync is off or stale) is logged at warning level with who lifted it:
this install then no longer follows the hosted list for that row.

The paths sit beside the review queue (``admin_themes.py``), whose ``/{package_id}/<action>``
routes have two segments and fixed tails, so neither router shadows the other. Plain ``def``
handlers over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.roles import caller_is_admin
from core.auth.step_up import require_step_up
from core.database.database import get_db
from core.models.themes import THEME_REVOCATION_ORIGIN_FEED
from services.themes import revocation_admin
from services.themes.choices import CONFLICT, ThemeInstallError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/themes", tags=["Admin Themes"])

OK, CREATED, FORBIDDEN = 200, 201, 403
ADMIN_REQUIRED = "Only an Automatos admin can revoke a theme or lift a revocation."
CONCURRENT_CHANGE = "Someone changed this revocation at the same moment. Reload and try again."

STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)

T = TypeVar("T")


class RevocationBody(BaseModel):
    """``POST /api/admin/themes/revocations``: a digest, or a theme id with an optional range."""

    model_config = ConfigDict(extra="forbid")
    digest: Optional[str] = None
    themeId: Optional[str] = None  # noqa: N815 — the wire name the revocation feed uses
    versionRange: Optional[str] = None  # noqa: N815 — the wire name the revocation feed uses
    level: str
    reason: str


def _require_admin(ctx: RequestContext) -> None:
    """403 unless the caller is a platform admin (``super_admin`` included)."""
    user = getattr(ctx, "user", None)
    if user is None or not caller_is_admin(user):
        raise HTTPException(status_code=FORBIDDEN, detail=ADMIN_REQUIRED)


def require_revocation_admin(ctx: RequestContext = Depends(get_request_context_hybrid)) -> None:
    """``_require_admin`` as a dependency, listed before ``STEP_UP``: a non-admin is never asked to reverify."""
    _require_admin(ctx)


# The admin check, then a fresh step-up: on the routes that revoke or lift. The order is the control
# (a non-admin never sees a reverification prompt); the test of the route table pins it.
ADMIN_STEP_UP = [Depends(require_revocation_admin), Depends(STEP_UP)]


def _run(db: Session, work: Callable[[], T]) -> T:
    """Run one write and commit it once; a refusal answers its own status, a lost race a 409."""
    try:
        result = work()
        db.commit()
        return result
    except ThemeInstallError as refusal:
        db.rollback()
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal
    except IntegrityError as clash:
        db.rollback()
        logger.warning("theme revocation write lost a race: %s", clash.orig.__class__.__name__)
        raise HTTPException(status_code=CONFLICT, detail=CONCURRENT_CHANGE) from clash


@router.get("/revocations")
def list_theme_revocations(
    limit: Optional[int] = Query(None, ge=1, le=revocation_admin.LIST_LIMIT_MAX),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Every revocation, live ones first, then the newest, each with its origin."""
    _require_admin(ctx)
    size = revocation_admin.LIST_LIMIT_DEFAULT if limit is None else limit
    return {"revocations": revocation_admin.list_revocations(db, size)}


@router.post("/revocations", status_code=CREATED, dependencies=ADMIN_STEP_UP)
def create_theme_revocation(
    body: RevocationBody, response: Response,
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Revoke a digest, or a theme id (optionally a version range of it): 201 with the new row,
    or 200 with the live row that already says the same."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    new = revocation_admin.NewRevocation(body.digest, body.themeId, body.versionRange, body.level, body.reason)
    row, created = _run(db, lambda: revocation_admin.create_revocation(db, new, actor))
    if created:
        logger.info("theme revocation %s (%s) created by user %s", row.id, row.level, actor)
    else:
        response.status_code = OK
    return {"revocation": revocation_admin.revocation_view(row), "created": created}


@router.post("/revocations/{revocation_id}/lift", dependencies=ADMIN_STEP_UP)
def lift_theme_revocation(
    revocation_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Lift a revocation written here; a row the hosted feed sent is the feed's to lift (409) while
    revocation sync works."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    row, lifted = _run(db, lambda: revocation_admin.lift_revocation(db, revocation_id))
    if lifted and row.origin == THEME_REVOCATION_ORIGIN_FEED:
        logger.warning("theme revocation %s from the feed lifted here by user %s (revocation sync off or stale)",
                       revocation_id, actor)
    elif lifted:
        logger.info("theme revocation %s lifted by user %s", revocation_id, actor)
    return {"revocation": revocation_admin.revocation_view(row)}
