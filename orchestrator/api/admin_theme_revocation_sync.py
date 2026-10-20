"""PRD-261 C-05h — the local edition's revocation sync: its status, and Check now.

Routes (C3-wire registers the router, lists both in ``reports/route-manifest.json`` and adds the
pull to the authZ sweep's ``ADMIN_GATED_IN_HANDLER`` bucket):

=================================================  ==================================================
``GET  /api/admin/themes/revocations/sync``         ``{enabled, lastAttemptAt, lastSuccessAt,
                                                    lastError, feedGeneratedAt, rows, revokedKeys,
                                                    stale}``
``POST /api/admin/themes/revocations/sync/pull``    one pull now (``revocation_pull.pull_feed``):
                                                    200 with the status, 502 with the plain error
                                                    it recorded, 409 while the feed URL is empty
=================================================  ==================================================

``rows`` counts the revocations pulled from the feed and held here; ``stale`` is true when no
pull succeeded in the last 7 days (or ever). Both handlers check ``_require_admin(ctx)`` first
(``caller_is_admin``: a platform admin; the local edition's anonymous owner is a super admin), so
a non-admin gets the plain 403 in either edition; then the edition: the hosted edition serves the
feed and pulls nothing, so both answer 404 there. No step-up: the routes exist only in the local
edition, where step-up passes. Plain ``def`` handlers over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.roles import caller_is_admin
from core.database.database import get_db
from services.themes import revocation_pull
from services.themes.revocation_pull import ClientFactory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/themes/revocations/sync", tags=["Admin Themes"])

FORBIDDEN, NOT_FOUND, CONFLICT, BAD_GATEWAY = 403, 404, 409, 502
ADMIN_REQUIRED = "Only an Automatos admin can see or run the theme revocation sync."
HOSTED_NO_PULL = "The hosted edition serves the revocation feed; it has no sync to show or run."


def _require_admin(ctx: RequestContext) -> None:
    """403 unless the caller is a platform admin (``super_admin`` included)."""
    user = getattr(ctx, "user", None)
    if user is None or not caller_is_admin(user):
        raise HTTPException(status_code=FORBIDDEN, detail=ADMIN_REQUIRED)


def _require_local_edition() -> None:
    if not config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=HOSTED_NO_PULL)


def get_pull_client_factory() -> Optional[ClientFactory]:
    """The HTTP client a pull uses; None is the pull's own (a dependency, so tests answer from a mock)."""
    return None


@router.get("")
def get_revocation_sync(
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The sync status: the last attempt, success and error, and the feed's rows held here."""
    _require_admin(ctx)
    _require_local_edition()
    return revocation_pull.sync_status(db)


@router.post("/pull")
def pull_revocation_feed(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
    client_factory: Optional[ClientFactory] = Depends(get_pull_client_factory),
) -> Dict[str, Any]:
    """Check now: pull the feed once and answer the status, or the plain error the pull recorded."""
    _require_admin(ctx)
    _require_local_edition()
    if not revocation_pull.feed_url():
        raise HTTPException(status_code=CONFLICT, detail=revocation_pull.PULL_OFF)
    outcome = revocation_pull.pull_feed(db, client_factory)
    if not outcome.ok:
        raise HTTPException(status_code=BAD_GATEWAY, detail=outcome.error)
    logger.info("theme revocation feed pulled on demand by user %s", getattr(ctx.user, "id", None))
    return revocation_pull.sync_status(db)


__all__ = ["get_pull_client_factory", "router"]
