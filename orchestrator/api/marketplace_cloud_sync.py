"""PRD-270 US-270-03b (§5.1, §5.5, §8-Q1, K17) — Sync with Automatos Cloud: its status, and the Sync.

Routes (registered through ``RouterSpec('api.marketplace_cloud_sync')`` in ``router_manifest.py`` and
listed in ``reports/route-manifest.json``):

===========================================  =====================================================
``GET  /api/marketplace/cloud-sync/status``   ``{enabled, cloudUrl, lastAttemptAt, lastSuccessAt,
                                              lastError, generatedAt, counts, themes, stale}``
``POST /api/marketplace/cloud-sync``          one Sync now (``pull.sync_catalog``): 200
                                              ``{counts, status}``, 502 with the plain error it
                                              recorded, 409 while ``AUTOMATOS_CLOUD_URL`` is empty
                                              or another Sync is running
===========================================  =====================================================

Gates, in order (the handler's first two dependencies): the edition first, because the hosted edition
serves the catalogue and syncs nothing, so both routes answer 404 there to every caller, signed in or
not; then ``require_workspace_permission('workspace:manage')``, the theme install's gate: a workspace
owner or admin (the local operator is the owner, §8-Q1), anyone else 403. The POST is the authZ sweep's
class (a) by that gate. No step-up: the routes exist only in the local edition, where step-up passes.
Nothing schedules a Sync (§8-Q2): this button is the only way one runs. Plain ``def`` handlers over the
synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.marketplace_cloud import pull
from services.marketplace_cloud.fetch import CLOUD_OFF, ClientFactory
from services.marketplace_cloud.urls import configured_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace/cloud-sync", tags=["Marketplace"])

MANAGE = require_workspace_permission("workspace:manage")
NOT_FOUND, CONFLICT, BAD_GATEWAY = 404, 409, 502
HOSTED_NO_SYNC = "The hosted edition is Automatos Cloud: it serves the catalogue and has no Sync to show or run."


def require_local_edition() -> None:
    """404 in the hosted edition, before the caller is looked at: only a local install syncs."""
    if not config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=HOSTED_NO_SYNC)


def get_cloud_client_factory() -> Optional[ClientFactory]:
    """The HTTP client a Sync uses; None is the Sync's own (a dependency, so tests answer from a mock)."""
    return None


def _refusal(outcome: pull.SyncOutcome) -> HTTPException:
    """409 when the Sync couldn't start (off, or another one running), else 502 with the error it recorded."""
    status_code = CONFLICT if outcome.off or outcome.busy else BAD_GATEWAY
    return HTTPException(status_code=status_code, detail=outcome.error)


@router.get("/status")
def get_cloud_sync_status(
    _local: None = Depends(require_local_edition),
    _ctx: RequestContext = Depends(MANAGE),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The Sync's status: on or off, the origin in use, the last attempt, success and error, the counts,
    the themes held from Automatos Cloud and whether the last success is over a week old."""
    return pull.sync_status(db)


@router.post("")
def run_cloud_sync(
    _local: None = Depends(require_local_edition),
    ctx: RequestContext = Depends(MANAGE),
    db: Session = Depends(get_db),
    client_factory: Optional[ClientFactory] = Depends(get_cloud_client_factory),
) -> Dict[str, Any]:
    """Sync now: pull the catalogue once and answer its counts and the status, or why it didn't run."""
    if not configured_url():
        raise HTTPException(status_code=CONFLICT, detail=CLOUD_OFF)
    outcome = pull.sync_catalog(db, client_factory)
    if not outcome.ok:
        raise _refusal(outcome)
    logger.info("Sync with Automatos Cloud run on demand by user %s", getattr(ctx.user, "id", None))
    return {"counts": outcome.counts.view(), "status": pull.sync_status(db)}


__all__ = ["get_cloud_client_factory", "require_local_edition", "router"]
