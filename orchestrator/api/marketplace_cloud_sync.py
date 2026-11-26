"""PRD-270 US-270-03b (§5.1, §5.5, §8-Q1, K17) — Sync with Automatos Cloud: its status, and the Sync.

Routes (registered through ``RouterSpec('api.marketplace_cloud_sync')`` in ``router_manifest.py`` and
listed in ``reports/route-manifest.json``):

===========================================  =====================================================
``GET  /api/marketplace/cloud-sync/status``   ``{enabled, cloudUrl, lastAttemptAt, lastSuccessAt,
                                              lastError, generatedAt, counts, themes, items, stale}``
``POST /api/marketplace/cloud-sync``          one Sync now (``pull.sync_catalog``), W2's plugin and
                                              item passes (``sync_items.run_item_passes``), then the
                                              auto-update (``updates.apply_cloud_updates``): 200
                                              ``{counts, items, plugins, updatesApplied,
                                              updatesFailed, status}``,
                                              502 with the plain error the Sync recorded, 409 while
                                              ``AUTOMATOS_CLOUD_URL`` is empty or another Sync is
                                              running
===========================================  =====================================================

Gates, in order (the handler's first two dependencies): ``calls_cloud_sync`` first, because the hosted
edition serves the catalogue and syncs nothing, and an enterprise install syncs only while its operator
switched it on (PRD-277 US-704), so both routes answer 404 otherwise to every caller, signed in or not
(with the hosted sentence on the hosted edition); then ``require_workspace_permission('workspace:manage')``, the
theme install's gate: a workspace owner or admin (the local operator is the owner, §8-Q1), anyone else 403. The
POST is the authZ sweep's class (a) by that gate.

A Sync brings packages and items in and moves installs for the whole install, so where the person using the
app is not the person who runs the server (``operator_is_user`` false: an enterprise install, where many people
sign in) the POST also needs an install admin (``caller_is_admin``, 403 ``SYNC_ADMIN_REQUIRED``) and then a
fresh step-up (``require_step_up``, as the admin theme routes have: ``admin_theme_revocations.py``). The admin
check runs first, so a workspace admin who is not an install admin is never asked to sign in again. The local
edition is as before: its operator is the person using it, so no install-admin check runs there, and its
step-up is ``skip`` (``core/auth/step_up.py``), so the step-up passes. Nothing schedules a Sync (§8-Q2): this
button is the only way one runs. Each Sync that ran, good or failed, writes one ``marketplace:cloud_sync``
row in ``audit_logs`` with who ran it and its counts (``services/admin_audit.py``, #1657); a Sync that couldn't
start (409) writes none. Plain ``def`` handlers over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config import config
from core import editions
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.roles import caller_is_admin
from core.auth.step_up import require_step_up
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services import admin_audit
from services.marketplace_bundles.sync_items import run_item_passes
from services.marketplace_cloud import pull, updates
from services.marketplace_cloud.fetch import CLOUD_OFF, ClientFactory
from services.marketplace_cloud.urls import configured_url

router = APIRouter(prefix="/api/marketplace/cloud-sync", tags=["Marketplace"])

MANAGE = require_workspace_permission("workspace:manage")
STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)
FORBIDDEN, NOT_FOUND, CONFLICT, BAD_GATEWAY = 403, 404, 409, 502
HOSTED_NO_SYNC = "The hosted edition is Automatos Cloud: it serves the catalogue and has no Sync to show or run."
SYNC_NOT_SWITCHED_ON = "This install doesn't sync with Automatos Cloud: its operator hasn't switched it on."
SYNC_ADMIN_REQUIRED = "Only an Automatos admin can sync this install with Automatos Cloud."


def require_local_edition() -> None:
    """404 before the caller is looked at, unless this install syncs with Automatos Cloud."""
    capabilities = editions.current()
    if not capabilities.calls_cloud_sync:
        detail = HOSTED_NO_SYNC if capabilities.hosts_automatos_cloud else SYNC_NOT_SWITCHED_ON
        raise HTTPException(status_code=NOT_FOUND, detail=detail)


def require_sync_admin(ctx: RequestContext = Depends(get_request_context_hybrid)) -> None:
    """403 unless the caller is an install admin, where the person using the app isn't its operator
    (``operator_is_user`` false); listed before ``STEP_UP``, so a non-admin is never asked to sign in again."""
    if editions.current().operator_is_user:
        return
    user = getattr(ctx, "user", None)
    if user is None or not caller_is_admin(user):
        raise HTTPException(status_code=FORBIDDEN, detail=SYNC_ADMIN_REQUIRED)


def get_cloud_client_factory() -> Optional[ClientFactory]:
    """The HTTP client a Sync uses; None is the Sync's own (a dependency, so tests answer from a mock)."""
    return None


def _refusal(outcome: pull.SyncOutcome) -> HTTPException:
    """409 when the Sync couldn't start (off, or another one running), else 502 with the error it recorded."""
    status_code = CONFLICT if outcome.off or outcome.busy else BAD_GATEWAY
    return HTTPException(status_code=status_code, detail=outcome.error)


def _record(db: Session, ctx: RequestContext, request: Request, details: Dict[str, Any]) -> None:
    """The Sync's ``marketplace:cloud_sync`` row (counts only); the Sync stands if it can't be written."""
    who = admin_audit.admin_request(db, ctx, request)
    admin_audit.record_after(db, who, action=admin_audit.CLOUD_SYNC, resource_type=admin_audit.CLOUD_SYNC_RESOURCE,
                             details=details)


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
    request: Request,
    _local: None = Depends(require_local_edition),
    ctx: RequestContext = Depends(MANAGE),
    _admin: None = Depends(require_sync_admin),
    _fresh: RequestContext = Depends(STEP_UP),
    db: Session = Depends(get_db),
    client_factory: Optional[ClientFactory] = Depends(get_cloud_client_factory),
) -> Dict[str, Any]:
    """Sync now: pull the catalogue once, bring its plugins and items, move the unpinned installs to the newest
    Cloud versions, and answer the counts (the whole Sync's), each item type's, the plugins waiting, the installs
    moved, the themes whose update failed and the status; or why the Sync didn't run."""
    if not configured_url():
        raise HTTPException(status_code=CONFLICT, detail=CLOUD_OFF)
    outcome = pull.sync_catalog(db, client_factory)
    if not (outcome.ok or outcome.off or outcome.busy):
        _record(db, ctx, request, {"outcome": admin_audit.OUTCOME_FAILED})
    if not outcome.ok:
        raise _refusal(outcome)
    passes = run_item_passes(db, outcome, client_factory, asyncio.run)
    moved = updates.apply_cloud_updates(db, client_factory)
    _record(db, ctx, request, {"outcome": admin_audit.OUTCOME_OK, "counts": passes["counts"],
                               "pluginsWaiting": passes["plugins"]["waiting"], "updatesApplied": moved.applied,
                               "updatesFailed": moved.failed})
    return {**passes, "updatesApplied": moved.applied, "updatesFailed": moved.failed, "status": pull.sync_status(db)}


__all__ = ["get_cloud_client_factory", "require_local_edition", "require_sync_admin", "router"]
