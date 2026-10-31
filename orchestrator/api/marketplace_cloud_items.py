"""PRD-270 W2 US-270-14a (§5.6, K17, K18) — the local reads the web shows Automatos Cloud items with.

Routes (registered through ``RouterSpec('api.marketplace_cloud_items')`` in ``router_manifest.py`` and listed in
``reports/route-manifest.json``; beside W1's ``/status`` and ``POST`` under the same prefix, no path shared):

==========================================================  ==================================================
``GET /api/marketplace/cloud-sync/items``                   the Cloud rows of each type, by the key its install
                                                            route takes (``local_reads.cloud_items``)
``GET /api/marketplace/cloud-sync/items/{kind}/{key}/       everything one Cloud item will use
disclosure``                                                (``disclosure.disclosure``); 404 for an unknown kind
                                                            and for a row no mapping points at
``GET /api/marketplace/cloud-sync/updates``                 the caller's workspace's copies with a newer Cloud
                                                            version and its Cloud plugins that moved
                                                            (``updates.updates``)
==========================================================  ==================================================

Gates, in order (each handler's first two dependencies): the edition first, because the hosted edition is
Automatos Cloud and holds no item brought from it, so every route answers 404 there to every caller, signed in
or not; then ``get_request_context_hybrid``: any member of the workspace, since these read catalogue content any
member may install and only the caller's own workspace's copies (``/updates`` filters by ``ctx.workspace_id``).
All three are GETs over local rows only, with no network call. Plain ``def`` handlers over the synchronous
session (AGENTS.md).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.marketplace_bundles.disclosure import NotACloudItem, disclosure
from services.marketplace_bundles.local_reads import cloud_items
from services.marketplace_bundles.updates import updates

router = APIRouter(prefix="/api/marketplace/cloud-sync", tags=["Marketplace"])

NOT_FOUND = 404
HOSTED_NO_CLOUD_ITEMS = "The hosted edition is Automatos Cloud: it holds no items brought from Automatos Cloud to show."
NOT_A_CLOUD_ITEM = "That isn't an item this install brought from Automatos Cloud."


def require_local_edition() -> None:
    """404 in the hosted edition, before the caller is looked at: only a local install holds Cloud items."""
    if not config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=HOSTED_NO_CLOUD_ITEMS)


@router.get("/items")
def get_cloud_items(
    _local: None = Depends(require_local_edition),
    _ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The Cloud agents, Playbooks, packages and plugins this install holds, by the keys their installs take."""
    return cloud_items(db)


@router.get("/items/{kind}/{key}/disclosure")
def get_cloud_item_disclosure(
    kind: str,
    key: str,
    _local: None = Depends(require_local_edition),
    _ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Everything the Cloud item will use (its prompt, model, apps, skills, plugins, steps, members or reports),
    from this install's rows; 404 unless it is a Cloud item of that kind."""
    try:
        return disclosure(db, kind, key)
    except NotACloudItem:
        raise HTTPException(status_code=NOT_FOUND, detail=NOT_A_CLOUD_ITEM) from None


@router.get("/updates")
def get_cloud_updates(
    _local: None = Depends(require_local_edition),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """This workspace's copies of Cloud agents and Playbooks with a newer version, and the Cloud plugins it
    enabled that have moved since."""
    return updates(db, ctx.workspace_id)


__all__ = ["require_local_edition", "router"]
