"""PRD-270 US-270-01 (§5.2, K8) — ``GET /api/marketplace/catalog``: the hosted catalogue, schema 1.

The hosted edition's public list of the signed theme versions it lists
(``services/marketplace_cloud/catalog.py``), which every local install's Sync with Automatos Cloud
reads and the public ``/marketplace`` shows. Public by design: no credentials and no workspace, so
nothing in it depends on the caller, and nothing in it is a workspace's. It copies the revocation
feed's rules (``api/themes_revocations_feed.py``):

* the local edition answers 404: it syncs the catalogue, it doesn't serve one. The edition is checked
  first, so a local request never counts against the limit;
* ``Cache-Control: public, max-age=300`` and ``nosniff``, with a weak ETag over everything but
  ``generatedAt`` (``revocation_feed.feed_etag``), so a matching ``If-None-Match`` is a 304 until a
  listed version changes;
* 503, logged as an error, when the signing keys file doesn't load, when storage can't give back a
  listed version's theme.json, or when the catalogue is over ``MARKETPLACE_CATALOG_FEED_MAX_KB``:
  served, it would fail every Sync. The size is checked before the ETag, so such a catalogue is never
  confirmed with a 304 either. A refusal (404, 429, 503) carries ``Cache-Control: no-store``;
* each client IP gets ``MARKETPLACE_PUBLIC_RATE_PER_MINUTE`` reads a minute in its own bucket
  (``api/public_rate_limit.py``): over it is 429 with ``Retry-After``;
* the catalogue is built at most once per ``catalog.CATALOG_KEPT_SECONDS`` for every caller
  (``catalog.catalog_snapshot``), so a public request never rebuilds it on its own.

A plain ``def``: the session is synchronous. Registered through ``RouterSpec`` (``router_manifest.py``)
and listed in ``reports/route-manifest.json``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.public_rate_limit import REFUSAL_HEADERS, per_ip_limit
from config import config
from core.database.database import get_db
from services.marketplace_cloud.catalog import CatalogSnapshot, CatalogUnavailable, catalog_max_bytes, catalog_snapshot
from services.marketplace_cloud.urls import CATALOG_ROUTE
from services.themes.revocation_feed import etag_matches
from services.themes.signature_verify import SigningKeysError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Marketplace"])

CATALOG_MAX_AGE_SECONDS = 300
CATALOG_HEADERS = {"Cache-Control": f"public, max-age={CATALOG_MAX_AGE_SECONDS}", "X-Content-Type-Options": "nosniff"}
CATALOG_MEDIA_TYPE = "application/json"
NOT_MODIFIED, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 404, 503
CATALOG_RATE_BUCKET = "marketplace-catalog:ip"
CATALOG_RATE_LABEL = "marketplace catalogue rate limiter"
NOT_HOSTED = "The Automatos Cloud catalogue is served by the hosted edition; this edition syncs it."
CATALOG_DOWN = "The Automatos Cloud catalogue can't be built right now. Try again in a moment."
CATALOG_TOO_LARGE = "The Automatos Cloud catalogue is too large to serve right now. Try again later."


def require_hosted_edition() -> None:
    """404 in the local edition, before anything else runs: a local install syncs the catalogue, never serves it."""
    if config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=NOT_HOSTED, headers={**REFUSAL_HEADERS})


limit_catalog_reads = per_ip_limit(CATALOG_RATE_BUCKET, CATALOG_RATE_LABEL)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=SERVICE_UNAVAILABLE, detail=detail, headers={**REFUSAL_HEADERS})


def _snapshot(db: Session) -> CatalogSnapshot:
    """The kept or freshly built catalogue, or the route's 503 (logged) when the keys file or a theme.json can't
    be read."""
    try:
        return catalog_snapshot(db)
    except SigningKeysError as broken:
        logger.error("marketplace catalogue: the signing keys file did not load: %s", broken)
        raise _unavailable(CATALOG_DOWN) from broken
    except CatalogUnavailable as unreadable:
        logger.error("marketplace catalogue not served: %s", unreadable)
        raise _unavailable(CATALOG_DOWN) from unreadable


@router.get(CATALOG_ROUTE)
def get_catalog(
    request: Request,
    _hosted: None = Depends(require_hosted_edition),
    _limited: None = Depends(limit_catalog_reads),
    db: Session = Depends(get_db),
) -> Response:
    """The catalogue (200), or 304 when ``If-None-Match`` names its ETag; 404 in the local edition; 503 when
    the keys file or a theme.json can't be read or the catalogue is over its size; 429 over the per-IP limit."""
    snapshot, cap = _snapshot(db), catalog_max_bytes()
    if len(snapshot.body) > cap:
        logger.error("marketplace catalogue not served: %d bytes, %d items, over the %d-byte cap",
                     len(snapshot.body), len(snapshot.catalog["items"]), cap)
        raise _unavailable(CATALOG_TOO_LARGE)
    headers = {**CATALOG_HEADERS, "ETag": snapshot.etag}
    if etag_matches(request.headers.get("if-none-match"), snapshot.etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    return Response(content=snapshot.body, media_type=CATALOG_MEDIA_TYPE, headers=headers)


__all__ = ["CATALOG_HEADERS", "router"]
