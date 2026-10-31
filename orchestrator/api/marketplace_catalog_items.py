"""PRD-270 W2 US-270-12c (§5.6, K12) — the hosted catalogue's item bundles, public.

``GET /api/marketplace/catalog/bundles/{revision}`` answers the canonical bytes of the bundle the hosted
catalogue lists under ``revision`` (``services/marketplace_bundles/catalog_items.bundle_bytes_for``): the
path every catalogue entry names, from which a local install's Sync fetches a new or changed item. It
answers nothing else: a revision no listed bundle has, or one that is not 64 lower-case hex characters, is
404. It copies the rules of W1's catalogue files route (``api/marketplace_catalog.py``):

* 404 in the local edition (``require_hosted_bundles``), checked first, so a local request never counts
  against the limit;
* ``SERVING_HEADERS`` (nosniff and the sandboxed CSP) on every answer, a refusal included, and every
  refusal ``no-store``;
* ``Cache-Control: public, max-age=<THEME_FILE_CACHE_SECONDS>, immutable`` and the strong ETag
  ``"<revision>"`` with 304: a bundle's bytes are named by their own sha256, so they never change;
* its own per-IP bucket (``api/public_rate_limit.py``): over it is 429 with ``Retry-After``;
* 503, logged, when the database can't give the listing back.

Public by design: no credentials and no workspace, so nothing in it depends on the caller. A plain ``def``:
the session is synchronous. Registered through ``RouterSpec`` (``router_manifest.py``) and listed in
``reports/route-manifest.json``.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.marketplace_catalog import NOT_HOSTED
from api.public_rate_limit import REFUSAL_HEADERS, per_ip_limit
from api.themes_files import SERVING_HEADERS
from config import config
from core.database.database import get_db
from services.marketplace_bundles.catalog_items import bundle_bytes_for
from services.marketplace_bundles.catalog_paths import BUNDLES_ROUTE
from services.marketplace_bundles.schema import REVISION_PATTERN
from services.themes.revocation_feed import etag_matches
from services.themes.serve import file_cache_seconds

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Marketplace"])

BUNDLE_MEDIA_TYPE = "application/json"
NOT_MODIFIED, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 404, 503
BUNDLES_RATE_BUCKET = "marketplace-catalog-bundles:ip"
BUNDLES_RATE_LABEL = "marketplace catalogue bundles rate limiter"
BUNDLE_REFUSAL_HEADERS = {**SERVING_HEADERS, **REFUSAL_HEADERS}
BUNDLE_NOT_LISTED = "That bundle isn't in the Automatos Cloud catalogue."
BUNDLES_DOWN = "Automatos Cloud bundles can't be read right now. Try again in a moment."
_REVISION = re.compile(REVISION_PATTERN)


def _refusal(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail, headers={**BUNDLE_REFUSAL_HEADERS})


def require_hosted_bundles() -> None:
    """404 in the local edition, before anything else runs, with the serving headers: a local install
    fetches bundles, it never serves them."""
    if config.IS_LOCAL_EDITION:
        raise _refusal(NOT_FOUND, NOT_HOSTED)


limit_bundle_reads = per_ip_limit(BUNDLES_RATE_BUCKET, BUNDLES_RATE_LABEL, BUNDLE_REFUSAL_HEADERS)


def _listed_bytes(db: Session, revision: str) -> Optional[bytes]:
    """The listed bundle's bytes, None for a revision off the pattern or not listed; 503 (logged) when the
    database fails."""
    if _REVISION.fullmatch(revision) is None:
        return None
    try:
        return bundle_bytes_for(db, revision)
    except SQLAlchemyError as failed:
        logger.exception("marketplace catalogue bundles: the listing could not be read")
        raise _refusal(SERVICE_UNAVAILABLE, BUNDLES_DOWN) from failed


def _bundle_headers(revision: str) -> Dict[str, str]:
    return {
        "Cache-Control": f"public, max-age={file_cache_seconds()}, immutable", "ETag": f'"{revision}"',
        **SERVING_HEADERS,
    }


@router.get(BUNDLES_ROUTE + "/{revision}")
def get_catalog_bundle(
    revision: str,
    request: Request,
    _hosted: None = Depends(require_hosted_bundles),
    _limited: None = Depends(limit_bundle_reads),
    db: Session = Depends(get_db),
) -> Response:
    """A listed bundle's canonical bytes (200), or 304 when ``If-None-Match`` names its ETag; 404 in the
    local edition and for any revision the catalogue doesn't list; 503 when the database fails; 429 over
    the per-IP limit."""
    data = _listed_bytes(db, revision)
    if data is None:
        raise _refusal(NOT_FOUND, BUNDLE_NOT_LISTED)
    headers = _bundle_headers(revision)
    if etag_matches(request.headers.get("if-none-match"), f'"{revision}"'):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    return Response(content=data, media_type=BUNDLE_MEDIA_TYPE, headers=headers)


__all__ = ["BUNDLES_RATE_BUCKET", "limit_bundle_reads", "require_hosted_bundles", "router"]
