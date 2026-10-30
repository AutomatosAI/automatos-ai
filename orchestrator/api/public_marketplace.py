"""PRD-270 US-270-07a (§5.7, K10) — the public Marketplace reads, ``GET /api/public/marketplace/{type}``.

What a visitor with no account sees of each Marketplace tab but Themes (the catalogue is their public
read): one page of the type's approved catalogue rows, display fields only
(``services/marketplace_public/``). Public by design, on PRD-264's precedent (``api/provenance.py``): no
credentials and no workspace, so nothing in an answer depends on the caller, and nothing in it is a
workspace's. Its rules are the catalogue's (``api/marketplace_catalog.py``):

* the local edition answers 404: §5.7 is the hosted edition's. The edition is checked first, so a local
  request never counts against the limit;
* each client IP gets ``MARKETPLACE_PUBLIC_RATE_PER_MINUTE`` reads a minute in its own bucket,
  ``marketplace-public:ip`` (``api/public_rate_limit.py``: the widget API's ``RateLimitStore`` window
  over ``core.security.ip_bucket.ip_rate_bucket``): over it is 429 with ``Retry-After``;
* a type with no reader is 404; ``?limit`` is 1 to 100 (50 by default) and ``?offset`` 0 or more, else
  422;
* the answer is ``{type, items, limit, offset, more}``, with ``Cache-Control: public, max-age=300``,
  nosniff and a weak ETag over the answer (``revocation_feed.feed_etag``), so a matching
  ``If-None-Match`` is a 304 until the page changes;
* a database that fails the read is a logged 503. Every refusal (404, 429, 503) is ``no-store``.

A plain ``def``: the session is synchronous. Registered through ``RouterSpec`` (``router_manifest.py``)
and listed in ``reports/route-manifest.json``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.public_rate_limit import REFUSAL_HEADERS, per_ip_limit
from config import config
from core.database.database import get_db
from services.marketplace_public.reads import public_page
from services.themes.revocation_feed import etag_matches, feed_etag

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Marketplace"])

PUBLIC_READS_ROUTE = "/api/public/marketplace/{type}"
DEFAULT_LIMIT, MAX_LIMIT, MAX_OFFSET = 50, 100, 100_000
PUBLIC_MAX_AGE_SECONDS = 300
PUBLIC_HEADERS = {"Cache-Control": f"public, max-age={PUBLIC_MAX_AGE_SECONDS}", "X-Content-Type-Options": "nosniff"}
PUBLIC_MEDIA_TYPE = "application/json"
NOT_MODIFIED, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 404, 503
PUBLIC_RATE_BUCKET = "marketplace-public:ip"
PUBLIC_RATE_LABEL = "public marketplace reads rate limiter"
NOT_HOSTED = "The public Marketplace is served by the hosted edition."
NO_SUCH_TYPE = "The Marketplace has no such tab."
READS_DOWN = "The Marketplace can't be read right now. Try again in a moment."


def _refusal(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail, headers={**REFUSAL_HEADERS})


def require_hosted_edition() -> None:
    """404 in the local edition, before anything else runs: the public Marketplace is the hosted edition's."""
    if config.IS_LOCAL_EDITION:
        raise _refusal(NOT_FOUND, NOT_HOSTED)


limit_public_reads = per_ip_limit(PUBLIC_RATE_BUCKET, PUBLIC_RATE_LABEL)


def _page(db: Session, kind: str, limit: int, offset: int) -> Dict[str, Any]:
    """The page, or 404 for a type with no reader, or a logged 503 when the database fails the read."""
    try:
        page = public_page(db, kind, limit, offset)
    except SQLAlchemyError as failed:
        logger.exception("public marketplace: the %s read failed", kind)
        raise _refusal(SERVICE_UNAVAILABLE, READS_DOWN) from failed
    if page is None:
        raise _refusal(NOT_FOUND, NO_SUCH_TYPE)
    return page


@router.get(PUBLIC_READS_ROUTE)
def get_public_marketplace(
    type: str,
    request: Request,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0, le=MAX_OFFSET),
    _hosted: None = Depends(require_hosted_edition),
    _limited: None = Depends(limit_public_reads),
    db: Session = Depends(get_db),
) -> Response:
    """One page of the type's public items (200), or 304 when ``If-None-Match`` names its ETag; 404 in the
    local edition and for an unknown type; 422 for a limit or offset out of range; 429 over the per-IP
    limit; 503 when the database fails the read."""
    page = _page(db, type, limit, offset)
    etag = feed_etag(page)
    headers = {**PUBLIC_HEADERS, "ETag": etag}
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    body = json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return Response(content=body, media_type=PUBLIC_MEDIA_TYPE, headers=headers)


__all__ = ["PUBLIC_HEADERS", "PUBLIC_READS_ROUTE", "router"]
