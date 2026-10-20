"""PRD-261 C-05g — ``GET /api/themes/revocations``: the hosted revocation feed (plan §2.2.7, schema 1).

The hosted edition's public list of the revocations that cover Marketplace packages
(``services/themes/revocation_feed.py``), which every local edition pulls (C-05h). Public by
design: no credentials and no workspace, so nothing in it depends on the caller, and nothing in it
covers a workspace's private upload. ``Cache-Control: public, max-age=300`` with a weak ETag over
everything but ``generatedAt``, so a matching ``If-None-Match`` is a 304 until a revocation or a
revoked key changes. The local edition answers 404: it pulls the feed, it doesn't serve one. A
signing keys file that does not load is a 503: the feed's ``revokedKeys`` would be wrong, and a
local edition keeps what it has on any failed pull.

Mounted at ``/api/themes/revocations``, not under ``/api/marketplace/themes``, whose
``/{listing_id}`` route would catch it. A plain ``def``: the session is synchronous. C3-wire
registers the router in ``main.py`` and lists the route in ``reports/route-manifest.json``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from services.themes.revocation_feed import build_feed, etag_matches, feed_bytes, feed_etag, feed_max_bytes
from services.themes.signature_verify import SigningKeysError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/themes", tags=["Themes"])

FEED_PATH = "/revocations"
FEED_MAX_AGE_SECONDS = 300
FEED_HEADERS = {"Cache-Control": f"public, max-age={FEED_MAX_AGE_SECONDS}", "X-Content-Type-Options": "nosniff"}
FEED_MEDIA_TYPE = "application/json"
NOT_MODIFIED, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 404, 503
NOT_HOSTED = "The revocation feed is served by the hosted edition; this edition pulls it."
FEED_DOWN = "The revocation feed can't be built right now. Try again in a moment."


@router.get(FEED_PATH)
def get_revocation_feed(request: Request, db: Session = Depends(get_db)) -> Response:
    """The feed (200), or 304 when ``If-None-Match`` names its ETag; 404 in the local edition."""
    if config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=NOT_HOSTED)
    try:
        feed = build_feed(db)
    except SigningKeysError as broken:
        logger.error("theme revocation feed: the signing keys file did not load: %s", broken)
        raise HTTPException(status_code=SERVICE_UNAVAILABLE, detail=FEED_DOWN) from broken
    etag = feed_etag(feed)
    headers = {**FEED_HEADERS, "ETag": etag}
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    body = feed_bytes(feed)
    if len(body) > feed_max_bytes():
        logger.error("theme revocation feed is %d bytes, more than a local edition reads (%d)", len(body),
                     feed_max_bytes())
    return Response(content=body, media_type=FEED_MEDIA_TYPE, headers=headers)


__all__ = ["router"]
