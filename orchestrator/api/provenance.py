"""The public detection route: was this file made by Automatos AI? (PRD-264 EUAI-10, D16).

``GET /api/public/provenance/{sha256}`` is the one endpoint PRD-264 leaves without
authentication (decision D16). Article 50(2) and Guidelines para. 75 ask that the
people who see generated content can detect it, and they have no Automatos account.
So the route is held small:

* It takes a SHA-256, never a file: the /verify page hashes the file in the visitor's
  browser. Anything but 64 lower-case hex characters is refused with 422.
* It answers only ``{"made_by_automatos_ai": true, "kind": ..., "date": "YYYY-MM-DD"}``
  or ``{"made_by_automatos_ai": false}`` (``services.content_provenance``), never the
  workspace, the person or the file name.
* Each client IP gets ``config.PROVENANCE_RATE_PER_MINUTE`` checks a minute, counted in
  the widget API's shared Redis sliding window; over it is 429 with ``Retry-After``. The
  limit runs before the hash is checked, so malformed requests count too.

Both editions serve it the same way. A bearer token or workspace header the browser
sends along is ignored.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.credentials import get_client_ip
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.database.database import get_db
from services.content_provenance import is_sha256_hex, provenance_answer

PROVENANCE_ROUTE_PREFIX = "/api/public/provenance"
PROVENANCE_WINDOW_SECONDS = 60
RATE_BUCKET_PREFIX = "provenance:ip"
BAD_HASH_DETAIL = "Send the file's SHA-256 as 64 lower-case hex characters."
RATE_LIMITED_DETAIL = "Too many checks from this address. Try again in a minute."

router = APIRouter(prefix=PROVENANCE_ROUTE_PREFIX, tags=["Provenance"])

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def limit_checks_per_ip(request: Request) -> None:
    """Refuse with 429 once this client IP has used its checks for the minute.

    A plain ``def``: the Redis round trip runs in FastAPI's threadpool, never on the
    event loop. Redis unreachable means the store lets the check through and logs it.
    """
    bucket = f"{RATE_BUCKET_PREFIX}:{get_client_ip(request)}"
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(
        bucket, config.PROVENANCE_RATE_PER_MINUTE, PROVENANCE_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=RATE_LIMITED_DETAIL,
            headers={"Retry-After": str(reset_seconds)},
        )


@router.get("/{sha256}")
def check_provenance(
    sha256: str,
    _rate_limited: None = Depends(limit_checks_per_ip),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Whether a file with this SHA-256 is one Automatos AI made and marked.

    A plain ``def`` (AGENTS.md): one indexed read on a synchronous session, in the
    threadpool. A re-encoded or edited file has a different hash and answers no.
    """
    if not is_sha256_hex(sha256):
        raise HTTPException(status_code=422, detail=BAD_HASH_DETAIL)
    return provenance_answer(db, sha256)
