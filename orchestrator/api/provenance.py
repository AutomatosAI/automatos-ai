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
  limit runs before the hash is checked, so malformed requests count too. An IPv6 caller
  is counted per /64 (one subscriber), and a client-IP value that is not an IP address
  shares one bucket, so it can't write an arbitrary key into Redis (``rate_bucket_for``).

Both editions serve it the same way. A bearer token or workspace header the browser
sends along is ignored.
"""
from __future__ import annotations

import ipaddress
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
# One IPv6 subscriber is usually handed a whole /64, so per-address buckets would let
# them rotate addresses inside it; the bucket is the /64 instead.
IPV6_BUCKET_PREFIX_LENGTH = 64
UNPARSABLE_IP_BUCKET = f"{RATE_BUCKET_PREFIX}:unparsable"
BAD_HASH_DETAIL = "Send the file's SHA-256 as 64 lower-case hex characters."
RATE_LIMITED_DETAIL = "Too many checks from this address. Try again in a minute."

router = APIRouter(prefix=PROVENANCE_ROUTE_PREFIX, tags=["Provenance"])

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def rate_bucket_for(request: Request) -> str:
    """The rate-limit bucket for this caller: one IPv4 address or one IPv6 /64.

    The address comes from ``get_client_ip`` (the first X-Forwarded-For hop, as main.py's
    global limiter reads it; on Railway the edge replaces a client-sent header,
    so the first hop is the connecting address). It is parsed as an IP so the Redis key
    and the fail-open log line only ever hold an address: anything else shares
    ``UNPARSABLE_IP_BUCKET``. An IPv4-mapped IPv6 address counts as its IPv4 address.
    """
    try:
        address = ipaddress.ip_address(get_client_ip(request))
    except ValueError:
        return UNPARSABLE_IP_BUCKET
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if address.version == 6:
        network = ipaddress.IPv6Network((int(address), IPV6_BUCKET_PREFIX_LENGTH), strict=False)
        return f"{RATE_BUCKET_PREFIX}:{network}"
    return f"{RATE_BUCKET_PREFIX}:{address}"


def limit_checks_per_ip(request: Request) -> None:
    """Refuse with 429 once this client IP has used its checks for the minute.

    A plain ``def``: the Redis round trip runs in FastAPI's threadpool, never on the
    event loop. Redis unreachable means the store lets the check through and logs it.
    """
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(
        rate_bucket_for(request), config.PROVENANCE_RATE_PER_MINUTE, PROVENANCE_WINDOW_SECONDS,
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
