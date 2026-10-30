"""PRD-270 K7 — the per-IP limit of the public Marketplace routes (the catalogue, its files, the public reads).

A public route takes no account, so it counts its callers by address, as the public detection route
does (``api/provenance.py``): ``config.MARKETPLACE_PUBLIC_RATE_PER_MINUTE`` requests a minute per
client IP, in the widget API's shared Redis sliding window (``RateLimitStore``), bucketed by
``core.security.ip_bucket.ip_rate_bucket`` (one IPv4 address or one IPv6 /64; a client-IP value that
isn't an address shares one bucket). Each route passes its own bucket prefix, so each has its own
budget. Over it is 429 with ``Retry-After`` and ``Cache-Control: no-store`` (a cache must never keep a
refusal for everyone). Redis unreachable lets the request through and logs it (the store's rule).
"""

from __future__ import annotations

from typing import Callable

from fastapi import HTTPException, Request

from api.credentials import get_client_ip
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.security.ip_bucket import ip_rate_bucket

PUBLIC_WINDOW_SECONDS = 60
TOO_MANY_REQUESTS = 429
RATE_LIMITED_DETAIL = "Too many requests from this address. Try again in a minute."
REFUSAL_HEADERS = {"Cache-Control": "no-store"}

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def per_ip_limit(bucket_prefix: str, label: str) -> Callable[[Request], None]:
    """A dependency that refuses with 429 once this client IP has used the route's requests for the minute.

    ``label`` names the limiter in the store's fail-open ERROR line. The dependency is a plain ``def``:
    the Redis round trip runs in FastAPI's threadpool, never on the event loop.
    """

    def limit(request: Request) -> None:
        allowed, _limit, _remaining, reset_seconds = _rate_store.check(
            ip_rate_bucket(get_client_ip(request), bucket_prefix), config.MARKETPLACE_PUBLIC_RATE_PER_MINUTE,
            PUBLIC_WINDOW_SECONDS, label=label,
        )
        if not allowed:
            raise HTTPException(
                status_code=TOO_MANY_REQUESTS, detail=RATE_LIMITED_DETAIL,
                headers={**REFUSAL_HEADERS, "Retry-After": str(reset_seconds)},
            )

    return limit


__all__ = ["PUBLIC_WINDOW_SECONDS", "RATE_LIMITED_DETAIL", "REFUSAL_HEADERS", "per_ip_limit"]
