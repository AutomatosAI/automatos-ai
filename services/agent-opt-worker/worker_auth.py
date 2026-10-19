"""The agent-opt worker answers only the platform (PRD-262 US-006).

Every route except /health and /metrics sits on one APIRouter whose dependency
requires ``X-Internal-Token``. It is a router-level dependency, not a middleware
that compares paths: under starlette 0.41.3 a forged Host header could rewrite
``request.url.path`` (GHSA-86qp-5c8j-p5mr), so a path exemption could be talked
past. It also never skips the check when the token is empty; the worker refuses
to start without one (worker_config.internal_token).
"""
from __future__ import annotations

import hmac
from typing import Awaitable, Callable

from fastapi import HTTPException, Request

TOKEN_HEADER = "X-Internal-Token"


def require_internal_token(expected: str) -> Callable[[Request], Awaitable[None]]:
    """A FastAPI dependency that answers 401 unless the request carries ``expected``."""
    if not expected:
        raise ValueError("the internal token must not be empty")
    expected_bytes = expected.encode("utf-8")

    async def check_internal_token(request: Request) -> None:
        supplied = request.headers.get(TOKEN_HEADER, "")
        # Constant time: a plain != stops at the first differing byte.
        if not hmac.compare_digest(supplied.encode("utf-8"), expected_bytes):
            raise HTTPException(status_code=401, detail="Unauthorized")

    return check_internal_token
