"""Unhandled exceptions become a JSON 500 *inside* the CORS layer (PRD-242 S1).

Starlette runs ``@app.exception_handler(Exception)`` in ``ServerErrorMiddleware``
— the outermost layer, outside ``CORSMiddleware``. A 500 minted there carries no
``Access-Control-Allow-Origin``, so the browser refuses to read it and every
unhandled backend error reaches the UI as ``TypeError: Failed to fetch`` — the
real status and message never surface (the 2026-09-11 Template Studio report:
a psycopg2 type error on ``/api/documents/variables`` showed up as a network
failure).

Mounted BEFORE ``CORSMiddleware`` in ``main.py`` (Starlette wraps later-added
middleware *outside* earlier ones), this converts the exception to a plain JSON
500 that then passes through CORS on the way out. ``HTTPException`` never
reaches here — ``ExceptionMiddleware`` (innermost) already turned it into a
response. The global handler in ``main.py`` stays as the last-resort backstop.
"""

from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

INTERNAL_ERROR_DETAIL = "Internal server error"

# A request id is client-supplied; keep only what a correlation id needs so a
# crafted header cannot forge log lines (CRLF) or bloat the response.
_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9._:-]")
MAX_REQUEST_ID_LEN = 128


def safe_request_id(raw: str | None) -> str:
    return _REQUEST_ID_RE.sub("", raw or "")[:MAX_REQUEST_ID_LEN]


async def json_500_for_unhandled_errors(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """``@app.middleware("http")`` dispatch: unhandled → logged JSON 500."""
    try:
        return await call_next(request)
    except Exception:  # noqa: BLE001 — this IS the catch-all; it logs and answers
        request_id = safe_request_id(request.headers.get("X-Request-ID"))
        logger.exception(
            "Unhandled error in %s %s (request_id=%s)", request.method, request.url.path, request_id
        )
        body = {"detail": INTERNAL_ERROR_DETAIL}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(status_code=500, content=body)


__all__ = ["INTERNAL_ERROR_DETAIL", "json_500_for_unhandled_errors", "safe_request_id"]
