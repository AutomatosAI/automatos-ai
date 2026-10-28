"""PRD-268 §6.11.3, US-C18 — what Auto depends on, and what couldn't be checked (``/api/readiness``).

* ``GET /api/readiness`` — ``Readiness@1``: the workspace's last check while it is held (the
  check window), else a check now.
* ``POST /api/readiness/check`` — "Check again": a check now, held for the next ``GET``.
  Limited per workspace to ``readiness_check`` in ``DEFAULT_LIMITS``, counted in the shared
  Redis window by ``RateLimitStore`` (the ``api/mobile_devices.py`` pattern; Redis
  unreachable lets the check through and logs an ERROR naming this limiter). Over it is 429
  ``{"code": "checked_recently", "detail"}`` with ``Retry-After``, and nothing is checked.

Any member of the workspace, in both editions; a caller with no workspace is 404
``no_workspace``. The copy is in English or Portuguese (Brazil) by ``Accept-Language`` (K13).
The probes and the held check are ``services.readiness``. Plain ``def`` routes (each probe
runs on its own session, in its own thread), registered through ``RouterSpec("api.readiness")``
in ``router_manifest.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse

from api.widgets.rate_limit import RateLimitStore
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.security.rate_limiter import DEFAULT_LIMITS
from modules.proactive.locale import request_locale
from services.readiness import service
from services.readiness.strings import CHECKED_RECENTLY, NO_WORKSPACE, say

router = APIRouter(prefix="/api/readiness", tags=["readiness"])

# Every workspace role holds it.
MEMBER_PERMISSION = "members:read"
RATE_LIMITER_LABEL = "readiness check rate limiter"
HTTP_NOT_FOUND, HTTP_TOO_MANY_REQUESTS = 404, 429

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def _refused(locale: str, code: str, status_code: int, headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "detail": say(locale, code)},
                        headers=headers)


def _checked_recently(workspace_id: Any, locale: str) -> Optional[JSONResponse]:
    """429 once this workspace has spent its checks in the window, else None (and the check counted)."""
    limit, window = DEFAULT_LIMITS[service.CHECK_OPERATION]
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(
        f"{service.CHECK_OPERATION}:{workspace_id}", limit, window, label=RATE_LIMITER_LABEL,
    )
    if allowed:
        return None
    return _refused(locale, CHECKED_RECENTLY, HTTP_TOO_MANY_REQUESTS, {"Retry-After": str(reset_seconds)})


@router.get("", response_model=None)
def get_readiness(
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MEMBER_PERMISSION)),
) -> Union[Dict[str, Any], JSONResponse]:
    """``Readiness@1``: the last check while it is held, else a check now."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    return service.current(ctx.workspace_id, locale)


@router.post("/check", response_model=None)
def check_readiness(
    accept_language: Optional[str] = Header(None),
    ctx: RequestContext = Depends(require_workspace_permission(MEMBER_PERMISSION)),
) -> Union[Dict[str, Any], JSONResponse]:
    """Check again: every item now, limited per workspace."""
    locale = request_locale(accept_language)
    if ctx.workspace_id is None:
        return _refused(locale, NO_WORKSPACE, HTTP_NOT_FOUND)
    refused = _checked_recently(ctx.workspace_id, locale)
    if refused is not None:
        return refused
    return service.check_now(ctx.workspace_id, locale)
