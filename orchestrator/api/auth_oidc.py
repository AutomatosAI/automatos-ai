"""PRD-277 US-402 (§5.2): signing in with the organisation's identity provider (``/api/v1/auth/oidc``).

Unless the edition signs in with OIDC (``sign_in`` is ``oidc``: enterprise), both routes answer
404 before anything else, so ``local`` and ``saas`` serve exactly what they served before.

Both are public (nobody is signed in yet), plain ``def`` (the sync session and the calls to the
provider run in the threadpool), and limited per client IP before any database or provider work
(``oidc_start_ip``, ``oidc_callback_ip``: 429 with ``Retry-After``, the ``/mobile/pair`` pattern).
The address is ``CLIENT_IP_HEADER``'s, which start-up requires where sign-in is OIDC. When the
shared limiter's store cannot be reached, these two routes count in this process instead
(``core.security.local_window``), with the same limits, so they stay bounded.

* ``GET /start?return_to=<path>`` starts a sign-in (``core.auth.sign_in_flow.start_sign_in``) and
  answers 303 to the provider's sign-in page. ``return_to`` is kept only when it is a path on this
  install (``sign_in_methods.safe_return_path``), else ``/``. The sign-in's browser binding, a
  secret of its own that is never in an address, goes in a ``__Host-`` cookie (``SIGN_IN_COOKIE``:
  Secure, HttpOnly, SameSite=Lax, ten minutes) that binds the callback to this browser. The same
  client address bounds the sign-ins one caller may hold unfinished (``core.auth.oidc_pending``).
* ``GET /start?step_up=1&return_to=<path>`` (US-406) starts a step-up for the live session this
  browser's session cookie holds (``sign_in_flow.start_step_up``), sending the provider
  ``max_age=0``; with no live session it is refused (``step_up_refused``) and nothing is stored, as
  it is when the browser's ``Sec-Fetch-Site`` says another site sent it (``_step_up_sent_from_here``).
* ``GET /callback`` finishes it (``sign_in_flow.finish_sign_in``): 303 to the stored path, with the
  session cookie set (``session_cookie.set_session_cookie``). After a step-up no session is issued
  and the cookie is left as it is: the browser's own session now carries a fresh ``auth_time``. A
  callback with more than ``CALLBACK_PARAMS_MAX`` parameters is read as carrying none, so it is
  refused.

Every answer carries ``Cache-Control: no-store`` and ``Referrer-Policy: no-referrer`` (the
callback's address holds the one-time code, which must never be sent on as a referrer), and a refusal is 303 to ``/sign-in?error=<code>``, the code one of ``sign_in_flow.REFUSAL_CODES``,
never the provider's own error text. The callback clears the binding cookie whatever happens.
Registered through ``RouterSpec("api.auth_oidc")`` in ``router_manifest.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.credentials import get_client_ip
from api.widgets.rate_limit import RateLimitStore
from core import editions
from core.auth import session_cookie, sign_in_flow
from core.auth.bearer_registry import session_cookie_token
from core.auth.oidc_pending import SIGN_IN_TTL_SECONDS
from core.auth.sign_in_methods import DEFAULT_RETURN_PATH, Callback
from core.database.database import get_db
from core.security.ip_bucket import ip_rate_bucket
from core.security.local_window import LocalWindow
from core.security.rate_limiter import DEFAULT_LIMITS

NOT_FOUND = "Not Found"
SIGN_IN_COOKIE = "__Host-automatos_sign_in"
SIGN_IN_PAGE = "/sign-in"
NO_STORE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
RATE_LIMITED_DETAIL = "Too many sign-in attempts from this address. Try again later."
# A callback carries a handful of parameters; more than this is not one.
CALLBACK_PARAMS_MAX = 32
# Fetch Metadata: where the browser says a request came from. A step-up starts only from these.
FETCH_SITE_HEADER = "sec-fetch-site"
STEP_UP_FETCH_SITES = ("same-origin", "none")

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()
# Where these two routes count while the shared store cannot be reached (never fail open).
_local_window = LocalWindow()


@dataclass(frozen=True)
class _IpLimit:
    operation: str
    bucket_prefix: str
    label: str


_START_LIMIT = _IpLimit("oidc_start_ip", "oidc_start:ip", "sign-in start rate limiter")
_CALLBACK_LIMIT = _IpLimit("oidc_callback_ip", "oidc_callback:ip", "sign-in callback rate limiter")


def _signs_in_with_oidc() -> None:
    """404 where the edition does not sign in with OIDC, before anything else."""
    if editions.current().sign_in != editions.SIGN_IN_OIDC:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)


router = APIRouter(prefix="/api/v1/auth/oidc", tags=["auth"], dependencies=[Depends(_signs_in_with_oidc)])


def _check_ip_limit(request: Request, ip_limit: _IpLimit) -> None:
    """429 with ``Retry-After`` once this client IP has spent ``ip_limit`` in its window."""
    limit, window = DEFAULT_LIMITS[ip_limit.operation]
    bucket = ip_rate_bucket(get_client_ip(request), ip_limit.bucket_prefix)
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(
        bucket, limit, window, label=ip_limit.label,
        when_unavailable=lambda: _local_window.check(bucket, limit, window),
    )
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=RATE_LIMITED_DETAIL,
                            headers={"Retry-After": str(reset_seconds)})


def limit_starts_per_ip(request: Request) -> None:
    """429 once this client IP has started too many sign-ins. Runs before anything is stored."""
    _check_ip_limit(request, _START_LIMIT)


def limit_callbacks_per_ip(request: Request) -> None:
    """429 once this client IP has sent too many callbacks. Runs before the state is read."""
    _check_ip_limit(request, _CALLBACK_LIMIT)


def _redirect(to: str) -> RedirectResponse:
    return RedirectResponse(to, status_code=status.HTTP_303_SEE_OTHER, headers=dict(NO_STORE_HEADERS))


def _refused(code: str) -> RedirectResponse:
    return _redirect(f"{SIGN_IN_PAGE}?{urlencode({'error': code})}")


def _clear_binding(response: RedirectResponse) -> None:
    response.delete_cookie(key=SIGN_IN_COOKIE, path=session_cookie.COOKIE_PATH, secure=True, httponly=True,
                           samesite=session_cookie.COOKIE_SAME_SITE)


def _browser(request: Request) -> sign_in_flow.Browser:
    return sign_in_flow.Browser(ip_address=get_client_ip(request), user_agent=request.headers.get("user-agent"))


def _params(request: Request) -> Dict[str, Tuple[str, ...]]:
    """The callback's query, each name with every value it was sent (a repeated name is not one value).

    More than ``CALLBACK_PARAMS_MAX`` parameters is not a callback: it is read as carrying none.
    """
    items = request.query_params.multi_items()
    if len(items) > CALLBACK_PARAMS_MAX:
        return {}
    grouped: Dict[str, Tuple[str, ...]] = {}
    for name, value in items:
        grouped = {**grouped, name: (*grouped.get(name, ()), value)}
    return grouped


def _step_up_sent_from_here(request: Request) -> bool:
    """Whether a step-up start came from this install's own pages or the address bar.

    ``Sec-Fetch-Site`` (Fetch Metadata) is set by the browser and cannot be set by a page. Only
    ``same-origin`` (the web app shares the API's origin, §5.3) and ``none`` (typed or bookmarked)
    may start one; ``same-site`` and ``cross-site`` may not. A request without the header (a
    browser too old to send it) is let through: the start only sends the browser to the provider,
    the callback is bound to this browser by its cookie, and a step-up confirms nobody but the
    session's own person, so another site gains nothing it could not already cause.
    """
    sent_from = request.headers.get(FETCH_SITE_HEADER)
    return sent_from is None or sent_from.strip().lower() in STEP_UP_FETCH_SITES


def _started(request: Request, db: Session, return_to: str, step_up: bool):
    if step_up:
        if not _step_up_sent_from_here(request):
            raise sign_in_flow.SignInRefusal(sign_in_flow.STEP_UP_REFUSED)
        return sign_in_flow.start_step_up(db, session_cookie_token(request), return_to, get_client_ip(request))
    return sign_in_flow.start_sign_in(db, return_to, get_client_ip(request))


@router.get("/start", dependencies=[Depends(limit_starts_per_ip)])
def start(request: Request, return_to: str = Query(DEFAULT_RETURN_PATH), step_up: bool = Query(False),
          db: Session = Depends(get_db)) -> RedirectResponse:
    """303 to the provider's sign-in page, with this browser's sign-in bound to it by cookie.

    ``step_up=1`` (US-406) signs the person in again for the session this browser's cookie holds.
    """
    try:
        started = _started(request, db, return_to, step_up)
    except sign_in_flow.SignInRefusal as refusal:
        return _refused(refusal.code)
    response = _redirect(started.redirect_to)
    response.set_cookie(key=SIGN_IN_COOKIE, value=started.browser_binding, max_age=SIGN_IN_TTL_SECONDS,
                        path=session_cookie.COOKIE_PATH, secure=True, httponly=True,
                        samesite=session_cookie.COOKIE_SAME_SITE)
    return response


@router.get("/callback", dependencies=[Depends(limit_callbacks_per_ip)])
def callback(request: Request, db: Session = Depends(get_db)) -> RedirectResponse:
    """303 to where the person was going, signed in; or to the sign-in page with the refusal's code."""
    sent = Callback(params=_params(request), browser_binding=request.cookies.get(SIGN_IN_COOKIE))
    try:
        done = sign_in_flow.finish_sign_in(db, sent, _browser(request))
    except sign_in_flow.SignInRefusal as refusal:
        response = _refused(refusal.code)
    else:
        response = _redirect(done.return_to)
        if done.session is not None:  # None after a step-up: the browser keeps its own, refreshed session
            session_cookie.set_session_cookie(response, done.session.token, done.session.absolute_expires_at)
    _clear_binding(response)
    return response


__all__ = ["SIGN_IN_COOKIE", "SIGN_IN_PAGE", "callback", "router", "start"]
