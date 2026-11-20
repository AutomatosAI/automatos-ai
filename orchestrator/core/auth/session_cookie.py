"""PRD-277 §5.3, §5.11, US-301, US-302: the session cookie, and the check on requests it authenticates.

**The cookie.** A browser holds its session token in ``SESSION_COOKIE_NAME`` (default
``__Host-automatos_session``): ``Secure``, ``HttpOnly``, ``SameSite=Lax``, ``Path=/`` and no
``Domain``, which is what a ``__Host-`` name requires, so the page's scripts cannot read it and no
other host can set or receive it. Its ``Max-Age`` is what is left of the session's absolute
lifetime. Start-up refuses a name without that prefix (``core.edition_start_up``).

**Reading it.** ``cookie_token`` reads the cookie only where the edition issues sessions; the
bearer registry asks for it only when the request carries no ``Authorization`` header and no API
key, so a token in a header always wins and every other lane is untouched.

**Cross-site request forgery (US-302).** A request the cookie authenticates whose method is not
``GET``, ``HEAD`` or ``OPTIONS`` is refused (403 ``cross_site_request_refused``) unless both hold:

* its ``Origin`` (or, when a browser sent none, its ``Referer``'s origin) is the install's own,
  ``FRONTEND_URL``'s scheme, host and port (the web app and the API share that address, §5.3);
* it carries ``SESSION_CSRF_HEADER`` (default ``X-Automatos-CSRF``) with a value. A form on
  another site cannot send a custom header, and a script on another site cannot send this one:
  it is not among the headers the API's CORS settings allow, so the browser's preflight fails.

A request authenticated by a header is never checked: no browser attaches one by itself. Nothing
here reads a request that the cookie does not authenticate.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import HTTPException, Response, status
from starlette.requests import cookie_parser

from config import config
from core import editions

logger = logging.getLogger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
COOKIE_PATH = "/"
COOKIE_SAME_SITE = "lax"
DEFAULT_PORTS = {"http": 80, "https": 443}
# What a browser sends as Origin from a sandboxed frame or a privacy-sensitive context.
OPAQUE_ORIGIN = "null"

CROSS_SITE_REFUSED = "cross_site_request_refused"
CROSS_SITE_MESSAGE = "This request did not come from this install's own pages, so it was refused."


def cookie_name() -> str:
    """The session cookie's name (``SESSION_COOKIE_NAME``)."""
    return config.SESSION_COOKIE_NAME


def cookie_token(request: Any) -> Optional[str]:
    """The session cookie's value, or None; never read where the edition issues no sessions."""
    if not editions.current().issues_sessions:
        return None
    header = request.headers.get("cookie")
    if not header:
        return None
    value = cookie_parser(header).get(cookie_name())
    return (value.strip() or None) if isinstance(value, str) else None


def set_session_cookie(response: Response, token: str, absolute_expires_at: datetime,
                       now: Optional[datetime] = None) -> None:
    """Put ``token`` in the session cookie, kept until the session's absolute expiry."""
    moment = now or datetime.now(timezone.utc)
    max_age = max(0, int((absolute_expires_at - moment).total_seconds()))
    response.set_cookie(key=cookie_name(), value=token, max_age=max_age, path=COOKIE_PATH,
                        secure=True, httponly=True, samesite=COOKIE_SAME_SITE)


def clear_session_cookie(response: Response) -> None:
    """Tell the browser to drop the session cookie (the same attributes it was set with)."""
    response.delete_cookie(key=cookie_name(), path=COOKIE_PATH, secure=True, httponly=True,
                           samesite=COOKIE_SAME_SITE)


def _origin_of(url: Optional[str]) -> Optional[str]:
    """``scheme://host[:port]`` of ``url``, lower-cased, a default port dropped; None when it has none."""
    if not url or url.strip().lower() == OPAQUE_ORIGIN:
        return None
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme, host = (parts.scheme or "").lower(), (parts.hostname or "").lower()
    if scheme not in DEFAULT_PORTS or not host:
        return None
    shown_port = "" if port in (None, DEFAULT_PORTS[scheme]) else f":{port}"
    return f"{scheme}://{host}{shown_port}"


def install_origin() -> Optional[str]:
    """The install's own origin: ``FRONTEND_URL``'s."""
    return _origin_of(config.FRONTEND_URL)


def request_origin(request: Any) -> Optional[str]:
    """Where the request says it came from: its ``Origin``, else its ``Referer``'s origin."""
    origin = request.headers.get("origin")
    if origin is not None:
        return _origin_of(origin)
    return _origin_of(request.headers.get("referer"))


def _carries_csrf_header(request: Any) -> bool:
    value = request.headers.get(config.SESSION_CSRF_HEADER.lower())
    return isinstance(value, str) and bool(value.strip())


def assert_cookie_request_allowed(request: Any) -> None:
    """403 a cookie-authenticated request that changes something, unless same-origin with the header."""
    if str(getattr(request, "method", "")).upper() in SAFE_METHODS:
        return
    expected = install_origin()
    if expected is not None and request_origin(request) == expected and _carries_csrf_header(request):
        return
    logger.warning("Session lane: a cookie-authenticated %s was refused as cross-site", request.method)
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail={"code": CROSS_SITE_REFUSED, "message": CROSS_SITE_MESSAGE})


__all__ = [
    "CROSS_SITE_REFUSED",
    "SAFE_METHODS",
    "assert_cookie_request_allowed",
    "clear_session_cookie",
    "cookie_name",
    "cookie_token",
    "install_origin",
    "request_origin",
    "set_session_cookie",
]
