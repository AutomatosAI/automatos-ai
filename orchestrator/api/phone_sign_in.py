"""PRD-277 US-603 (§5.7): the routes of the page a phone's sign-in waits on (``/api/v1/auth/oidc/phone``).

After a phone's callback (``api/auth_oidc.py``) the browser holds ``phone_sign_in.CONFIRM_COOKIE`` and
is on the web app's confirmation page (``phone_sign_in.CONFIRM_PAGE``). These three routes are that
page's; the cookie is their only credential (``services.mobile.sign_in_confirmation``), and nothing
else here identifies anyone. Like the start and the callback they answer 404 unless the edition signs
in with OIDC, are plain ``def`` and are limited per client IP before the cookie is read
(``auth_oidc.limit_phone_confirmations_per_ip``, ``oidc_phone_confirm_ip``).

* ``GET /waiting`` — what awaits the person: ``{person, app, platform, removed_before}`` and nothing
  else (their e-mail address or name, the app's name from ``phone_sign_in.APP_NAMES``, ``ios`` or
  ``android``, and whether someone else had removed this phone). Nothing waiting (no cookie, expired,
  used, cancelled) is ``404 {"code": "nothing_waiting"}``, with the cookie cleared.
* ``POST /continue`` — the only step that makes a code: ``{app_link}``, the app link carrying the
  one-time code, for the page to open, with the cookie cleared and ``auth:sign_in`` (a phone's) written
  in the same commit as the code. Used once. Nothing waiting is the 404 above; a device cap reached
  since the callback is ``409 {"code": "device_limit_reached"}``, audited as the callback's refusal is.
* ``POST /cancel`` — deletes what waits, clears the cookie, ``204``; the same when nothing waits.

**Requests that change something** (the two POSTs) pass the edition's US-302 rule for cookie requests
(``session_cookie.assert_cookie_request_allowed``) before anything else, though no session exists:
the request's ``Origin`` (else its ``Referer``'s origin) must be the install's own, ``FRONTEND_URL``'s,
and it must carry ``SESSION_CSRF_HEADER``. A form another site posts carries that site's ``Origin``,
which the origin comparison refuses (403 ``cross_site_request_refused``); it cannot add the header
either. A navigation is never a POST, and the cookie is ``SameSite=Lax``, so a cross-site POST does not
even carry it. Every answer is ``Cache-Control: no-store`` and ``Referrer-Policy: no-referrer``.
Registered through ``RouterSpec("api.phone_sign_in")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.auth_oidc import NO_STORE_HEADERS, _browser, _signs_in_with_oidc, limit_phone_confirmations_per_ip
from core.auth import session_audit, session_cookie
from core.auth.phone_sign_in import APP_NAMES, CONFIRM_COOKIE, app_answer, clear_confirm_cookie
from core.database.database import get_db
from services.mobile import sign_in_codes, sign_in_confirmation

logger = logging.getLogger(__name__)

NOTHING_WAITING = "nothing_waiting"
FAILED = "sign_in_failed"

router = APIRouter(prefix="/api/v1/auth/oidc/phone", tags=["auth"], dependencies=[Depends(_signs_in_with_oidc)])


def sent_from_this_install(request: Request) -> None:
    """US-302's rule for a cookie request that changes something: same origin, with the header; else 403."""
    session_cookie.assert_cookie_request_allowed(request)


def _token(request: Request) -> Optional[str]:
    return request.cookies.get(CONFIRM_COOKIE)


def _answer(content: Optional[dict], code: int = status.HTTP_200_OK, *, clear: bool = False) -> Response:
    """The answer, never stored; with the confirmation cookie cleared when ``clear``."""
    if content is None:
        response = Response(status_code=code, headers=dict(NO_STORE_HEADERS))
    else:
        response = JSONResponse(content=content, status_code=code, headers=dict(NO_STORE_HEADERS))
    if clear:
        clear_confirm_cookie(response)
    return response


def _refusal(code: str, status_code: int) -> Response:
    """``{"detail": {"code": code}}``; the cookie cleared unless the failure was ours (503: try again)."""
    keep = status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    return _answer({"detail": {"code": code}}, status_code, clear=not keep)


@router.get("/waiting", dependencies=[Depends(limit_phone_confirmations_per_ip)])
def waiting(request: Request, db: Session = Depends(get_db)) -> Response:
    """What awaits the person in this browser, the four fields only; 404 ``nothing_waiting``."""
    found = sign_in_confirmation.waiting(db, _token(request))
    if found is None:
        return _refusal(NOTHING_WAITING, status.HTTP_404_NOT_FOUND)
    return _answer({"person": found.person, "app": APP_NAMES[found.app_link], "platform": found.platform,
                    "removed_before": found.removed_before})


def _confirmed(request: Request, db: Session):
    """Continue, its ``auth:sign_in`` row committed with the code; ``(confirmed or None, refusal or None)``."""
    browser = _browser(request)
    try:
        confirmed = sign_in_confirmation.confirm(db, _token(request))
        if confirmed is None:
            db.commit()  # an expired record, deleted
            return None, None
        session_audit.record_phone_sign_in(db, user_id=confirmed.user_pk, workspace_id=confirmed.workspace_id,
                                           platform=confirmed.platform, ip_address=browser.ip_address,
                                           user_agent=browser.user_agent)
        return confirmed, None
    except sign_in_codes.NoSignInCode as refused:
        session_audit.record_sign_in_refused(db, reason=refused.code, user_id=refused.user_pk,
                                             ip_address=browser.ip_address, user_agent=browser.user_agent)
        return None, refused.code
    except (session_audit.SessionAuditFailed, SQLAlchemyError) as error:
        db.rollback()
        logger.error("A phone's sign-in could not be confirmed (%s)", type(error).__name__)
        return None, FAILED


@router.post("/continue", dependencies=[Depends(sent_from_this_install), Depends(limit_phone_confirmations_per_ip)])
def continue_sign_in(request: Request, db: Session = Depends(get_db)) -> Response:
    """Continue: the app link carrying the one-time code, once; the cookie cleared."""
    confirmed, refusal = _confirmed(request, db)
    if refusal == FAILED:
        return _refusal(FAILED, status.HTTP_503_SERVICE_UNAVAILABLE)
    if refusal is not None:
        return _refusal(refusal, status.HTTP_409_CONFLICT)
    if confirmed is None:
        return _refusal(NOTHING_WAITING, status.HTTP_404_NOT_FOUND)
    return _answer({"app_link": app_answer(confirmed.app_link, code=confirmed.code)}, clear=True)


@router.post("/cancel", dependencies=[Depends(sent_from_this_install), Depends(limit_phone_confirmations_per_ip)])
def cancel_sign_in(request: Request, db: Session = Depends(get_db)) -> Response:
    """Cancel: what waits is deleted and the cookie cleared; 204 whether or not anything waited."""
    try:
        sign_in_confirmation.cancel(db, _token(request))
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("A phone's sign-in could not be cancelled (%s)", type(error).__name__)
        return _refusal(FAILED, status.HTTP_503_SERVICE_UNAVAILABLE)
    return _answer(None, status.HTTP_204_NO_CONTENT, clear=True)


__all__ = ["NOTHING_WAITING", "cancel_sign_in", "continue_sign_in", "router", "sent_from_this_install", "waiting"]
