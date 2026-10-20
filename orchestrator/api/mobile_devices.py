"""PRD-266 §4.3 — the Automatos Go device routes (``/api/v1/mobile``).

* ``POST /pairing-codes`` — a workspace owner or admin (``require_workspace_admin``; the
  local operator passes as super_admin) mints a one-time code that pairs a phone or a
  browser to themselves: ``{code, expires_at, qr_payload}``. Local edition only: 404 in saas,
  where phones sign in with Clerk.
* ``POST /pair`` — **PUBLIC.** AGENTS.md exception 1, approved by Gerard on 20 Oct 2026
  (PRD-266 §4.10, tracking issue #1253), as PRD-264 D16 did for ``/verify``: the phone has
  no account yet, so the one-time code is its only credential. It answers 404 unless
  ``AUTH_EDITION=local`` and ``LOCAL_REMOTE_ACCESS=paired``. A wrong, expired or used code is
  ``401 {"code": "invalid_pairing_code"}`` and nothing more, and never spends the pending
  code. The token is in the answer exactly once.

The limit on ``/pair`` (K13, the lead's review of 20 Oct): each client IP (an IPv6 /64,
``core.security.ip_bucket``) gets the ``mobile_pair_ip`` attempts of ``DEFAULT_LIMITS`` a
window, counted in the shared Redis sliding window (``RateLimitStore``, the
``api/provenance.py`` pattern) BEFORE the body's code is read; over it is 429 with
``Retry-After``. There is no instance-wide bucket: 8 characters from a 32-symbol alphabet
alive for 600 s cannot be guessed at any rate the API serves, and a shared bucket would let
one caller block pairing for everyone. The address is ``get_client_ip``'s (the first
``X-Forwarded-For`` hop), so the bucket is only as honest as the proxy in front: an install
published for phones sits behind one that replaces that header. Redis unreachable lets the
attempt through and logs an ERROR naming this limiter.

Every answer that carries a code or a token is ``Cache-Control: no-store``. Nothing here logs
a code, a token or a hash: a database error is logged by its type only, because its text
carries the statement's bound parameters. The routes and the limiter are plain ``def``
(AGENTS.md: the sync session and the Redis round trip run in the threadpool). Registered
through ``RouterSpec("api.mobile_devices")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.credentials import get_client_ip
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.mobile_devices import (
    APP_VERSION_LENGTH,
    INSTALL_ID_LENGTH,
    MOBILE_DEVICE_PLATFORMS,
    NAME_LENGTH,
)
from core.security.ip_bucket import ip_rate_bucket
from core.security.rate_limiter import DEFAULT_LIMITS
from services.mobile import pairing
from services.mobile.urls import MobileUrlError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mobile", tags=["mobile"])

T = TypeVar("T")

PAIRED_MODE = "paired"
PAIR_RATE_LIMIT_OPERATION = "mobile_pair_ip"
PAIR_RATE_BUCKET_PREFIX = "mobile_pair:ip"
# What the shared limiter's fail-open ERROR line calls this limiter.
RATE_LIMITER_LABEL = "mobile pairing rate limiter"
INVALID_PAIRING_CODE = "invalid_pairing_code"
# Room for the spaces a person may type; the kernel normalises what is hashed.
PAIRING_CODE_MAX_LENGTH = 32
INSTALL_ID_MIN_LENGTH = 8
INSTALL_ID_PATTERN = rf"^[A-Za-z0-9_-]{{{INSTALL_ID_MIN_LENGTH},{INSTALL_ID_LENGTH}}}$"
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
# What FastAPI answers for a route that is not there, so a closed route looks absent.
NOT_FOUND = "Not Found"
NO_WORKSPACE = "Choose a workspace first."
NO_PERSON = "A device is paired to a person, and this caller is not one."
RATE_LIMITED_DETAIL = "Too many pairing attempts from this address. Try again later."
CONCURRENT_CHANGE = "This device changed at the same moment. Try again."
UNAVAILABLE = "Pairing is unavailable right now. Try again in a minute."
NOT_SET_UP = "Pairing is not set up on this install: {reason}."

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


class PairRequest(BaseModel):
    """``POST /pair``: the one-time code and the device that presents it."""

    code: str = Field(min_length=1, max_length=PAIRING_CODE_MAX_LENGTH)
    name: Optional[str] = Field(default=None, max_length=NAME_LENGTH)
    platform: str
    app_version: Optional[str] = Field(default=None, max_length=APP_VERSION_LENGTH)
    install_id: str = Field(pattern=INSTALL_ID_PATTERN)

    @field_validator("platform")
    @classmethod
    def _known_platform(cls, value: str) -> str:
        if value not in MOBILE_DEVICE_PLATFORMS:
            raise ValueError(f"platform must be one of: {', '.join(MOBILE_DEVICE_PLATFORMS)}")
        return value

    @field_validator("name", "app_version")
    @classmethod
    def _blank_is_none(cls, value: Optional[str]) -> Optional[str]:
        return (value or "").strip() or None


def _require_local_edition() -> None:
    """404 outside the local edition: saas phones sign in with Clerk and never pair."""
    if not config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=404, detail=NOT_FOUND)


def _require_paired_mode() -> None:
    """404 unless this local install takes paired devices (``LOCAL_REMOTE_ACCESS=paired``)."""
    _require_local_edition()
    if config.LOCAL_REMOTE_ACCESS != PAIRED_MODE:
        raise HTTPException(status_code=404, detail=NOT_FOUND)


def limit_pair_attempts_per_ip(request: Request) -> None:
    """429 once this client IP has spent its pairing attempts. Runs before the code is read."""
    limit, window = DEFAULT_LIMITS[PAIR_RATE_LIMIT_OPERATION]
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(
        ip_rate_bucket(get_client_ip(request), PAIR_RATE_BUCKET_PREFIX), limit, window,
        label=RATE_LIMITER_LABEL,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=RATE_LIMITED_DETAIL,
                            headers={"Retry-After": str(reset_seconds)})


def _write(db: Session, work: Callable[[], T]) -> T:
    """Run one service call and commit it once.

    A lost race on a unique index is 409; any other database error is 503. Both are logged
    by type only: the error's text carries the bound parameters, a code or token hash among
    them, and the app's global handler would log that text.
    """
    try:
        result = work()
        db.commit()
        return result
    except IntegrityError as clash:
        db.rollback()
        logger.warning("mobile device write lost a race on a unique index: %s", type(clash.orig).__name__)
        raise HTTPException(status_code=409, detail=CONCURRENT_CHANGE) from clash
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("mobile device write failed: %s", type(failure).__name__)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from failure


@router.post("/pairing-codes", dependencies=[Depends(_require_local_edition)])
def create_pairing_code(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Mint a one-time code that pairs a phone or a browser to the caller. Shown once; ten minutes."""
    if ctx.workspace_id is None:
        raise HTTPException(status_code=404, detail=NO_WORKSPACE)
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        raise HTTPException(status_code=403, detail=NO_PERSON)
    base_url = str(request.base_url)
    try:
        issued = _write(db, lambda: pairing.create_pairing_code(
            db, ctx.workspace_id, user_pk, request_base_url=base_url))
    except MobileUrlError as misconfigured:
        logger.error("pairing code refused: %s", misconfigured)
        raise HTTPException(status_code=500, detail=NOT_SET_UP.format(reason=misconfigured)) from misconfigured
    response.headers.update(NO_STORE_HEADERS)
    return {"code": issued.code, "expires_at": issued.expires_at.isoformat(), "qr_payload": issued.qr_payload}


@router.post("/pair", dependencies=[Depends(_require_paired_mode), Depends(limit_pair_attempts_per_ip)])
def pair(body: PairRequest, response: Response, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Exchange a one-time code for a device token, returned exactly once (PUBLIC: module docstring)."""
    paired = _write(db, lambda: pairing.pair_device(
        db, body.code, body.name, body.platform, body.app_version, body.install_id))
    if paired is None:
        raise HTTPException(status_code=401, detail={"code": INVALID_PAIRING_CODE})
    response.headers.update(NO_STORE_HEADERS)
    return {
        "device_id": str(paired.device_id),
        "workspace_id": str(paired.workspace_id),
        "token": paired.token,
        "rotate_after": paired.rotate_after.isoformat(),
    }
