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
* ``POST /devices`` — a device-lane caller (``Bearer amd_…``) updates its own row; a Clerk
  caller upserts a ``clerk`` row by (workspace, person, ``install_id``) with no token hash
  (§8). ``{device_id, push_key}``: the key is in clear only on the first registration
  (K14). Any other lane (the anonymous local operator, an API key) is 403.
* ``GET /devices`` — the caller's own devices; a workspace owner/admin (and the local
  operator) also sees the workspace's (``may_see_own_workspace_health``). ``to_dict()`` only.
* ``PATCH /devices/{device_id}`` — the device's person only: ``push_enabled``,
  ``digest_time``, ``timezone``; any other field is 422.
* ``DELETE /devices/{device_id}`` — the device's person, or a workspace owner/admin:
  revoked, its token hash kept, so its next call is ``401 {"code": "device_revoked"}``.
* ``POST /devices/{device_id}/rotate`` — the device lane on its own device, with its current
  token: a new token once; the old one lives on for ``MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS``.

Every device route is scoped to the caller's workspace (``services.mobile.devices``): a
device the caller may not act on, another workspace's included, is 404, never 403. A call
the device's row contradicts (its token names another install; a rotated-out token asking
to rotate; a clerk row an admin revoked) is ``409 {"code": …}``, and so is an Expo push
token another row already holds (``push_token_in_use``, the unique index).

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
from typing import Annotated, Any, Callable, Dict, List, Optional, TypeVar
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from api.credentials import get_client_ip
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.bearer_registry import bearer_token
from core.auth.dependencies import RequestContext
from core.auth.device import DEVICE_AUTH_TYPE
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_admin import may_see_own_workspace_health, require_workspace_admin
from core.database.database import get_db
from core.models.mobile_devices import (
    APP_VERSION_LENGTH,
    EXPO_PUSH_TOKEN_LENGTH,
    INSTALL_ID_LENGTH,
    MOBILE_DEVICE_PLATFORMS,
    NAME_LENGTH,
    TIMEZONE_LENGTH,
)
from core.security.ip_bucket import ip_rate_bucket
from core.security.rate_limiter import DEFAULT_LIMITS
from services.mobile import devices, pairing
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
CLERK_AUTH_TYPE = "clerk"
# The lanes that carry a phone: its device token, or a signed-in (Clerk) account.
REGISTRAR_AUTH_TYPES = (DEVICE_AUTH_TYPE, CLERK_AUTH_TYPE)
NOT_A_REGISTRAR = "A phone registers with its device token or a signed-in account."
NO_SUCH_DEVICE = "Device not found."
BAD_DEVICE_CLAIM = "Invalid or expired token"
# BCP 47's recommended maximum length for a language tag (RFC 5646 §4.4.1).
LOCALE_MAX_LENGTH = 35
DIGEST_TIME_PATTERN = r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$"
EXPO_PUSH_TOKEN_PATTERN = r"^ExponentPushToken\[[^\[\]\s]+\]$"
# The unique index on expo_push_token (core/models/mobile_devices.py): another row holds it.
PUSH_TOKEN_INDEX = "uq_mobile_devices_expo_push_token"
PUSH_TOKEN_IN_USE = "push_token_in_use"

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def _known_platform(value: str) -> str:
    if value not in MOBILE_DEVICE_PLATFORMS:
        raise ValueError(f"platform must be one of: {', '.join(MOBILE_DEVICE_PLATFORMS)}")
    return value


def _iana_zone(value: Optional[str]) -> Optional[str]:
    """An IANA time zone name zoneinfo knows (``Europe/Dublin``), or ``None`` (cleared)."""
    if value is None:
        return None
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError) as unknown:
        raise ValueError("timezone must be an IANA time zone name, such as Europe/Dublin") from unknown
    return value


Platform = Annotated[str, AfterValidator(_known_platform)]
InstallId = Annotated[str, Field(pattern=INSTALL_ID_PATTERN)]
TimeZoneName = Annotated[str, Field(min_length=1, max_length=TIMEZONE_LENGTH), AfterValidator(_iana_zone)]
DigestTime = Annotated[str, Field(pattern=DIGEST_TIME_PATTERN)]


class PairRequest(BaseModel):
    """``POST /pair``: the one-time code and the device that presents it."""

    code: str = Field(min_length=1, max_length=PAIRING_CODE_MAX_LENGTH)
    name: Optional[str] = Field(default=None, max_length=NAME_LENGTH)
    platform: Platform
    app_version: Optional[str] = Field(default=None, max_length=APP_VERSION_LENGTH)
    install_id: InstallId

    @field_validator("name", "app_version")
    @classmethod
    def _blank_is_none(cls, value: Optional[str]) -> Optional[str]:
        return (value or "").strip() or None


class RegisterDeviceRequest(BaseModel):
    """``POST /devices``: this phone as it is now. An optional field left out keeps its value."""

    install_id: InstallId
    platform: Platform
    app_version: str = Field(min_length=1, max_length=APP_VERSION_LENGTH)
    locale: str = Field(min_length=1, max_length=LOCALE_MAX_LENGTH)
    timezone: TimeZoneName
    expo_push_token: Optional[str] = Field(default=None, max_length=EXPO_PUSH_TOKEN_LENGTH,
                                           pattern=EXPO_PUSH_TOKEN_PATTERN)
    digest_time: Optional[DigestTime] = None


class UpdateDeviceRequest(BaseModel):
    """``PATCH /devices/{device_id}``: only these three; any other field is 422."""

    model_config = ConfigDict(extra="forbid")

    push_enabled: Optional[bool] = None
    digest_time: Optional[DigestTime] = None
    timezone: Optional[TimeZoneName] = None

    @field_validator("push_enabled")
    @classmethod
    def _a_switch_is_on_or_off(cls, value: Optional[bool]) -> bool:
        # Runs only when the field is sent: left out keeps the switch, null is refused.
        if value is None:
            raise ValueError("push_enabled must be true or false")
        return value


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


def _clash_detail(clash: IntegrityError) -> Any:
    """A 409's detail: the push token's own code when its unique index refused the write."""
    constraint = getattr(getattr(clash.orig, "diag", None), "constraint_name", None)
    return {"code": PUSH_TOKEN_IN_USE} if constraint == PUSH_TOKEN_INDEX else CONCURRENT_CHANGE


def _write(db: Session, work: Callable[[], T]) -> T:
    """Run one service call and commit it once.

    A ``DeviceConflict`` is ``409 {"code": …}``. A unique index is 409 (``push_token_in_use``
    when another row holds the Expo token); any other database error is 503. Both are logged
    by type only: the error's text carries the bound parameters, a code or token hash among
    them, and the app's global handler would log that text.
    """
    try:
        result = work()
        db.commit()
        return result
    except devices.DeviceConflict as conflict:
        db.rollback()
        raise HTTPException(status_code=409, detail={"code": conflict.code}) from conflict
    except IntegrityError as clash:
        db.rollback()
        logger.warning("mobile device write lost a race on a unique index: %s", type(clash.orig).__name__)
        raise HTTPException(status_code=409, detail=_clash_detail(clash)) from clash
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("mobile device write failed: %s", type(failure).__name__)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from failure


def _person(db: Session, ctx: RequestContext) -> int:
    """The caller's ``users.id``: 404 without a workspace, 403 when no person stands behind it."""
    if ctx.workspace_id is None:
        raise HTTPException(status_code=404, detail=NO_WORKSPACE)
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        raise HTTPException(status_code=403, detail=NO_PERSON)
    return user_pk


def _lane_device_id(ctx: RequestContext) -> Optional[UUID]:
    """The device a device-lane caller's token opened (``raw_claims.device_id``); None otherwise."""
    if ctx.auth_type != DEVICE_AUTH_TYPE:
        return None
    try:
        return UUID(str((ctx.user.raw_claims or {})["device_id"]))
    except (KeyError, ValueError) as malformed:
        raise HTTPException(status_code=401, detail=BAD_DEVICE_CLAIM) from malformed


def _caller(db: Session, ctx: RequestContext, *, workspace_view: bool = False) -> devices.DeviceCaller:
    """The caller as the devices service sees it; ``workspace_view`` asks whether it is an owner/admin."""
    user_pk = _person(db, ctx)
    return devices.DeviceCaller(
        workspace_id=ctx.workspace_id,
        user_pk=user_pk,
        device_id=_lane_device_id(ctx),
        sees_workspace=workspace_view and may_see_own_workspace_health(db, ctx),
    )


def _found(value: Optional[T]) -> T:
    """404 for a device the caller may not act on, another workspace's included."""
    if value is None:
        raise HTTPException(status_code=404, detail=NO_SUCH_DEVICE)
    return value


@router.post("/pairing-codes", dependencies=[Depends(_require_local_edition)])
def create_pairing_code(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Mint a one-time code that pairs a phone or a browser to the caller. Shown once; ten minutes."""
    user_pk = _person(db, ctx)
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


@router.post("/devices")
def register_device(
    body: RegisterDeviceRequest,
    response: Response,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Optional[str]]:
    """Register this phone, or update its row: ``{device_id, push_key}``, the key only the first time."""
    if ctx.auth_type not in REGISTRAR_AUTH_TYPES:
        raise HTTPException(status_code=403, detail=NOT_A_REGISTRAR)
    caller = _caller(db, ctx)
    described = body.model_dump(exclude_unset=True, exclude={"install_id"})
    registered = _found(_write(db, lambda: devices.register_device(db, caller, body.install_id, described)))
    response.headers.update(NO_STORE_HEADERS)
    return {"device_id": str(registered.device_id), "push_key": registered.push_key}


@router.get("/devices")
def list_devices(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, List[Dict[str, Any]]]:
    """The caller's devices; a workspace owner/admin (and the local operator) sees the workspace's."""
    caller = _caller(db, ctx, workspace_view=True)
    return {"devices": [device.to_dict() for device in devices.list_devices(db, caller)]}


@router.patch("/devices/{device_id}")
def update_device(
    device_id: UUID,
    body: UpdateDeviceRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Change the push switch, digest time or time zone of one of the caller's own devices."""
    caller = _caller(db, ctx)
    changes = body.model_dump(exclude_unset=True)
    return _found(_write(db, lambda: devices.update_device(db, caller, device_id, changes))).to_dict()


@router.delete("/devices/{device_id}")
def revoke_device(
    device_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Revoke a device (its person, or a workspace owner/admin); its next call is 401 device_revoked."""
    caller = _caller(db, ctx, workspace_view=True)
    return _found(_write(db, lambda: devices.revoke_device(db, caller, device_id))).to_dict()


@router.post("/devices/{device_id}/rotate")
def rotate_device_token(
    device_id: UUID,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Swap the calling device's token for a new one, returned once; the old one lives on briefly."""
    caller = _caller(db, ctx)
    presented = bearer_token(request)
    rotated = _found(_write(db, lambda: devices.rotate_token(db, caller, device_id, presented)))
    response.headers.update(NO_STORE_HEADERS)
    return {
        "device_id": str(rotated.device_id),
        "token": rotated.token,
        "rotate_after": rotated.rotate_after.isoformat(),
        "previous_token_valid_until": rotated.previous_valid_until.isoformat(),
    }
