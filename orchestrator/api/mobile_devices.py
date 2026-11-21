"""PRD-266 §4.3 — the Automatos Go device routes (``/api/v1/mobile``).

* ``POST /pairing-codes`` — mints a one-time code that pairs a phone or a browser to the
  caller: ``{code, expires_at, qr_payload}``. In local, a workspace owner or admin
  (``require_workspace_admin``; the operator passes as super_admin). In enterprise (PRD-277
  US-601), any person signed in on the session lane, for themselves, in a workspace they are a
  member of (``pairing_code_owner``): never a device, a key or an all-workspaces view. 404 where
  devices never pair (saas, where phones sign in with Clerk; enterprise with
  ``ENTERPRISE_DEVICE_PAIRING`` off).
* ``POST /pair`` — **PUBLIC.** AGENTS.md exception 1, approved by Gerard on 20 Oct 2026
  (PRD-266 §4.10, tracking issue #1253), as PRD-264 D16 did for ``/verify``: the phone has
  no account yet, so the one-time code is its only credential. It answers 404 unless devices
  pair here and, on an install with an operator (local), ``LOCAL_REMOTE_ACCESS=paired``. A wrong, expired or used code is
  ``401 {"code": "invalid_pairing_code"}`` and nothing more, and never spends the pending
  code. The token is in the answer exactly once. (The web app offers browser pairing in local
  only; in enterprise a browser signs in.)
* ``POST /sign-in/exchange`` — **PUBLIC**, as ``/pair`` is (PRD-277 US-603): a phone that signed in
  at the install's own OIDC start address exchanges the one-time code the callback sent its app link,
  with its PKCE verifier, for a device token of kind ``oidc`` (``services.mobile.sign_in_codes``),
  returned once. 404 unless the install takes such devices (enterprise, pairing on or off); limited
  per IP with ``/pair``'s own bucket. Any code found is spent; every refusal is
  ``401 {"code": "invalid_sign_in_code"}``.
* ``POST /devices`` — a device-lane caller (``Bearer amd_…``) updates its own row; a Clerk
  caller upserts a ``clerk`` row by (workspace, person, ``install_id``) with no token hash
  (§8). ``{device_id, push_key}``: the key is in clear only on the first registration
  (K14). Any other lane (the anonymous local operator, an API key) is 403. A person holds at
  most ``devices.MAX_ACTIVE_DEVICES_PER_PERSON`` active devices in a workspace: one more is
  ``409 {"code": "device_limit_reached", "message": …}``.
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
one caller block pairing for everyone. The address is ``get_client_ip``'s, which reads it from
``core/security/client_ip.py`` and its two settings (``CLIENT_IP_HEADER``, and
``CLIENT_IP_PROXY_COUNT`` for proxies that append), so an install published for phones names the
header its proxy writes. Redis unreachable lets the
attempt through and logs an ERROR naming this limiter. ``POST /devices`` is limited the same
way, with ``mobile_devices_ip`` (the W2 review: each registration can add a push target).

In enterprise a code made, a phone paired and a device revoked are each one ``audit_logs`` row
in the same commit (``services.mobile.device_audit``); revoking a device already revoked writes
none. There ``POST /pairing-codes`` is also limited per person (``mobile_pairing_codes_person``,
counted the same way), and a person holds one pending code and makes none past the device cap
(``services.mobile.pairing``). Both pairing routes first ask that no
device paired under another edition is still live (``edition_change.ensure_confirmed``).

Every answer that carries a code or a token is ``Cache-Control: no-store``. Nothing here logs
a code, a token or a hash: a database error is logged by its type only, because its text
carries the statement's bound parameters. The routes and the limiter are plain ``def``
(AGENTS.md: the sync session and the Redis round trip run in the threadpool). Registered
through ``RouterSpec("api.mobile_devices")`` in ``router_manifest.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
from core import editions
from core.auth.bearer_registry import bearer_token
from core.auth.dependencies import RequestContext
from core.auth import hybrid
from core.auth.device import DEVICE_AUTH_TYPE, token_kinds
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import claimed_user_pk, resolve_user_pk
from core.auth.session_lane import SESSION_AUTH_TYPE
from core.auth.workspace_admin import may_see_own_workspace_health, require_workspace_admin
from core.database.database import get_db
from core.models.mobile_devices import (
    APP_VERSION_LENGTH,
    EXPO_PUSH_TOKEN_LENGTH,
    INSTALL_ID_PATTERN,
    MOBILE_DEVICE_PLATFORMS,
    NAME_LENGTH,
    TIMEZONE_LENGTH,
    MobileDevice,
    MobileDeviceKind,
)
from core.security.ip_bucket import ip_rate_bucket
from core.security.rate_limiter import DEFAULT_LIMITS
from services.mobile import device_audit, devices, edition_change, pairing, sign_in_codes
from services.mobile.urls import MobileUrlError
from services.push.payload import OPTIONAL_PUSH_TYPES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mobile", tags=["mobile"])

T = TypeVar("T")

PAIRED_MODE = "paired"
PAIR_RATE_LIMIT_OPERATION = "mobile_pair_ip"
PAIR_RATE_BUCKET_PREFIX = "mobile_pair:ip"
# What the shared limiter's fail-open ERROR line calls this limiter.
RATE_LIMITER_LABEL = "mobile pairing rate limiter"
DEVICES_RATE_LIMIT_OPERATION = "mobile_devices_ip"
DEVICES_RATE_BUCKET_PREFIX = "mobile_devices:ip"
DEVICES_RATE_LIMITER_LABEL = "mobile device registration rate limiter"
CODES_RATE_LIMIT_OPERATION = "mobile_pairing_codes_person"
CODES_RATE_BUCKET_PREFIX = "mobile_pairing_codes:person"
CODES_RATE_LIMITER_LABEL = "mobile pairing code rate limiter"
CODES_RATE_LIMITED_DETAIL = "Too many pairing codes made. Try again in a few minutes."
INVALID_PAIRING_CODE = "invalid_pairing_code"
INVALID_SIGN_IN_CODE = "invalid_sign_in_code"
# Room for the spaces a person may type; the kernel normalises what is hashed.
PAIRING_CODE_MAX_LENGTH = 32
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
# What FastAPI answers for a route that is not there, so a closed route looks absent.
NOT_FOUND = "Not Found"
NO_WORKSPACE = "Choose a workspace first."
NO_PERSON = "A device is paired to a person, and this caller is not one."
PAIR_FROM_THE_BROWSER = "Pair a phone from Automatos in your browser, signed in as yourself."
PAIR_IN_YOUR_WORKSPACE = "Pair a phone from a workspace you are a member of."
RATE_LIMITED_DETAIL = "Too many pairing attempts from this address. Try again later."
DEVICES_RATE_LIMITED_DETAIL = "Too many device registrations from this address. Try again later."
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
PUSH_TOKEN_IN_USE = devices.PUSH_TOKEN_IN_USE

# Module-level so a test can hand it a fake Redis (RateLimitStore's redis_factory seam).
_rate_store = RateLimitStore()


def _known_platform(value: str) -> str:
    if value not in MOBILE_DEVICE_PLATFORMS:
        raise ValueError(f"platform must be one of: {', '.join(MOBILE_DEVICE_PLATFORMS)}")
    return value


def _known_push_category(value: str) -> str:
    if value not in OPTIONAL_PUSH_TYPES:
        raise ValueError(f"push_categories may hold only: {', '.join(OPTIONAL_PUSH_TYPES)}")
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
PushCategory = Annotated[str, AfterValidator(_known_push_category)]
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


class SignInExchangeRequest(BaseModel):
    """``POST /sign-in/exchange``: a phone's one-time sign-in code, its PKCE verifier, and how it is named.

    The code and the verifier carry no length rule here: any wrong one is the exchange's one refusal
    (``sign_in_codes.exchange_code``), never a validation error that would echo what was sent.
    """

    code: str
    code_verifier: str
    name: Optional[str] = Field(default=None, max_length=NAME_LENGTH)
    app_version: Optional[str] = Field(default=None, max_length=APP_VERSION_LENGTH)

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
    name: Optional[str] = Field(default=None, min_length=1, max_length=NAME_LENGTH)
    push_categories: List[PushCategory] = Field(default_factory=list, max_length=len(OPTIONAL_PUSH_TYPES))


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


def _require_device_pairing() -> None:
    """404 where devices never pair (``device_pairing`` off): saas phones sign in with Clerk.

    Then 503 while this process cannot confirm no device of another edition is live.
    """
    if not editions.current().device_pairing:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    edition_change.ensure_confirmed()


def _require_paired_mode() -> None:
    """404 unless this install takes paired devices.

    On an install with an operator (local) that is ``LOCAL_REMOTE_ACCESS=paired``; elsewhere
    (enterprise) ``device_pairing`` alone, which ``ENTERPRISE_DEVICE_PAIRING`` decides.
    """
    _require_device_pairing()
    if editions.current().anonymous_operator and config.LOCAL_REMOTE_ACCESS != PAIRED_MODE:
        raise HTTPException(status_code=404, detail=NOT_FOUND)


def _require_phone_sign_in() -> None:
    """404 unless this install takes devices signed in on the phone (``token_kinds``: enterprise,
    whatever the pairing setting); then 503 while no device of another edition is known gone."""
    if MobileDeviceKind.OIDC.value not in token_kinds():
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    edition_change.ensure_confirmed()


def pairing_code_owner(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> RequestContext:
    """Who may make a pairing code, which pairs a device to its maker.

    Where one install holds many people (``multi_person``: enterprise), a person signed in on the
    session lane, in a workspace whose member they are by the device lane's own rule
    (``hybrid._user_pk_is_workspace_member``), so a code is never made that its device could not
    use; any workspace role. A device, a key or an install admin's all-workspaces view is 403.
    Elsewhere (local) a workspace owner or admin, the operator included, as before.
    """
    if not editions.current().multi_person:
        return require_workspace_admin(ctx, db)
    person = claimed_user_pk(ctx) if ctx.auth_type == SESSION_AUTH_TYPE else None
    if person is None:
        raise HTTPException(status_code=403, detail=PAIR_FROM_THE_BROWSER)
    if ctx.admin_all_workspaces or ctx.workspace_id is None \
            or not hybrid._user_pk_is_workspace_member(db, ctx.workspace_id, person):
        raise HTTPException(status_code=403, detail=PAIR_IN_YOUR_WORKSPACE)
    return ctx


def _browser(request: Request) -> Dict[str, Optional[str]]:
    """The caller's address and browser, for an audit row."""
    return {"ip_address": get_client_ip(request), "user_agent": request.headers.get("user-agent")}


@dataclass(frozen=True)
class _IpLimit:
    """One limit (per IP, or per person): its ``DEFAULT_LIMITS`` operation, Redis bucket, log label and 429 words."""

    operation: str
    bucket_prefix: str
    label: str
    detail: str


_PAIR_LIMIT = _IpLimit(PAIR_RATE_LIMIT_OPERATION, PAIR_RATE_BUCKET_PREFIX, RATE_LIMITER_LABEL, RATE_LIMITED_DETAIL)
_DEVICES_LIMIT = _IpLimit(DEVICES_RATE_LIMIT_OPERATION, DEVICES_RATE_BUCKET_PREFIX, DEVICES_RATE_LIMITER_LABEL,
                          DEVICES_RATE_LIMITED_DETAIL)
_CODES_LIMIT = _IpLimit(CODES_RATE_LIMIT_OPERATION, CODES_RATE_BUCKET_PREFIX, CODES_RATE_LIMITER_LABEL,
                        CODES_RATE_LIMITED_DETAIL)


def _check_bucket(bucket: str, spec: _IpLimit) -> None:
    """429 with ``Retry-After`` once ``bucket`` has spent ``spec``'s limit in its window."""
    limit, window = DEFAULT_LIMITS[spec.operation]
    allowed, _limit, _remaining, reset_seconds = _rate_store.check(bucket, limit, window, label=spec.label)
    if not allowed:
        raise HTTPException(status_code=429, detail=spec.detail, headers={"Retry-After": str(reset_seconds)})


def _check_ip_limit(request: Request, ip_limit: _IpLimit) -> None:
    """429 with ``Retry-After`` once this client IP has spent ``ip_limit`` in its window."""
    _check_bucket(ip_rate_bucket(get_client_ip(request), ip_limit.bucket_prefix), ip_limit)


def limit_codes_per_person(user_pk: int) -> None:
    """429 once this person has made ``mobile_pairing_codes_person`` codes in its window.

    Only where codes are a person's own (``pairing.codes_are_a_persons_own``: enterprise).
    """
    if pairing.codes_are_a_persons_own():
        _check_bucket(f"{CODES_RATE_BUCKET_PREFIX}:{user_pk}", _CODES_LIMIT)


def limit_pair_attempts_per_ip(request: Request) -> None:
    """429 once this client IP has spent its pairing attempts. Runs before the code is read."""
    _check_ip_limit(request, _PAIR_LIMIT)


def limit_device_registrations_per_ip(request: Request) -> None:
    """429 once this client IP has spent its device registrations. Runs before the body is read."""
    _check_ip_limit(request, _DEVICES_LIMIT)


def _clash_detail(clash: IntegrityError) -> Any:
    """A 409's detail: the push token's own code when its unique index refused the write."""
    constraint = getattr(getattr(clash.orig, "diag", None), "constraint_name", None)
    return {"code": PUSH_TOKEN_IN_USE} if constraint == PUSH_TOKEN_INDEX else CONCURRENT_CHANGE


def _write(db: Session, work: Callable[[], T]) -> T:
    """Run one service call and commit it once.

    A ``DeviceConflict`` is ``409 {"code": …}``, with its plain ``message`` when it carries one
    (``device_limit_reached``). A unique index is 409 (``push_token_in_use``
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
        detail = {"code": conflict.code, **({"message": conflict.message} if conflict.message else {})}
        raise HTTPException(status_code=409, detail=detail) from conflict
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


@router.post("/pairing-codes", dependencies=[Depends(_require_device_pairing)])
def create_pairing_code(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(pairing_code_owner),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Mint a one-time code that pairs a phone or a browser to the caller. Shown once; ten minutes.

    In enterprise a person makes ``mobile_pairing_codes_person`` codes a window (429 beyond), holds
    one at a time, and none while they hold the most active devices (409 ``device_limit_reached``).
    """
    user_pk = _person(db, ctx)
    limit_codes_per_person(user_pk)
    base_url = str(request.base_url)

    def issue() -> pairing.PairingCode:
        made = pairing.create_pairing_code(db, ctx.workspace_id, user_pk, request_base_url=base_url)
        if device_audit.records_devices():
            device_audit.stage(db, action=device_audit.PAIRING_CODE_CREATED, workspace_id=ctx.workspace_id,
                               actor=user_pk, details={"expires_at": made.expires_at.isoformat()},
                               **_browser(request))
        return made

    try:
        issued = _write(db, issue)
    except MobileUrlError as misconfigured:
        logger.error("pairing code refused: %s", misconfigured)
        raise HTTPException(status_code=500, detail=NOT_SET_UP.format(reason=misconfigured)) from misconfigured
    response.headers.update(NO_STORE_HEADERS)
    return {"code": issued.code, "expires_at": issued.expires_at.isoformat(), "qr_payload": issued.qr_payload}


@router.post("/pair", dependencies=[Depends(_require_paired_mode), Depends(limit_pair_attempts_per_ip)])
def pair(body: PairRequest, request: Request, response: Response, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Exchange a one-time code for a device token, returned exactly once (PUBLIC: module docstring)."""

    def claim() -> Optional[pairing.PairedDevice]:
        claimed = pairing.pair_device(db, body.code, body.name, body.platform, body.app_version, body.install_id)
        if claimed is not None and device_audit.records_devices():
            person = db.get(MobileDevice, claimed.device_id).user_id
            device_audit.stage(db, action=device_audit.DEVICE_PAIRED, workspace_id=claimed.workspace_id,
                               actor=person, resource_id=claimed.device_id,
                               details={"platform": body.platform}, **_browser(request))
        return claimed

    paired = _write(db, claim)
    if paired is None:
        raise HTTPException(status_code=401, detail={"code": INVALID_PAIRING_CODE})
    response.headers.update(NO_STORE_HEADERS)
    return {
        "device_id": str(paired.device_id),
        "workspace_id": str(paired.workspace_id),
        "token": paired.token,
        "rotate_after": paired.rotate_after.isoformat(),
    }


@router.post("/sign-in/exchange",
             dependencies=[Depends(_require_phone_sign_in), Depends(limit_pair_attempts_per_ip)])
def exchange_sign_in_code(body: SignInExchangeRequest, request: Request, response: Response,
                          db: Session = Depends(get_db)) -> Dict[str, str]:
    """Exchange a phone's one-time sign-in code and its PKCE verifier for a device token, returned once.

    PUBLIC, as ``/pair`` is (PRD-277 US-603): the code and the verifier are the phone's only
    credential. Any code found is spent, so a wrong verifier ends it. An unknown, used, expired or
    wrong-verifier code is ``401 {"code": "invalid_sign_in_code"}`` and nothing more.
    """
    described = (body.name, body.app_version)

    def claim() -> Optional[sign_in_codes.Exchanged]:
        exchanged = sign_in_codes.exchange_code(db, body.code, body.code_verifier, described)
        if exchanged is not None and device_audit.records_devices():
            device_audit.stage(db, action=device_audit.DEVICE_PAIRED, workspace_id=exchanged.device.workspace_id,
                               actor=exchanged.user_pk, resource_id=exchanged.device.device_id,
                               details={"platform": exchanged.platform, "kind": MobileDeviceKind.OIDC.value},
                               **_browser(request))
        return exchanged

    done = _write(db, claim)
    if done is None:
        raise HTTPException(status_code=401, detail={"code": INVALID_SIGN_IN_CODE}, headers=dict(NO_STORE_HEADERS))
    response.headers.update(NO_STORE_HEADERS)
    return {
        "device_id": str(done.device.device_id),
        "workspace_id": str(done.device.workspace_id),
        "token": done.device.token,
        "rotate_after": done.device.rotate_after.isoformat(),
    }


@router.post("/devices", dependencies=[Depends(limit_device_registrations_per_ip)])
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
    return {"devices": devices.list_devices(db, caller)}


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
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Revoke a device (its person, or a workspace owner/admin); its next call is 401 device_revoked."""
    caller = _caller(db, ctx, workspace_view=True)

    def revoke() -> Optional[MobileDevice]:
        done = devices.revoke_device(db, caller, device_id)
        if done is None:
            return None
        if done.changed and device_audit.records_devices():  # a device already revoked: no second row
            device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=caller.workspace_id,
                               actor=caller.user_pk, resource_id=done.device.id,
                               details={"reason": device_audit.REVOKED_BY_PERSON, "person": done.device.user_id},
                               **_browser(request))
        return done.device

    return _found(_write(db, revoke)).to_dict()


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
