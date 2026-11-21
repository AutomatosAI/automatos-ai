"""PRD-266 §4.3 — pairing a phone (or a browser, K6) with a local install.

Studio mints a one-time code for the caller (``create_pairing_code``): a PENDING
``mobile_devices`` row of kind ``paired``, bound to the caller's workspace and person, that
holds only the SHA-256 of the code and its expiry. The phone exchanges the code once
(``pair_device``) for a device token, ``amd_`` plus 32 random bytes; only the token's
SHA-256 is kept and the token itself is returned exactly once.

* Alphabet, length, TTL, hashing, minting and the constant-time compare are the kernel's
  (``core.auth.pairing``); nothing here has its own.
* An unknown, expired or used code answers ``None`` and changes nothing, so a wrong guess
  never spends or locks the code the person is about to type.
* A phone that pairs again (same workspace, person and ``install_id``) keeps its EXISTING
  row: the new token, name, platform and version land on it, its old tokens and its push key
  die (its next ``POST /devices`` is a first registration again), a revoked row is active
  again, its audit keeps its history plus ``enrolled``, and the pending row the new code made
  is deleted. One phone is always one row
  (``uq_mobile_devices_workspace_user_install``) and its device id never changes.
* Minting a code first deletes the caller's own pending codes that have expired, so a dead
  code's row (and its hash) does not linger. Where codes are a person's own (no operator:
  enterprise, PRD-277 US-601) a person holds one pending code per workspace, so a new code
  replaces theirs, expired or not, and none is made while they hold
  ``MAX_ACTIVE_DEVICES_PER_PERSON`` active devices there (``DeviceConflict``, 409). Counted, not
  locked, as registration is.
* Pairing exists where ``device_pairing`` is on (local; enterprise while
  ``ENTERPRISE_DEVICE_PAIRING`` is): on saas these refuse (``PairingUnavailable``), so a
  paired row never exists there (§8), whatever calls them.
* These functions flush and never commit: the router commits once and maps a database error
  to 409/503 by its type. Nothing here logs, so a code, a token or a hash never reaches a log.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import UUID

from sqlalchemy.orm import Query, Session

from config import config
from core import editions
# The bearer prefix names the lane: the registry routes amd_ to the device lane.
from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX
from core.auth.pairing import (
    PAIRING_CODE_TTL_SECONDS,
    hash_secret,
    mint_token,
    new_pairing_code,
    normalize_pairing_code,
    secrets_match,
)
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from services.mobile.events import ENROLLED, with_event
from services.mobile.urls import api_url, web_url

DEVICE_TOKEN_BYTES = 32
# The QR payload: a universal link on the web app, never a custom scheme (any app can
# register a scheme, §4.1/§4.3). ``query`` is the url-encoded ``api`` and ``code``.
PAIR_LINK_TEMPLATE = "{web_url}/go/pair?{query}"


class PairingUnavailable(RuntimeError):
    """Pairing was asked of a saas instance, where phones sign in with Clerk instead."""


@dataclass(frozen=True)
class PairingCode:
    """A minted code, in clear exactly once: the code, when it dies and its pair link."""

    code: str
    expires_at: datetime
    qr_payload: str


@dataclass(frozen=True)
class PairedDevice:
    """A paired device: its ids, its token (in clear exactly once) and when to rotate it."""

    device_id: UUID
    workspace_id: UUID
    token: str
    rotate_after: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _unexpired(expires_at: Optional[datetime], now: datetime) -> bool:
    """True strictly before ``expires_at``; a missing expiry never pairs. Naive reads as UTC."""
    if expires_at is None:
        return False
    deadline = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=timezone.utc)
    return now < deadline


def _require_device_pairing() -> None:
    if not editions.current().device_pairing:
        raise PairingUnavailable("devices do not pair on this install")


def _pending_codes(db: Session, workspace_id: UUID, user_pk: int) -> Query:
    """This person's pending codes in this workspace."""
    return db.query(MobileDevice).filter(
        MobileDevice.workspace_id == workspace_id,
        MobileDevice.user_id == user_pk,
        MobileDevice.kind == MobileDeviceKind.PAIRED.value,
        MobileDevice.status == MobileDeviceStatus.PENDING.value,
    )


def codes_are_a_persons_own() -> bool:
    """Where the install has no operator (enterprise), a code is one person's own: they hold one at a time."""
    return not editions.current().anonymous_operator


def _make_room_for_a_code(db: Session, workspace_id: UUID, user_pk: int, now: datetime) -> None:
    """Clear the way for this person's new code.

    Where an operator makes codes (local): drop their codes whose time is up (they never pair), and
    keep the live ones (the console's App Review code among them). Where codes are a person's own
    (enterprise): ``DeviceConflict(device_limit_reached)`` while they hold
    ``MAX_ACTIVE_DEVICES_PER_PERSON`` active devices here, then the new code replaces every pending
    code of theirs, expired or not.
    """
    if not codes_are_a_persons_own():
        _pending_codes(db, workspace_id, user_pk).filter(
            MobileDevice.pairing_expires_at <= now,
        ).delete(synchronize_session=False)
        return
    from services.mobile.devices import assert_room_for_a_device  # devices imports this module

    assert_room_for_a_device(db, workspace_id, user_pk)
    _pending_codes(db, workspace_id, user_pk).delete(synchronize_session=False)


def pair_link(api: str, code: str) -> str:
    """The QR payload: ``{web_url}/go/pair?api=<url-encoded api>&code=<code>``."""
    return PAIR_LINK_TEMPLATE.format(web_url=web_url(), query=urlencode({"api": api, "code": code}))


def create_pairing_code(
    db: Session, workspace_id: Any, user_pk: int, *, request_base_url: str,
    name: Optional[str] = None, ttl_seconds: Optional[int] = None,
) -> PairingCode:
    """Add a PENDING paired row for this workspace and person and return its code once.

    ``request_base_url`` is the base URL the caller reached the API on; the pair link names
    it only when a local install has no ``PUBLIC_API_HOST`` of its own (K15); in enterprise the
    link names the install's configured public address (``urls.api_url``). The link is built
    before the row, so a misconfigured address (``MobileUrlError``) leaves nothing behind.
    ``name`` (the console's ``--name``) labels the device until it pairs with a name of its own.
    ``ttl_seconds`` overrides ``PAIRING_CODE_TTL_SECONDS`` for the console's App Review code
    (``--review --ttl-hours N``, D13) and nothing else: no route, request field or setting
    passes it, so Studio's codes always last ten minutes.
    """
    _require_device_pairing()
    code = new_pairing_code()
    qr_payload = pair_link(api_url(request_base_url), code)
    now = _now()
    lifetime = PAIRING_CODE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    expires_at = now + timedelta(seconds=lifetime)
    _make_room_for_a_code(db, _as_uuid(workspace_id), user_pk, now)
    db.add(MobileDevice(
        workspace_id=_as_uuid(workspace_id),
        user_id=user_pk,
        kind=MobileDeviceKind.PAIRED.value,
        status=MobileDeviceStatus.PENDING.value,
        name=name,
        pairing_code_hash=hash_secret(normalize_pairing_code(code)),
        pairing_expires_at=expires_at,
        events=[],
    ))
    db.flush()
    return PairingCode(code=code, expires_at=expires_at, qr_payload=qr_payload)


def _claimable(db: Session, code: str, now: datetime) -> Optional[MobileDevice]:
    """The pending paired row this code opens, locked for this pairing, or ``None``."""
    norm = normalize_pairing_code(code)
    if not norm:
        return None
    row = (
        db.query(MobileDevice)
        .filter(
            MobileDevice.pairing_code_hash == hash_secret(norm),
            MobileDevice.status == MobileDeviceStatus.PENDING.value,
            MobileDevice.kind == MobileDeviceKind.PAIRED.value,
        )
        .with_for_update()
        .first()
    )
    if row is None or not secrets_match(row.pairing_code_hash, norm):
        return None
    return row if _unexpired(row.pairing_expires_at, now) else None


def _same_phone(db: Session, pending: MobileDevice, install_id: str) -> Optional[MobileDevice]:
    """The row this phone already has in the pending row's workspace for its person, if any."""
    return (
        db.query(MobileDevice)
        .filter(
            MobileDevice.workspace_id == pending.workspace_id,
            MobileDevice.user_id == pending.user_id,
            MobileDevice.install_id == install_id,
            MobileDevice.id != pending.id,
        )
        .with_for_update()
        .first()
    )


@dataclass(frozen=True)
class Enrolment:
    """The phone a claimed code becomes: its kind, its install id and how it describes itself."""

    kind: str
    install_id: str
    name: Optional[str]
    platform: str
    app_version: Optional[str]


def _paired_fields(device: MobileDevice, enrolment: Enrolment, *, token_hash: str, now: datetime) -> Dict[str, Any]:
    """Every column enrolling writes, as a new mapping: active, the new token only, no code or push key."""
    return {
        "kind": enrolment.kind,
        "status": MobileDeviceStatus.ACTIVE.value,
        "install_id": enrolment.install_id,
        "name": enrolment.name or device.name,
        "platform": enrolment.platform,
        "app_version": enrolment.app_version or device.app_version,
        "token_hash": token_hash,
        "token_rotated_at": None,
        "prev_token_hash": None,
        "prev_token_valid_until": None,
        "push_key": None,
        "push_disabled_reason": None,
        "pairing_code_hash": None,
        "pairing_expires_at": None,
        "sign_in_challenge": None,
        "sign_in_install_id": None,
        "paired_at": now,
        "last_seen_at": now,
        "revoked_at": None,
        "revoked_by": None,
        "events": with_event(device.events, ENROLLED, now),
    }


def pair_device(
    db: Session, code: str, name: Optional[str], platform: str,
    app_version: Optional[str], install_id: str,
) -> Optional[PairedDevice]:
    """Exchange a valid, unexpired, unused code for a device token (returned once), or ``None``.

    ``rotate_after`` is ``paired_at + MOBILE_TOKEN_ROTATE_DAYS``. A phone already paired to
    this workspace and person keeps its row, and the pending row is deleted (module docstring).
    """
    _require_device_pairing()
    now = _now()
    pending = _claimable(db, code, now)
    if pending is None:
        return None
    enrolment = Enrolment(kind=MobileDeviceKind.PAIRED.value, install_id=install_id, name=name,
                          platform=platform, app_version=app_version)
    return enrol(db, pending, enrolment, now)


def enrol(db: Session, pending: MobileDevice, enrolment: Enrolment, now: datetime) -> PairedDevice:
    """Make the claimed ``pending`` row a device, holding a new token (returned once).

    A phone already known to the pending row's workspace and person (``enrolment.install_id``) keeps
    its row, which takes the new token and ``enrolment.kind``, and the pending row is deleted (module
    docstring). Signing in on the phone (PRD-277 US-603) enrols a code the same way.
    """
    device = _same_phone(db, pending, enrolment.install_id) or pending
    token, token_hash = mint_token(DEVICE_TOKEN_PREFIX, DEVICE_TOKEN_BYTES)
    fields = _paired_fields(device, enrolment, token_hash=token_hash, now=now)
    if device is not pending:
        db.delete(pending)
    for column, value in fields.items():
        setattr(device, column, value)
    db.flush()
    return PairedDevice(
        device_id=device.id,
        workspace_id=device.workspace_id,
        token=token,
        rotate_after=now + timedelta(days=config.MOBILE_TOKEN_ROTATE_DAYS),
    )
