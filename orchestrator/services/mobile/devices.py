"""PRD-266 §4.3 (K5) — a phone's own row: register, list, update, rotate and revoke.

The router (``api/mobile_devices.py``) resolves the caller into a ``DeviceCaller``: the
workspace, the person's ``users.id``, the device id when the call came through the device
lane, and whether the caller may see the workspace's devices. Everything here acts inside
that one workspace. A row that is not the caller's to act on answers ``None``, which the
router turns into 404, so another workspace's id (or another person's) is never told apart
from an id that does not exist.

* ``register_device``: a device-lane caller updates ITS row (the one its token opened); a
  Clerk caller upserts a ``clerk`` row by (workspace, person, ``install_id``), which never
  holds a token or code hash (§8). The per-device ``push_key`` (W2 signs pushes with it) is
  minted when the row has none (pairing and revoking clear it, so a phone paired again gets
  a new one) or when a clerk row its own person revoked (signing out) registers again, and
  only then is it returned (K14). A clerk row someone else revoked (an admin) stays revoked:
  its person's next sign-in does not undo it (``device_revoked``).
* ``list_devices``: the caller's own devices; a workspace owner/admin (or the local
  operator) sees the workspace's. A pending row is a code waiting for a phone, not a
  device: it is never listed or acted on here.
* ``update_device``: the device's person only, an active device only, and only
  ``push_enabled``, ``digest_time`` and ``timezone``.
* ``revoke_device``: the device's person, or a workspace owner/admin. The token hash is
  KEPT, so the device lane answers the next call ``401 device_revoked``; its push state is
  cleared (switch off, Expo token and push key gone), so nothing is pushed to it and the
  phone's Expo token is free for its next registration. Revoking a revoked device changes
  nothing.
* ``rotate_token``: the device lane on its own device, presenting its CURRENT token. The
  new token is returned once; the old one stays valid for
  ``MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS``. A token already rotated out may not rotate
  again (``token_superseded``): a leaked old token could otherwise trade the last minutes of
  its overlap for a fresh long-lived one.

These functions flush and never commit (the router commits once) and never log, so a
token, a hash or a push key never reaches a log.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from sqlalchemy.orm import Query, Session

from config import config
from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX
from core.auth.pairing import mint_token, secrets_match
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from services.mobile.events import ENROLLED, REVOKED, ROTATED, with_event
from services.mobile.pairing import DEVICE_TOKEN_BYTES

PUSH_KEY_BYTES = 32
# The columns POST /devices writes from its body (install_id names the row, never changes it).
REGISTER_COLUMNS = ("platform", "app_version", "locale", "timezone", "expo_push_token", "digest_time")
UPDATE_COLUMNS = ("push_enabled", "digest_time", "timezone")
# DeviceConflict codes, answered as 409 {"code": …}.
INSTALL_MISMATCH = "install_id_mismatch"
INSTALL_PAIRED = "install_id_paired"
DEVICE_REVOKED = "device_revoked"
TOKEN_SUPERSEDED = "token_superseded"
# What push_disabled_reason says once a device is revoked.
PUSH_OFF_REVOKED = "device_revoked"


class DeviceConflict(RuntimeError):
    """The call contradicts the device's row; ``code`` says how. Raised before anything changes."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DeviceCaller:
    """Who is calling, as the router resolved it from the request."""

    workspace_id: UUID
    user_pk: int
    # The row the caller's device token opened; None for every other lane.
    device_id: Optional[UUID] = None
    # A workspace owner/admin, or the local operator: sees and revokes the workspace's devices.
    sees_workspace: bool = False


@dataclass(frozen=True)
class Registration:
    """A registered device; ``push_key`` is in clear only on the call that minted it."""

    device_id: UUID
    push_key: Optional[str]


@dataclass(frozen=True)
class RotatedToken:
    """A rotated device token, in clear exactly once, and how long the old one lives on."""

    device_id: UUID
    token: str
    rotate_after: datetime
    previous_valid_until: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_push_key() -> str:
    return secrets.token_urlsafe(PUSH_KEY_BYTES)


def _assign(device: MobileDevice, fields: Mapping[str, Any]) -> MobileDevice:
    """Write ``fields`` onto the ORM row (how SQLAlchemy records a change)."""
    for column, value in fields.items():
        setattr(device, column, value)
    return device


def _devices(db: Session, caller: DeviceCaller) -> Query:
    """The caller's workspace's devices: never another workspace's, never a pending code."""
    return db.query(MobileDevice).filter(
        MobileDevice.workspace_id == caller.workspace_id,
        MobileDevice.status != MobileDeviceStatus.PENDING.value,
    )


def _owned(db: Session, caller: DeviceCaller, device_id: Optional[UUID]) -> Optional[MobileDevice]:
    """The caller's person's ACTIVE device with this id, locked for this change, or ``None``."""
    if device_id is None:
        return None
    return (
        _devices(db, caller)
        .filter(
            MobileDevice.id == device_id,
            MobileDevice.user_id == caller.user_pk,
            MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
        )
        .with_for_update()
        .first()
    )


def _register_own(db: Session, caller: DeviceCaller, install_id: str,
                  described: Mapping[str, Any]) -> Optional[Registration]:
    """A device-lane caller: its own row takes the description; a push key once."""
    device = _owned(db, caller, caller.device_id)
    if device is None:
        return None
    if device.install_id != install_id:
        raise DeviceConflict(INSTALL_MISMATCH)
    minted = {"push_key": _new_push_key()} if device.push_key is None else {}
    _assign(device, {**described, **minted})
    db.flush()
    return Registration(device_id=device.id, push_key=minted.get("push_key"))


def _clerk_row(db: Session, caller: DeviceCaller, install_id: str) -> Optional[MobileDevice]:
    """This person's clerk row for this install in this workspace, locked, or ``None``.

    ``DeviceConflict`` when the install is a paired device here (it registers with its token),
    or a clerk row someone other than its person revoked (an admin's revoke stands).
    """
    device = (
        db.query(MobileDevice)
        .filter(
            MobileDevice.workspace_id == caller.workspace_id,
            MobileDevice.user_id == caller.user_pk,
            MobileDevice.install_id == install_id,
        )
        .with_for_update()
        .first()
    )
    if device is not None and device.kind != MobileDeviceKind.CLERK.value:
        raise DeviceConflict(INSTALL_PAIRED)
    if device is not None and device.status == MobileDeviceStatus.REVOKED.value \
            and device.revoked_by != caller.user_pk:
        raise DeviceConflict(DEVICE_REVOKED)
    return device


def _enrolment(device: MobileDevice, now: datetime) -> Dict[str, Any]:
    """What (re-)enrolling a clerk row writes: active, not revoked, a new push key, 'enrolled'."""
    return {
        "status": MobileDeviceStatus.ACTIVE.value,
        "revoked_at": None,
        "revoked_by": None,
        "push_disabled_reason": None,
        "push_key": _new_push_key(),
        "events": with_event(device.events, ENROLLED, now),
    }


def _register_clerk(db: Session, caller: DeviceCaller, install_id: str,
                    described: Mapping[str, Any]) -> Registration:
    """A Clerk caller: upsert its clerk row. No token or code hash is ever written (§8)."""
    now = _now()
    device = _clerk_row(db, caller, install_id)
    if device is None:
        device = MobileDevice(workspace_id=caller.workspace_id, user_id=caller.user_pk,
                              kind=MobileDeviceKind.CLERK.value, install_id=install_id, events=[])
        db.add(device)
    # A new row has no status until it is flushed; a revoked one registers afresh.
    enrolling = device.status != MobileDeviceStatus.ACTIVE.value
    fields = {**described, "last_seen_at": now, **(_enrolment(device, now) if enrolling else {})}
    _assign(device, fields)
    db.flush()
    return Registration(device_id=device.id, push_key=fields.get("push_key"))


def register_device(db: Session, caller: DeviceCaller, install_id: str,
                    body: Mapping[str, Any]) -> Optional[Registration]:
    """Register or update the calling phone from ``body`` (only the keys it carries are written).

    ``None`` when a device-lane caller's row is gone or revoked; ``DeviceConflict`` when its
    token belongs to another install, or a Clerk caller's install is paired here or was
    revoked by someone else.
    """
    described = {column: body[column] for column in REGISTER_COLUMNS if column in body}
    if caller.device_id is not None:
        return _register_own(db, caller, install_id, described)
    return _register_clerk(db, caller, install_id, described)


def list_devices(db: Session, caller: DeviceCaller) -> List[MobileDevice]:
    """The caller's own devices, or the workspace's for an owner/admin, oldest first."""
    query = _devices(db, caller)
    if not caller.sees_workspace:
        query = query.filter(MobileDevice.user_id == caller.user_pk)
    return query.order_by(MobileDevice.created_at, MobileDevice.id).all()


def update_device(db: Session, caller: DeviceCaller, device_id: UUID,
                  changes: Mapping[str, Any]) -> Optional[MobileDevice]:
    """Change the push switch, digest time or time zone of one of the caller's person's active devices."""
    device = _owned(db, caller, device_id)
    if device is None:
        return None
    _assign(device, {column: changes[column] for column in UPDATE_COLUMNS if column in changes})
    db.flush()
    return device


def revoke_device(db: Session, caller: DeviceCaller, device_id: UUID) -> Optional[MobileDevice]:
    """Cut a device off, keeping its token hash so its next call is told ``device_revoked``.

    Its push state goes with it: the switch off, the Expo token and the push key cleared.
    """
    query = _devices(db, caller).filter(MobileDevice.id == device_id)
    if not caller.sees_workspace:
        query = query.filter(MobileDevice.user_id == caller.user_pk)
    device = query.with_for_update().first()
    if device is None or device.status == MobileDeviceStatus.REVOKED.value:
        return device
    now = _now()
    _assign(device, {
        "status": MobileDeviceStatus.REVOKED.value,
        "revoked_at": now,
        "revoked_by": caller.user_pk,
        "push_enabled": False,
        "push_disabled_reason": PUSH_OFF_REVOKED,
        "expo_push_token": None,
        "push_key": None,
        "events": with_event(device.events, REVOKED, now),
    })
    db.flush()
    return device


def rotate_token(db: Session, caller: DeviceCaller, device_id: UUID,
                 presented: Optional[str]) -> Optional[RotatedToken]:
    """A new token for the calling device (returned once); the old one lives on for the overlap.

    ``None`` unless the caller is the device lane on this very device; ``DeviceConflict``
    when ``presented`` is not the device's current token.
    """
    if caller.device_id != device_id:
        return None
    device = _owned(db, caller, device_id)
    if device is None or device.kind != MobileDeviceKind.PAIRED.value:
        return None
    if not secrets_match(device.token_hash, presented):
        raise DeviceConflict(TOKEN_SUPERSEDED)
    now = _now()
    token, token_hash = mint_token(DEVICE_TOKEN_PREFIX, DEVICE_TOKEN_BYTES)
    overlap_until = now + timedelta(seconds=config.MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS)
    _assign(device, {
        "prev_token_hash": device.token_hash,
        "prev_token_valid_until": overlap_until,
        "token_hash": token_hash,
        "token_rotated_at": now,
        "events": with_event(device.events, ROTATED, now),
    })
    db.flush()
    return RotatedToken(
        device_id=device.id,
        token=token,
        rotate_after=now + timedelta(days=config.MOBILE_TOKEN_ROTATE_DAYS),
        previous_valid_until=overlap_until,
    )
