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

  Push (W2 US-009): a registration that carries an Expo token turns push on and clears the
  reason it was off (the person granted the OS permission); one without a token leaves push
  as it was. A token another row holds moves here only from a row of the same install or the
  same person, in any workspace (a phone that changed workspace or was reinstalled, W2 K14):
  that row loses it (``token_moved``, a ``push_token_moved`` event). From anyone else's row
  it is ``push_token_in_use`` and nothing changes, so one account never takes another's
  phone; that is decided here, before any write, and the unique index only backs it up
  against a race. The phone's name (its model, from the app) replaces the stored one; a
  blank name is not written.
* ``list_devices``: the caller's own devices; a workspace owner/admin (or the local
  operator) sees the workspace's. A pending row is a code waiting for a phone, not a
  device: it is never listed or acted on here. Each is its ``to_dict()``; a row that is
  not the caller's own adds ``owner_label`` (its person's name, else their email: W2
  US-019b), so Studio shows an owner or admin whose phone it is. The caller's own rows
  carry none, and a member only ever sees their own.
* ``update_device``: the device's person only, an active device only, and only
  ``push_enabled``, ``digest_time`` and ``timezone``. Push on needs a registered Expo token
  (``push_token_missing``); push off keeps the token.
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
from core.models.core import User
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from services.mobile.events import ENROLLED, PUSH_TOKEN_MOVED, REVOKED, ROTATED, with_event
from services.mobile.pairing import DEVICE_TOKEN_BYTES

PUSH_KEY_BYTES = 32
# The columns POST /devices writes from its body (install_id names the row, never changes it).
REGISTER_COLUMNS = (
    "platform", "app_version", "locale", "timezone", "expo_push_token", "digest_time", "name",
)
UPDATE_COLUMNS = ("push_enabled", "digest_time", "timezone")
# DeviceConflict codes, answered as 409 {"code": …}.
INSTALL_MISMATCH = "install_id_mismatch"
INSTALL_PAIRED = "install_id_paired"
DEVICE_REVOKED = "device_revoked"
TOKEN_SUPERSEDED = "token_superseded"
# Another person's phone holds this Expo token (K14).
PUSH_TOKEN_IN_USE = "push_token_in_use"
# Push on asked of a device that registered no Expo token.
PUSH_TOKEN_MISSING = "push_token_missing"
# What push_disabled_reason says once a device is revoked, or its token moved to a newer row.
PUSH_OFF_REVOKED = "device_revoked"
PUSH_OFF_TOKEN_MOVED = "token_moved"


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


def _description(body: Mapping[str, Any]) -> Dict[str, Any]:
    """The columns ``body`` carries; the name stripped, and not written at all when blank."""
    name = str(body.get("name") or "").strip()
    described = {column: body[column] for column in REGISTER_COLUMNS if column in body and column != "name"}
    return {**described, **({"name": name} if name else {})}


def _push_on(described: Mapping[str, Any]) -> Dict[str, Any]:
    """A registration carrying an Expo token turns push on (the OS permission was granted)."""
    if not described.get("expo_push_token"):
        return {}
    return {"push_enabled": True, "push_disabled_reason": None}


def _token_holder(db: Session, token: str, device: Optional[MobileDevice]) -> Optional[MobileDevice]:
    """The OTHER row holding this Expo token, in any workspace, locked; ``None`` when there is none."""
    query = db.query(MobileDevice).filter(MobileDevice.expo_push_token == token)
    if device is not None and device.id is not None:
        query = query.filter(MobileDevice.id != device.id)
    return query.with_for_update().first()


def _take_token(db: Session, caller: DeviceCaller, install_id: str, token: Optional[str],
                device: Optional[MobileDevice], now: datetime) -> None:
    """K14: move the Expo token here from a row of the same install or person, before any write.

    Any other holder is ``DeviceConflict(push_token_in_use)``. The holder is flushed before the
    registering row takes the token, so the unique index never sees it twice.
    """
    holder = _token_holder(db, token, device) if token else None
    if holder is None:
        return
    if holder.install_id != install_id and holder.user_id != caller.user_pk:
        raise DeviceConflict(PUSH_TOKEN_IN_USE)
    _assign(holder, {
        "expo_push_token": None,
        "push_enabled": False,
        "push_disabled_reason": PUSH_OFF_TOKEN_MOVED,
        "events": with_event(holder.events, PUSH_TOKEN_MOVED, now),
    })
    db.flush()


def _register_own(db: Session, caller: DeviceCaller, install_id: str,
                  described: Mapping[str, Any]) -> Optional[Registration]:
    """A device-lane caller: its own row takes the description; a push key once."""
    device = _owned(db, caller, caller.device_id)
    if device is None:
        return None
    if device.install_id != install_id:
        raise DeviceConflict(INSTALL_MISMATCH)
    _take_token(db, caller, install_id, described.get("expo_push_token"), device, _now())
    minted = {"push_key": _new_push_key()} if device.push_key is None else {}
    _assign(device, {**described, **_push_on(described), **minted})
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
    _take_token(db, caller, install_id, described.get("expo_push_token"), device, now)
    if device is None:
        device = MobileDevice(workspace_id=caller.workspace_id, user_id=caller.user_pk,
                              kind=MobileDeviceKind.CLERK.value, install_id=install_id, events=[])
        db.add(device)
    # A new row has no status until it is flushed; a revoked one registers afresh.
    enrolling = device.status != MobileDeviceStatus.ACTIVE.value
    fields = {**described, **_push_on(described), "last_seen_at": now,
              **(_enrolment(device, now) if enrolling else {})}
    _assign(device, fields)
    db.flush()
    return Registration(device_id=device.id, push_key=fields.get("push_key"))


def register_device(db: Session, caller: DeviceCaller, install_id: str,
                    body: Mapping[str, Any]) -> Optional[Registration]:
    """Register or update the calling phone from ``body`` (only the keys it carries are written).

    ``None`` when a device-lane caller's row is gone or revoked; ``DeviceConflict`` when its
    token belongs to another install, a Clerk caller's install is paired here or was
    revoked by someone else, or another person's phone holds the Expo token.
    """
    described = _description(body)
    if caller.device_id is not None:
        return _register_own(db, caller, install_id, described)
    return _register_clerk(db, caller, install_id, described)


def _owner_labels(db: Session, caller: DeviceCaller, rows: List[MobileDevice]) -> Dict[int, str]:
    """``users.id`` → name (else email) of each person, other than the caller, whose device is listed."""
    others = {row.user_id for row in rows if row.user_id != caller.user_pk}
    if not others:
        return {}
    people = db.query(User.id, User.name, User.email).filter(User.id.in_(others)).all()
    return {pk: (name or "").strip() or email for pk, name, email in people}


def _view(device: MobileDevice, labels: Mapping[int, str]) -> Dict[str, Any]:
    """The device as the API shows it, with ``owner_label`` when it is someone else's."""
    shown = device.to_dict()
    label = labels.get(device.user_id)
    return {**shown, "owner_label": label} if label else shown


def list_devices(db: Session, caller: DeviceCaller) -> List[Dict[str, Any]]:
    """The caller's own devices, or the workspace's for an owner/admin, oldest first, as views.

    Only the listed rows' people are read, so a label never names anyone outside this list.
    """
    query = _devices(db, caller)
    if not caller.sees_workspace:
        query = query.filter(MobileDevice.user_id == caller.user_pk)
    rows = query.order_by(MobileDevice.created_at, MobileDevice.id).all()
    labels = _owner_labels(db, caller, rows)
    return [_view(row, labels) for row in rows]


def update_device(db: Session, caller: DeviceCaller, device_id: UUID,
                  changes: Mapping[str, Any]) -> Optional[MobileDevice]:
    """Change the push switch, digest time or time zone of one of the caller's person's active devices.

    ``DeviceConflict(push_token_missing)`` for push on when the device registered no Expo token.
    """
    device = _owned(db, caller, device_id)
    if device is None:
        return None
    if changes.get("push_enabled") is True and not device.expo_push_token:
        raise DeviceConflict(PUSH_TOKEN_MISSING)
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
