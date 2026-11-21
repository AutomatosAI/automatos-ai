"""PRD-277 US-603 (§5.7, D-8): the one-time code a phone's sign-in ends with, and its exchange.

A phone signs in at the install's own start address (``core.auth.phone_sign_in``); once the callback
has found the person exactly as a browser's sign-in does, ``issue_code`` gives them a code for the
app instead of a session, and the app exchanges it with its PKCE verifier (``exchange_code``,
``POST /api/v1/mobile/sign-in/exchange``) for a device token of kind ``oidc``.

* **The code** is ``SIGN_IN_CODE_BYTES`` random bytes (256 bits), returned once. It is kept as a
  PENDING ``mobile_devices`` row of kind ``oidc``, as a pairing code is: only its SHA-256
  (``pairing_code_hash``), its expiry ``SIGN_IN_CODE_TTL_SECONDS`` on, the person, the workspace,
  the platform, the app's S256 challenge (``sign_in_challenge``) and its install id
  (``sign_in_install_id``). Nothing from the identity provider is on the row or in the answer. A
  pending row is never listed or acted on as a device; revoking the person's devices revokes it too.
  A new code first deletes the person's expired codes and any this same phone left unexchanged.
* **The workspace** is the person's one active membership of a workspace in use, the device lane's
  own rule (``workspace_members``). None is ``NO_WORKSPACE`` and more than one ``SEVERAL_WORKSPACES``:
  a device belongs to exactly one workspace, and the app has no way to choose here. A person at
  ``MAX_ACTIVE_DEVICES_PER_PERSON`` active devices there is ``DEVICE_LIMIT_REACHED``, unless this
  very phone is already one of them (signing in again replaces its token).
* **The exchange** locks the row by the code's digest, so of two exchanges at once the second waits,
  then finds it gone or used. The code is spent by any exchange that finds it: a wrong verifier
  deletes it as an expired code does. Unknown, used, expired and wrong-verifier codes all answer
  ``None``. S256(verifier) is compared with the stored challenge in constant time. A match enrols the
  row as pairing does (``pairing.enrol``): a phone already known to the workspace and person keeps
  its row, and the token is ``amd_`` with today's rotation and revoke.

These functions flush and never commit (the callback and the route commit) and never log, so a
code, a token or a hash never reaches a log.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from core.auth.oidc_pending import code_challenge
from core.auth.pairing import hash_secret, mint_token, secrets_match
from core.auth.phone_sign_in import DEVICE_LIMIT_REACHED, NO_WORKSPACE, SEVERAL_WORKSPACES, PhoneStart
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from services.mobile import devices
from services.mobile.pairing import Enrolment, PairedDevice, enrol

SIGN_IN_CODE_BYTES = 32
# How long the app has to exchange its code: it does so the moment its link opens.
SIGN_IN_CODE_TTL_SECONDS = 60
# A code this module made is 43 URL-safe characters; anything longer is not one.
SIGN_IN_CODE_MAX_LENGTH = 64
# RFC 7636 §4.1: a code verifier is 43 to 128 characters; any other is a wrong verifier.
CODE_VERIFIER_MAX_LENGTH = 128

# The person's active memberships of workspaces in use, at most two: the device lane's rule.
_WORKSPACES = text(
    "SELECT wm.workspace_id FROM workspace_members wm JOIN workspaces w ON w.id = wm.workspace_id "
    "WHERE wm.user_id = :user_pk AND wm.is_active = true AND w.is_active IS NOT FALSE "
    "AND w.deleted_at IS NULL ORDER BY wm.workspace_id LIMIT 2"
)


class NoSignInCode(RuntimeError):
    """The person cannot hold a device from this sign-in; ``code`` is what the app is told."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Exchanged:
    """A code exchanged: the device, and its person and platform (for the audit row)."""

    device: PairedDevice
    user_pk: int
    platform: Optional[str]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _only_workspace(db: Session, user_pk: int) -> UUID:
    """The person's one workspace a device can belong to; ``NoSignInCode`` for none or several."""
    rows = db.execute(_WORKSPACES, {"user_pk": user_pk}).fetchall()
    if not rows:
        raise NoSignInCode(NO_WORKSPACE)
    if len(rows) > 1:
        raise NoSignInCode(SEVERAL_WORKSPACES)
    found = rows[0][0]
    return found if isinstance(found, UUID) else UUID(str(found))


def _signs_in_again(db: Session, workspace_id: UUID, user_pk: int, install_id: str) -> bool:
    """Whether this phone is already one of the person's active devices in this workspace."""
    return db.query(MobileDevice.id).filter(
        MobileDevice.workspace_id == workspace_id, MobileDevice.user_id == user_pk,
        MobileDevice.install_id == install_id, MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
    ).first() is not None


def _clear_old_codes(db: Session, user_pk: int, install_id: str, now: datetime) -> None:
    """Delete the person's pending codes whose time is up, and any this same phone left unexchanged."""
    db.query(MobileDevice).filter(
        MobileDevice.user_id == user_pk,
        MobileDevice.kind == MobileDeviceKind.OIDC.value,
        MobileDevice.status == MobileDeviceStatus.PENDING.value,
        or_(MobileDevice.pairing_expires_at <= now, MobileDevice.sign_in_install_id == install_id),
    ).delete(synchronize_session=False)


def issue_code(db: Session, user_pk: int, phone: PhoneStart, now: Optional[datetime] = None) -> str:
    """A one-time code for ``user_pk``'s phone, returned once; its pending row added (flushed).

    ``NoSignInCode`` when the person is in no workspace, in several, or at the device cap there. The
    person's expired codes, and any earlier one of this phone, are deleted first.
    """
    moment = now or _now()
    workspace_id = _only_workspace(db, user_pk)
    if not _signs_in_again(db, workspace_id, user_pk, phone.install_id):
        try:
            devices.assert_room_for_a_device(db, workspace_id, user_pk)
        except devices.DeviceConflict:
            raise NoSignInCode(DEVICE_LIMIT_REACHED) from None
    _clear_old_codes(db, user_pk, phone.install_id, moment)
    code, code_hash = mint_token("", SIGN_IN_CODE_BYTES)
    db.add(MobileDevice(
        workspace_id=workspace_id, user_id=user_pk, kind=MobileDeviceKind.OIDC.value,
        status=MobileDeviceStatus.PENDING.value, platform=phone.platform, pairing_code_hash=code_hash,
        pairing_expires_at=moment + timedelta(seconds=SIGN_IN_CODE_TTL_SECONDS),
        sign_in_challenge=phone.code_challenge, sign_in_install_id=phone.install_id, events=[],
    ))
    db.flush()
    return code


def _claimed(db: Session, code: str) -> Optional[MobileDevice]:
    """The pending oidc row this code names, locked, or ``None``."""
    if not isinstance(code, str) or not code or len(code) > SIGN_IN_CODE_MAX_LENGTH:
        return None
    row = db.query(MobileDevice).filter(
        MobileDevice.pairing_code_hash == hash_secret(code),
        MobileDevice.kind == MobileDeviceKind.OIDC.value,
        MobileDevice.status == MobileDeviceStatus.PENDING.value,
    ).with_for_update().first()
    return row if row is not None and secrets_match(row.pairing_code_hash, code) else None


def _verifier_fits(row: MobileDevice, verifier: str) -> bool:
    """S256(``verifier``) is the row's challenge, compared in constant time."""
    expected = str(row.sign_in_challenge or "")
    if not expected or not isinstance(verifier, str) or not verifier.isascii() \
            or len(verifier) > CODE_VERIFIER_MAX_LENGTH:
        return False
    return hmac.compare_digest(code_challenge(verifier).encode("ascii"), expected.encode("ascii"))


def exchange_code(db: Session, code: str, verifier: str, described: Tuple[Optional[str], Optional[str]],
                  now: Optional[datetime] = None) -> Optional[Exchanged]:
    """The device a live code and its verifier make (its token returned once), or ``None``.

    Any code found is spent: a wrong verifier or an expired code deletes it. ``described`` is the
    phone's ``(name, app_version)``, as ``POST /pair`` takes them. Flushes, never commits.
    """
    row = _claimed(db, code)
    if row is None:
        return None
    moment = now or _now()
    live = row.pairing_expires_at is not None and moment < _as_utc(row.pairing_expires_at)
    if not live or not _verifier_fits(row, verifier):
        db.delete(row)
        db.flush()
        return None
    name, app_version = described
    enrolment = Enrolment(kind=MobileDeviceKind.OIDC.value, install_id=row.sign_in_install_id, name=name,
                          platform=row.platform, app_version=app_version)
    person, platform = row.user_id, row.platform  # read before enrol, which may delete the pending row
    return Exchanged(device=enrol(db, row, enrolment, moment), user_pk=person, platform=platform)


__all__ = [
    "CODE_VERIFIER_MAX_LENGTH",
    "SIGN_IN_CODE_BYTES",
    "SIGN_IN_CODE_MAX_LENGTH",
    "SIGN_IN_CODE_TTL_SECONDS",
    "Exchanged",
    "NoSignInCode",
    "exchange_code",
    "issue_code",
]
