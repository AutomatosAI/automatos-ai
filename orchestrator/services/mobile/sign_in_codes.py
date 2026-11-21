"""PRD-277 US-603 (§5.7, D-8): the one-time code a phone's sign-in ends with, and its exchange.

A phone signs in at the install's own start address (``core.auth.phone_sign_in``). Once the callback
has found the person exactly as a browser's sign-in does, the sign-in waits for them to confirm it on
the install's own page (``services.mobile.sign_in_confirmation``), and only their Continue makes the
code. The app exchanges it with its PKCE verifier (``exchange_code``,
``POST /api/v1/mobile/sign-in/exchange``) for a device token of kind ``oidc``.

* **The code** is ``SIGN_IN_CODE_BYTES`` random bytes (256 bits), returned once. It is kept on the
  PENDING ``mobile_devices`` row of kind ``oidc`` the confirmation left, as a pairing code is: only
  its SHA-256 (``pairing_code_hash``) and its expiry ``SIGN_IN_CODE_TTL_SECONDS`` on, beside the
  person, the workspace, the platform, the app's S256 challenge (``sign_in_challenge``), its install
  id (``sign_in_install_id``) and its app link (``sign_in_app_link``). Nothing from the identity
  provider is on the row or in the answer. A pending row is never listed or acted on as a device.
* **The workspace** is the person's one active membership of a workspace in use, the device lane's
  own rule (``workspace_members``), paused workspaces left out. None is ``NO_WORKSPACE`` and more
  than one ``SEVERAL_WORKSPACES``: a device belongs to exactly one workspace, and the app has no way
  to choose here.
* **The device cap** (``hold_room_for_a_device``): a person at ``MAX_ACTIVE_DEVICES_PER_PERSON``
  active devices there is ``DEVICE_LIMIT_REACHED``, unless this very phone is already one of them
  (signing in again replaces its token). It is counted under the person's own transaction-scoped
  advisory lock (``pg_advisory_xact_lock(DEVICE_CAP_LOCK_SPACE, users.id)``), so two exchanges of the
  same person at once count each other: the second waits for the first's commit, then counts its
  device. It is checked at Continue and again inside the exchange's transaction before a device is
  added; a cap reached in between spends the code as a wrong verifier does.
* **The exchange** locks the row by the code's digest, so of two exchanges of one code the second
  waits, then finds it gone or used. The code is spent by any exchange that finds it: a wrong verifier
  deletes it as an expired code does. Unknown, used, expired and wrong-verifier codes all answer
  ``None``. S256(verifier) is compared with the stored challenge in constant time. A match enrols the
  row as pairing does (``pairing.enrol``): a phone already known to the workspace and person keeps
  its row, which becomes ``oidc`` if it was ``paired`` (as pairing makes an ``oidc`` row ``paired``),
  and the token is ``amd_`` with today's rotation and revoke. ``Exchanged.re_enrolment`` says when it
  kept such a row, and who had revoked it, for the ``device:paired`` audit row.

These functions flush and never commit (the routes commit) and never log, so a code, a token or a
hash never reaches a log.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth.oidc_pending import code_challenge
from core.auth.pairing import hash_secret, secrets_match
from core.auth.phone_sign_in import DEVICE_LIMIT_REACHED, NO_WORKSPACE, SEVERAL_WORKSPACES
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
# The first key of the person's device-cap advisory lock ('phon'); the second is users.id.
DEVICE_CAP_LOCK_SPACE = 0x70686F6E

# The person's active memberships of workspaces in use (not deleted, not paused), at most two.
_WORKSPACES = text(
    "SELECT wm.workspace_id FROM workspace_members wm JOIN workspaces w ON w.id = wm.workspace_id "
    "WHERE wm.user_id = :user_pk AND wm.is_active = true AND w.is_active IS NOT FALSE "
    "AND w.deleted_at IS NULL AND w.paused_at IS NULL ORDER BY wm.workspace_id LIMIT 2"
)
_DEVICE_CAP_LOCK = text("SELECT pg_advisory_xact_lock(:space, :user_pk)")


class NoSignInCode(RuntimeError):
    """The person cannot hold a device from this sign-in; ``code`` is the refusal, ``user_pk`` its person."""

    def __init__(self, code: str, user_pk: Optional[int] = None) -> None:
        super().__init__(code)
        self.code = code
        self.user_pk = user_pk


@dataclass(frozen=True)
class Exchanged:
    """A code exchanged: the device, its person and platform, and what it re-enrolled (for the audit row)."""

    device: PairedDevice
    user_pk: int
    platform: Optional[str]
    re_enrolment: Mapping[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """``value`` with UTC as its zone when it carries none."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def only_workspace(db: Session, user_pk: int) -> UUID:
    """The person's one workspace a device can belong to; ``NoSignInCode`` for none or several."""
    rows = db.execute(_WORKSPACES, {"user_pk": user_pk}).fetchall()
    if not rows:
        raise NoSignInCode(NO_WORKSPACE, user_pk)
    if len(rows) > 1:
        raise NoSignInCode(SEVERAL_WORKSPACES, user_pk)
    found = rows[0][0]
    return found if isinstance(found, UUID) else UUID(str(found))


def _signs_in_again(db: Session, workspace_id: UUID, user_pk: int, install_id: Optional[str]) -> bool:
    """Whether this phone is already one of the person's active devices in this workspace."""
    return db.query(MobileDevice.id).filter(
        MobileDevice.workspace_id == workspace_id, MobileDevice.user_id == user_pk,
        MobileDevice.install_id == install_id, MobileDevice.status == MobileDeviceStatus.ACTIVE.value,
    ).first() is not None


def hold_room_for_a_device(db: Session, workspace_id: UUID, user_pk: int, install_id: Optional[str]) -> None:
    """Take the person's device-cap lock for this transaction, then ``NoSignInCode(DEVICE_LIMIT_REACHED)``
    when they hold the most active devices here and this phone is not already one of them."""
    db.execute(_DEVICE_CAP_LOCK, {"space": DEVICE_CAP_LOCK_SPACE, "user_pk": user_pk})
    if _signs_in_again(db, workspace_id, user_pk, install_id):
        return
    try:
        devices.assert_room_for_a_device(db, workspace_id, user_pk)
    except devices.DeviceConflict:
        raise NoSignInCode(DEVICE_LIMIT_REACHED, user_pk) from None


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


def _re_enrolment(db: Session, row: MobileDevice) -> Mapping[str, Any]:
    """What the row this phone already has here was, when it has one: for the ``device:paired`` row."""
    earlier = db.query(MobileDevice.kind, MobileDevice.status, MobileDevice.revoked_by).filter(
        MobileDevice.workspace_id == row.workspace_id, MobileDevice.user_id == row.user_id,
        MobileDevice.install_id == row.sign_in_install_id, MobileDevice.id != row.id,
    ).first()
    if earlier is None:
        return {}
    kind, status, revoked_by = earlier
    found = {"re_enrolled": True, "was_kind": kind, "was_status": status}
    if status == MobileDeviceStatus.REVOKED.value:
        return {**found, "revoked_by": revoked_by, "revoked_by_self": revoked_by == row.user_id}
    return found


def _spent(db: Session, row: MobileDevice) -> None:
    db.delete(row)
    db.flush()


def exchange_code(db: Session, code: str, verifier: str, described: Tuple[Optional[str], Optional[str]],
                  now: Optional[datetime] = None) -> Optional[Exchanged]:
    """The device a live code and its verifier make (its token returned once), or ``None``.

    Any code found is spent: a wrong verifier, an expired code or a cap reached since Continue
    deletes it. ``described`` is the phone's ``(name, app_version)``, as ``POST /pair`` takes them.
    Flushes, never commits.
    """
    row = _claimed(db, code)
    if row is None:
        return None
    moment = now or _now()
    live = row.pairing_expires_at is not None and moment < as_utc(row.pairing_expires_at)
    if not live or not _verifier_fits(row, verifier):
        _spent(db, row)
        return None
    try:
        hold_room_for_a_device(db, row.workspace_id, row.user_id, row.sign_in_install_id)
    except NoSignInCode:
        _spent(db, row)
        return None
    name, app_version = described
    enrolment = Enrolment(kind=MobileDeviceKind.OIDC.value, install_id=row.sign_in_install_id, name=name,
                          platform=row.platform, app_version=app_version)
    # Read before enrol, which may delete the pending row.
    person, platform, re_enrolment = row.user_id, row.platform, _re_enrolment(db, row)
    return Exchanged(device=enrol(db, row, enrolment, moment), user_pk=person, platform=platform,
                     re_enrolment=re_enrolment)


__all__ = [
    "CODE_VERIFIER_MAX_LENGTH",
    "DEVICE_CAP_LOCK_SPACE",
    "SIGN_IN_CODE_BYTES",
    "SIGN_IN_CODE_MAX_LENGTH",
    "SIGN_IN_CODE_TTL_SECONDS",
    "Exchanged",
    "NoSignInCode",
    "as_utc",
    "exchange_code",
    "hold_room_for_a_device",
    "only_workspace",
]
