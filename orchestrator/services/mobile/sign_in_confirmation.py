"""PRD-277 US-603 (§5.7): a phone's sign-in waits for its person's Continue on the install's own page.

The install cannot tell which app started a phone's sign-in (``core.auth.phone_sign_in``), so the
callback makes no code and no device. It records the sign-in as awaiting confirmation
(``await_confirmation``) and binds that record to the person's browser; the web app's page
(``phone_sign_in.CONFIRM_PAGE``) shows what waits (``waiting``) and offers Continue (``confirm``),
which makes the one-time code, and Cancel (``cancel``), which ends it.

* **What waits** is a PENDING ``mobile_devices`` row of kind ``oidc``: the person, their one workspace,
  the app's challenge, install id, app link and platform, and the SHA-256 of a confirmation token
  (``sign_in_confirm_hash``) that lives only in the browser's ``phone_sign_in.CONFIRM_COOKIE``. Its
  ``pairing_code_hash`` is null, so the exchange, which looks a code up by that column alone, can
  never be handed a confirmation token for a device. It lasts ``CONFIRM_TTL_SECONDS``.
* **One per person per workspace.** Recording one deletes every other pending ``oidc`` row of the
  person in that workspace (awaiting or holding a code), and every expired pending ``oidc`` row on the
  install, whoever's it is.
* **Continue** finds the row by the token's digest, locked, while it is live; counts the device cap
  again (``sign_in_codes.hold_room_for_a_device``); then swaps the confirmation hash for a code's
  (``SIGN_IN_CODE_TTL_SECONDS``). A token is used once: after Continue no row carries its digest.
  A cap reached since the callback deletes the row and is ``NoSignInCode(DEVICE_LIMIT_REACHED)``.
* **Removed before.** ``Waiting.removed_before`` is true when this person's row for this phone in this
  workspace was revoked by someone other than them (an admin, or the install itself): continuing adds
  the phone again, and the page says so.

These functions flush and never commit (the callback and the routes commit) and never log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from core.auth.pairing import hash_secret, mint_token, secrets_match
from core.auth.phone_sign_in import CONFIRM_TTL_SECONDS, PhoneStart
from core.auth.user_link import PENDING_EMAIL_DOMAIN
from core.models.core import User
from core.models.mobile_devices import MobileDevice, MobileDeviceKind, MobileDeviceStatus
from services.mobile.sign_in_codes import (
    SIGN_IN_CODE_BYTES,
    SIGN_IN_CODE_TTL_SECONDS,
    NoSignInCode,
    as_utc,
    hold_room_for_a_device,
    only_workspace,
)

CONFIRM_TOKEN_BYTES = 32
# A token this module made is 43 URL-safe characters; anything longer is not one.
CONFIRM_TOKEN_MAX_LENGTH = 64


@dataclass(frozen=True)
class Waiting:
    """What the confirmation page may show: who signs in, to which app, on which platform, and
    whether this phone was removed by someone else before."""

    person: Optional[str]
    app_link: str
    platform: Optional[str]
    removed_before: bool


@dataclass(frozen=True)
class Confirmed:
    """A Continue: the one-time code (returned once) and what its audit row names."""

    code: str = field(repr=False)
    app_link: str
    user_pk: int
    workspace_id: UUID
    platform: Optional[str]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pending_oidc():
    return and_(MobileDevice.kind == MobileDeviceKind.OIDC.value,
                MobileDevice.status == MobileDeviceStatus.PENDING.value)


def _clear(db: Session, workspace_id: UUID, user_pk: int, now: datetime) -> None:
    """Delete the person's pending oidc rows in this workspace, and every expired one on the install."""
    db.query(MobileDevice).filter(
        _pending_oidc(),
        or_(and_(MobileDevice.workspace_id == workspace_id, MobileDevice.user_id == user_pk),
            MobileDevice.pairing_expires_at <= now),
    ).delete(synchronize_session=False)


def await_confirmation(db: Session, user_pk: int, phone: PhoneStart, now: Optional[datetime] = None) -> str:
    """Record the person's phone sign-in as awaiting their confirmation; the token for their browser's cookie.

    ``NoSignInCode`` when they are in no workspace, in several, or at the device cap there. Flushes.
    """
    moment = now or _now()
    workspace_id = only_workspace(db, user_pk)
    hold_room_for_a_device(db, workspace_id, user_pk, phone.install_id)
    _clear(db, workspace_id, user_pk, moment)
    token, token_hash = mint_token("", CONFIRM_TOKEN_BYTES)
    db.add(MobileDevice(
        workspace_id=workspace_id, user_id=user_pk, kind=MobileDeviceKind.OIDC.value,
        status=MobileDeviceStatus.PENDING.value, platform=phone.platform, sign_in_confirm_hash=token_hash,
        pairing_expires_at=moment + timedelta(seconds=CONFIRM_TTL_SECONDS), sign_in_app_link=phone.app_link,
        sign_in_challenge=phone.code_challenge, sign_in_install_id=phone.install_id, events=[],
    ))
    db.flush()
    return token


def _awaiting(db: Session, token: Optional[str], *, lock: bool) -> Optional[MobileDevice]:
    """The row awaiting confirmation that ``token`` names, or ``None``; locked when ``lock``."""
    if not isinstance(token, str) or not token or len(token) > CONFIRM_TOKEN_MAX_LENGTH:
        return None
    query = db.query(MobileDevice).filter(MobileDevice.sign_in_confirm_hash == hash_secret(token), _pending_oidc())
    row = (query.with_for_update() if lock else query).first()
    return row if row is not None and secrets_match(row.sign_in_confirm_hash, token) else None


def _live(row: MobileDevice, now: datetime) -> bool:
    return row.pairing_expires_at is not None and now < as_utc(row.pairing_expires_at)


def _shown_person(db: Session, user_pk: int) -> Optional[str]:
    """The person's e-mail address, else their name; never the placeholder of an unverified address."""
    person = db.get(User, user_pk)
    email = (person.email or "") if person is not None else ""
    if email and not email.lower().endswith(PENDING_EMAIL_DOMAIN):
        return email
    return (person.name or None) if person is not None else None


def _removed_by_someone_else(db: Session, row: MobileDevice) -> bool:
    return db.query(MobileDevice.id).filter(
        MobileDevice.workspace_id == row.workspace_id, MobileDevice.user_id == row.user_id,
        MobileDevice.install_id == row.sign_in_install_id,
        MobileDevice.status == MobileDeviceStatus.REVOKED.value,
        or_(MobileDevice.revoked_by.is_(None), MobileDevice.revoked_by != row.user_id),
    ).first() is not None


def waiting(db: Session, token: Optional[str], now: Optional[datetime] = None) -> Optional[Waiting]:
    """What awaits confirmation in the browser holding ``token``, or ``None`` (none, expired or used)."""
    row = _awaiting(db, token, lock=False)
    if row is None or not _live(row, now or _now()):
        return None
    return Waiting(person=_shown_person(db, row.user_id), app_link=row.sign_in_app_link, platform=row.platform,
                   removed_before=_removed_by_someone_else(db, row))


def confirm(db: Session, token: Optional[str], now: Optional[datetime] = None) -> Optional[Confirmed]:
    """Continue: the one-time code for what ``token`` names, or ``None`` when nothing live awaits.

    An expired row is deleted. A cap reached since the callback deletes the row and raises
    ``NoSignInCode(DEVICE_LIMIT_REACHED)``. Flushes, never commits.
    """
    row = _awaiting(db, token, lock=True)
    if row is None:
        return None
    moment = now or _now()
    if not _live(row, moment):
        db.delete(row)
        db.flush()
        return None
    try:
        hold_room_for_a_device(db, row.workspace_id, row.user_id, row.sign_in_install_id)
    except NoSignInCode:
        db.delete(row)
        db.flush()
        raise
    code, code_hash = mint_token("", SIGN_IN_CODE_BYTES)
    row.sign_in_confirm_hash = None
    row.pairing_code_hash = code_hash
    row.pairing_expires_at = moment + timedelta(seconds=SIGN_IN_CODE_TTL_SECONDS)
    db.flush()
    return Confirmed(code=code, app_link=row.sign_in_app_link, user_pk=row.user_id,
                     workspace_id=row.workspace_id, platform=row.platform)


def cancel(db: Session, token: Optional[str]) -> bool:
    """Cancel: delete what ``token`` names; whether anything awaited. Flushes, never commits."""
    row = _awaiting(db, token, lock=True)
    if row is None:
        return False
    db.delete(row)
    db.flush()
    return True


__all__ = [
    "CONFIRM_TOKEN_BYTES",
    "CONFIRM_TOKEN_MAX_LENGTH",
    "Confirmed",
    "Waiting",
    "await_confirmation",
    "cancel",
    "confirm",
    "waiting",
]
