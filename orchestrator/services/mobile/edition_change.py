"""PRD-277 US-601: no paired device survives the install starting as another edition.

A device paired in one edition carries that edition's meaning: in ``local`` a phone may act as the
operator, in ``enterprise`` it is one person with their role. So when the install starts as an
edition other than the one it last started as, every paired device is revoked, once, before a device
token is accepted. Both directions follow the one rule.

* **The marker.** The edition the install last started as is one ``system_settings`` row (category
  ``boot``, as the seed fingerprints are; key ``edition``), so no migration is needed. An install
  with no marker last started before this check existed, and is taken to have been ``local``
  (``EARLIER_WHEN_UNMARKED``), the edition its paired devices were made in (``saas`` never pairs).
  Its first start as ``local`` revokes nothing and writes the marker; its first start as any other
  edition revokes every paired device, the safe side whatever it ran before.
* **Once, under a lock.** ``revoke_if_edition_changed`` takes a transaction-scoped advisory lock,
  reads the marker, revokes and records when the edition differs, writes the marker and commits.
  Two API processes starting together take turns: the second finds the marker already moved and
  revokes nothing. An install that never changes edition only ever reads the marker.
* **Before any token.** Every API process runs ``confirm_edition`` at boot (``main.py``
  ``_boot_phase_1_core``, right after ``create_tables``, even when that fails), before the server
  listens. A boot that cannot reach the database still starts, as it did before,
  but the check is then FAILED: the device lane and the pairing routes ask ``ensure_confirmed``
  first, which runs the check again and refuses them (503) while it cannot run, so no device token
  is accepted in this process before the check has passed. A process whose boot never ran it (a
  test client with no lifespan) is not held up.
* Each revoked device gets one ``device:revoked`` row in its workspace (``device_audit``), by the
  install, with both editions' names. Nothing here logs a token, a code or a hash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core import editions
from core.database.database import SessionLocal
from core.models.system_settings import SystemSetting
from services.mobile import device_audit
from services.mobile.devices import revoke_every_paired_device

logger = logging.getLogger(__name__)

MARKER_CATEGORY = "boot"
MARKER_KEY = "edition"
MARKER_DESCRIPTION = "PRD-277 US-601: the edition this install last started as"
EARLIER_WHEN_UNMARKED = editions.EDITION_LOCAL
_LOCK = text("SELECT pg_advisory_xact_lock(hashtext(:key))")
LOCK_KEY = "automatos:edition_change"
NOT_RUN, CONFIRMED, FAILED = "not_run", "confirmed", "failed"
NOT_CONFIRMED = "Phones are unavailable right now. Try again in a minute."

_status = NOT_RUN


def _now() -> datetime:
    return datetime.now(timezone.utc)


def stored_edition(db: Session) -> Optional[str]:
    """The edition the install last started as, or ``None`` when it never recorded one."""
    row = db.query(SystemSetting.value).filter(
        SystemSetting.category == MARKER_CATEGORY, SystemSetting.key == MARKER_KEY,
    ).order_by(SystemSetting.id.desc()).first()
    value = str(row[0] or "").strip().lower() if row is not None else ""
    return value or None


def _store_edition(db: Session, edition: str) -> None:
    """Write the marker: every matching row, or a new one when there is none."""
    updated = db.query(SystemSetting).filter(
        SystemSetting.category == MARKER_CATEGORY, SystemSetting.key == MARKER_KEY,
    ).update({SystemSetting.value: edition}, synchronize_session=False)
    if not updated:
        db.add(SystemSetting(category=MARKER_CATEGORY, key=MARKER_KEY, value=edition, value_type="string",
                             description=MARKER_DESCRIPTION, is_sensitive=False, created_by="system"))


def revoke_if_edition_changed(db: Session, now: datetime) -> int:
    """Revoke every paired device when the running edition is not the one the install last started as; commits.

    Returns how many were revoked. Holds the advisory lock until the commit. The comparison is
    ``editions.started_as_another``, the flag's own module.
    """
    db.execute(_LOCK, {"key": LOCK_KEY})
    stored = stored_edition(db)
    earlier = stored or EARLIER_WHEN_UNMARKED
    running = editions.current_name()
    revoked = revoke_every_paired_device(db, now) if editions.started_as_another(earlier) else []
    for device in revoked:
        device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=device.workspace_id, actor=None,
                           resource_id=device.id,
                           details={"reason": device_audit.REVOKED_EDITION_CHANGED, "person": device.user_id,
                                    "from_edition": earlier, "to_edition": running})
    if stored is None or editions.started_as_another(stored):
        _store_edition(db, running)
    db.commit()
    if revoked:
        logger.warning("Edition changed from %s to %s: %d paired devices revoked", earlier, running, len(revoked))
    return len(revoked)


def confirm_edition() -> bool:
    """Run the check on a session of its own; whether it ran.

    FAILED until it has run, so any error (a database one is logged and answered ``False``; any
    other propagates) leaves device tokens refused in this process.
    """
    global _status
    _status = FAILED
    db = SessionLocal()
    try:
        revoke_if_edition_changed(db, _now())
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("Edition check could not run (%s): device tokens are refused until it does",
                     type(failure).__name__)
        return False
    finally:
        db.close()
    _status = CONFIRMED
    return True


def needs_confirming() -> bool:
    """True while this process's check FAILED: a cheap read, safe on the event loop."""
    return _status == FAILED


def ensure_confirmed() -> None:
    """503 while this process's check FAILED and still cannot run; nothing otherwise."""
    if _status == FAILED and not confirm_edition():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=NOT_CONFIRMED)


__all__ = [
    "EARLIER_WHEN_UNMARKED",
    "MARKER_CATEGORY",
    "MARKER_KEY",
    "confirm_edition",
    "ensure_confirmed",
    "needs_confirming",
    "revoke_if_edition_changed",
    "stored_edition",
]
