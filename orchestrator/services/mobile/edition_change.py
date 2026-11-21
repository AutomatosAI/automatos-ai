"""PRD-277 US-601: no paired device survives the install starting as another edition, or with pairing off.

A device paired in one edition carries that edition's meaning: in ``local`` a phone may act as the
operator, in ``enterprise`` it is one person with their role. So when the install starts as other
than it last started as, every paired device is revoked, once, before a device token is accepted.
What it starts as is ``editions.started_as``: ``local``, ``saas``, or in enterprise
``enterprise:pairing_on`` / ``enterprise:pairing_off`` (``ENTERPRISE_DEVICE_PAIRING``). Every change
follows the one rule: switching pairing off revokes every paired device as an edition change does,
and switching it back on brings none back.

* **The marker.** What the install last started as is one ``system_settings`` row in the install's
  own category (``MARKER_CATEGORY``, which the settings API neither lists nor changes; key
  ``edition``), so no migration is needed. An install with no marker last started before this check
  existed, and is taken to have been ``local`` (``EARLIER_WHEN_UNMARKED``), the edition its paired
  devices were made in (``saas`` never pairs). Its first start as ``local`` revokes nothing and
  writes the marker; its first start as anything else revokes every paired device, the safe side
  whatever it ran before.
* **Once, under a lock.** ``revoke_if_edition_changed`` waits at most ``LOCK_WAIT_LIMIT`` for a
  transaction-scoped advisory lock, reads the marker, revokes and records when it differs, writes
  the marker and commits. Two API processes starting together take turns: the second finds the
  marker already moved and revokes nothing. An install that never changes only ever reads the marker.
* **Before any token, never in the way of a start.** Every API process runs ``confirm_edition`` at
  boot (``main.py`` ``_boot_phase_1_core``, right after ``create_tables``, even when that fails),
  before the server listens. It never raises: a check that cannot run (the database unreachable,
  the lock not had in time, any other error) is logged and leaves the check FAILED, and the start
  goes on. The device lane and the pairing routes ask ``ensure_confirmed`` first, which runs the
  check again and refuses them (503) while it cannot run, so no device token is accepted in this
  process before the check has passed. A process whose boot never ran it (a test client with no
  lifespan) is not held up.
* Each revoked device gets one ``device:revoked`` row in its workspace (``device_audit``), by the
  install, with what it started as before and now, and the reason: ``edition_changed`` when the
  edition differs, ``pairing_switched_off`` when only pairing does. Nothing here logs a token, a
  code or a hash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from core import editions
from core.database.database import SessionLocal
from core.models.system_settings import SystemSetting
from services.mobile import device_audit
from services.mobile.devices import revoke_every_paired_device

logger = logging.getLogger(__name__)

# The install's own category: only this module writes it. The settings API and Auto's settings tools
# neither list nor change a category in RESERVED_SETTING_CATEGORIES (``api/system_settings.py``,
# ``modules/tools/discovery/handlers_workspace.py``).
MARKER_CATEGORY = "install_state"
RESERVED_SETTING_CATEGORIES = (MARKER_CATEGORY,)
MARKER_KEY = "edition"
MARKER_DESCRIPTION = "PRD-277 US-601: what this install last started as (its edition; in enterprise, whether it pairs)"
EARLIER_WHEN_UNMARKED = editions.EDITION_LOCAL
# The longest a start waits for another process's check to finish before it counts as FAILED.
LOCK_WAIT_LIMIT = "5s"
_LOCK_WAIT = text("SELECT set_config('lock_timeout', :limit, true)")
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


def _reason(earlier: str) -> str:
    """Why the devices go: another edition, or the same edition now started with pairing off."""
    if editions.started_as_another_edition(earlier):
        return device_audit.REVOKED_EDITION_CHANGED
    return device_audit.REVOKED_PAIRING_SWITCHED_OFF


def revoke_if_edition_changed(db: Session, now: datetime) -> int:
    """Revoke every paired device when the install starts as other than it last started as; commits.

    What it starts as is ``editions.started_as``: the edition and, in enterprise, whether it pairs.
    Returns how many were revoked. Waits at most ``LOCK_WAIT_LIMIT`` for the advisory lock, then
    holds it until the commit. The comparison is ``editions.started_as_another``, the flag's own module.
    """
    db.execute(_LOCK_WAIT, {"limit": LOCK_WAIT_LIMIT})
    db.execute(_LOCK, {"key": LOCK_KEY})
    stored = stored_edition(db)
    earlier = stored or EARLIER_WHEN_UNMARKED
    running = editions.started_as()
    changed = editions.started_as_another(earlier)
    revoked = revoke_every_paired_device(db, now) if changed else []
    reason = _reason(earlier)
    for device in revoked:
        device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=device.workspace_id, actor=None,
                           resource_id=device.id,
                           details={"reason": reason, "person": device.user_id,
                                    "from_edition": earlier, "to_edition": running})
    if stored is None or changed:
        _store_edition(db, running)
    db.commit()
    if revoked:
        logger.warning("Install started as %s after %s: %d paired devices revoked", running, earlier, len(revoked))
    return len(revoked)


def _run_check() -> None:
    """``revoke_if_edition_changed`` on a session of its own, closed whatever happens (closing rolls back)."""
    db = SessionLocal()
    try:
        revoke_if_edition_changed(db, _now())
    finally:
        db.close()


def confirm_edition() -> bool:
    """Run the check; whether it ran. Never raises, so it never stops a start.

    FAILED until it has run: any error (the database unreachable, the lock not had within
    ``LOCK_WAIT_LIMIT``, anything else) is logged and answered ``False``, and device tokens stay
    refused in this process until a request runs it again (``ensure_confirmed``).
    """
    global _status
    _status = FAILED
    try:
        _run_check()
    except Exception:  # logged with its traceback; the check stays FAILED and the next request retries it
        logger.exception("Edition check could not run: device tokens are refused until it does")
        return False
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
    "LOCK_WAIT_LIMIT",
    "MARKER_CATEGORY",
    "MARKER_KEY",
    "RESERVED_SETTING_CATEGORIES",
    "confirm_edition",
    "ensure_confirmed",
    "needs_confirming",
    "revoke_if_edition_changed",
    "stored_edition",
]
