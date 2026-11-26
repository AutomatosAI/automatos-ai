"""PRD-277 US-601: no paired device survives the install starting as another edition, or with pairing off.

A device paired in one edition carries that edition's meaning: in ``local`` a phone may act as the
operator, in ``enterprise`` it is one person with their role. So when the install starts as other
than it last started as, every paired device is revoked, once, before a device token is accepted.
What it starts as is ``editions.started_as``: ``local``, ``saas``, or in enterprise
``enterprise:pairing_on`` / ``enterprise:pairing_off`` (``ENTERPRISE_DEVICE_PAIRING``). Every change
follows the one rule: switching pairing off revokes every paired device as an edition change does,
and switching it back on brings none back. A device whose person signed in on the phone (kind
``oidc``, US-603) needs no pairing: an edition change revokes it with the paired ones, switching
pairing off or on leaves it as it is (``KINDS_REVOKED``).

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
  goes on. The device lane and the pairing routes ask ``ensure_confirmed`` first, which refuses them
  (503) until the check has passed in this process, so no device token is accepted before it has.
  A check that never ran (NOT_RUN: a boot that stopped before it, a client with no lifespan) is run
  by the first request that asks. A FAILED one is run again at most once every
  ``RETRY_INTERVAL_SECONDS`` per process, by one request at a time; the others are answered 503 at
  once, without a session or a wait on the lock.
* Each revoked device gets one ``device:revoked`` row in its workspace (``device_audit``), by the
  install, with what it started as before and now, and the reason: ``edition_changed`` when the
  edition differs, ``pairing_switched_off`` when only pairing does. Nothing here logs a token, a
  code or a hash.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from core import editions
from core.database.database import SessionLocal
from core.models.mobile_devices import TOKEN_KINDS, MobileDeviceKind
from core.models.system_settings import SystemSetting
from services.mobile import device_audit
from services.mobile.devices import revoke_every_device_of

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
# The least time between two runs of a FAILED check in one process: in between, device requests get 503 at once.
RETRY_INTERVAL_SECONDS = 30.0

_status = NOT_RUN
_last_attempt = 0.0  # time.monotonic() when the check last started; read only while it is FAILED
_attempt_lock = threading.Lock()  # requests run in the threadpool: one of them claims each attempt


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seconds() -> float:
    return time.monotonic()


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


# Which devices each cause revokes (PRD-277 US-603): another edition, every device holding a token;
# only the pairing setting, only paired ones (a person who signed in on the phone needs no pairing).
KINDS_REVOKED = {
    device_audit.REVOKED_EDITION_CHANGED: TOKEN_KINDS,
    device_audit.REVOKED_PAIRING_SWITCHED_OFF: (MobileDeviceKind.PAIRED.value,),
}


def revoke_if_edition_changed(db: Session, now: datetime) -> int:
    """Revoke the devices its cause names (``KINDS_REVOKED``) when the install starts as other than it
    last started as; commits.

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
    reason = _reason(earlier)
    revoked = revoke_every_device_of(db, KINDS_REVOKED[reason], now) if changed else []
    for device in revoked:
        device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=device.workspace_id, actor=None,
                           resource_id=device.id,
                           details={"reason": reason, "person": device.user_id,
                                    "from_edition": earlier, "to_edition": running})
    if stored is None or changed:
        _store_edition(db, running)
    db.commit()
    if revoked:
        logger.warning("Install started as %s after %s: %d devices revoked", running, earlier, len(revoked))
    return len(revoked)


def _run_check() -> None:
    """``revoke_if_edition_changed`` on a session of its own, closed whatever happens (closing rolls back)."""
    db = SessionLocal()
    try:
        revoke_if_edition_changed(db, _now())
    finally:
        db.close()


def _start_attempt() -> None:
    """Mark the check FAILED until it passes, and when this attempt started. Under ``_attempt_lock``."""
    global _status, _last_attempt
    _status, _last_attempt = FAILED, _seconds()


def _claim_attempt() -> bool:
    """Whether this caller runs the check now: it never ran, or it FAILED at least
    ``RETRY_INTERVAL_SECONDS`` ago. Claiming starts the attempt, so concurrent callers do not also run it."""
    with _attempt_lock:
        due = _status == NOT_RUN or (_status == FAILED and _seconds() - _last_attempt >= RETRY_INTERVAL_SECONDS)
        if due:
            _start_attempt()
        return due


def _attempt() -> bool:
    """Run the check of an attempt already started; whether it ran. Never raises."""
    global _status
    try:
        _run_check()
    except Exception:  # logged with its traceback; the check stays FAILED and is retried after the interval
        logger.exception("Edition check could not run: device tokens are refused until it does")
        return False
    _status = CONFIRMED
    return True


def confirm_edition() -> bool:
    """Run the check now (at boot); whether it ran. Never raises, so it never stops a start.

    FAILED until it has run: any error (the database unreachable, the lock not had within
    ``LOCK_WAIT_LIMIT``, anything else) is logged and answered ``False``, and device tokens stay
    refused in this process until a request runs it again (``ensure_confirmed``).
    """
    with _attempt_lock:
        _start_attempt()
    return _attempt()


def needs_confirming() -> bool:
    """True until this process's check has passed: a cheap read, safe on the event loop."""
    return _status != CONFIRMED


def ensure_confirmed() -> None:
    """Nothing once this process's check has passed; until then run it when it is due
    (``_claim_attempt``), and 503 unless it has now passed."""
    if _status == CONFIRMED:
        return
    if _claim_attempt():
        _attempt()
    if _status != CONFIRMED:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=NOT_CONFIRMED)


__all__ = [
    "EARLIER_WHEN_UNMARKED",
    "LOCK_WAIT_LIMIT",
    "MARKER_CATEGORY",
    "MARKER_KEY",
    "RESERVED_SETTING_CATEGORIES",
    "RETRY_INTERVAL_SECONDS",
    "confirm_edition",
    "ensure_confirmed",
    "needs_confirming",
    "revoke_if_edition_changed",
    "stored_edition",
]
