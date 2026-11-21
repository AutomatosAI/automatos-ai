"""PRD-277 US-601: in an edition whose device tokens age out, a device token has a maximum age.

Where the edition says ``device_tokens_age_out`` (enterprise), a device's token is refused once
``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` days have passed since its person last proved who they are to
it: the phone's last pairing with a code, or its last sign-in on the phone. Both are an enrolment
(``services.mobile.pairing.enrol``), which stamps ``paired_at``. A rotation does not move it: a
rotation is the device trading its own token for a new one, and proves nothing about its person, so a
phone that rotates still pairs again or signs in again within the maximum age. A row with no
``paired_at`` counts from when it was created. Both kinds that hold a token (``paired``, ``oidc``)
age alike. ``local`` and ``saas`` never call this module and never read the setting.

The device lane (``core.auth.device``) asks ``is_past_max_age`` on every request. The first request
that finds a device past it revokes the device by the install (``devices.revoke_if_still_active``) and
records one ``device:revoked`` row, reason ``token_max_age_reached``, in the same commit; a request
that finds it already revoked records nothing. Nothing here logs a token or a hash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.models.mobile_devices import MobileDevice
from services.mobile import device_audit
from services.mobile.devices import revoke_if_still_active

logger = logging.getLogger(__name__)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def max_age() -> timedelta:
    """``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` as a span; start-up has checked it is 1 to 3650 days."""
    return timedelta(days=config.ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS)


def issued_at(device: MobileDevice) -> datetime:
    """When the device's age starts: its last pairing or sign-in on the phone, else its row's creation."""
    return _as_utc(device.paired_at or device.created_at)


def is_past_max_age(device: MobileDevice, now: datetime) -> bool:
    """True from the moment the device's age reaches ``max_age``."""
    return now >= issued_at(device) + max_age()


def revoke_aged_out(db: Session, device_id: UUID, now: datetime, *, ip_address: Optional[str],
                    user_agent: Optional[str]) -> bool:
    """Revoke the device and record why, committed together; whether this call revoked it.

    ``False`` when it was no longer active (nothing is recorded twice), or when the commit failed (rolled
    back and logged by type, so the next request tries again).
    """
    device = revoke_if_still_active(db, device_id, now)
    if device is None:
        return False
    device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=device.workspace_id, actor=None,
                       resource_id=device.id, ip_address=ip_address, user_agent=user_agent,
                       details={"reason": device_audit.REVOKED_TOKEN_MAX_AGE, "person": device.user_id,
                                "max_age_days": config.ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS})
    try:
        db.commit()
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("Device %s past its maximum age could not be revoked: %s", device_id, type(failure).__name__)
        return False
    return True


__all__ = ["is_past_max_age", "issued_at", "max_age", "revoke_aged_out"]
