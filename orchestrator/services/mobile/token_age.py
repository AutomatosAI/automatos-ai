"""PRD-277 US-601: in an edition whose device tokens age out, a device token has a maximum age.

Where the edition says ``device_tokens_age_out`` (enterprise), a device's token is refused once
``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` days have passed since its person last proved who they are to
it: the phone's last pairing with a code, or its last sign-in on the phone. Both are an enrolment
(``services.mobile.pairing.enrol``), which stamps ``paired_at``; enrolling again restarts the count. A
rotation does not: a rotation is the device trading its own token for a new one, and proves nothing
about its person. A row with no ``paired_at`` counts from when it was created. A ``paired_at`` later
than now is read as now, so it never makes a device past its age: such a device ages out one
maximum age after that time. Both kinds that hold a token (``paired``, ``oidc``) age alike. ``local`` and
``saas`` never call into the rule and never read the setting.

**One rule, three readers.** A device is past its maximum age when the moment its age starts
(``ISSUED_AT``: ``paired_at``, else ``created_at``) is at or before ``cutoff(now)``. The device lane
asks ``is_past_max_age`` on every request; push fan-out leaves such a device out
(``within_max_age_filters``, the same comparison in SQL); the device list shows it as
``EXPIRED_STATUS`` (``view_status``). Clamping a future ``paired_at`` to now changes no answer, since
the maximum age is at least one day.

**The revoke.** The first request that finds a device past it revokes the device by the install and
records one ``device:revoked`` row, reason ``token_max_age_reached``, in the same commit
(``revoke_aged_out``). The row is locked and read again first: one another request holds is left to
it, and one enrolled again meanwhile is not revoked. Any database error rolls back, is logged by type,
and leaves the device as it was; the request is refused all the same. Nothing here logs a token or a
hash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core import editions
from core.models.mobile_devices import MobileDevice, MobileDeviceStatus
from services.mobile import device_audit
from services.mobile.devices import revoke_active_device_if

logger = logging.getLogger(__name__)

# When a device's age starts, in SQL: its last enrolment, else its row's creation (``issued_at``).
ISSUED_AT = func.coalesce(MobileDevice.paired_at, MobileDevice.created_at)
# What the device list shows for an active device past its maximum age (the row stays active until
# its next request revokes it, or a person does).
EXPIRED_STATUS = "expired"


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def ages_out() -> bool:
    """Whether device tokens have a maximum age here (``device_tokens_age_out``: enterprise)."""
    return editions.current().device_tokens_age_out


def max_age() -> timedelta:
    """``ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`` as a span; start-up has checked it is 1 to 3650 days."""
    return timedelta(days=config.ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS)


def cutoff(now: datetime) -> datetime:
    """A device whose age started at or before this moment is past its maximum age."""
    return now - max_age()


def issued_at(device: MobileDevice, now: datetime) -> datetime:
    """When the device's age starts (``ISSUED_AT``), never later than ``now``."""
    return min(_as_utc(device.paired_at or device.created_at), now)


def is_past_max_age(device: MobileDevice, now: datetime) -> bool:
    """True from the moment the device's age reaches ``max_age``."""
    return issued_at(device, now) <= cutoff(now)


def within_max_age_filters(now: datetime) -> Tuple[Any, ...]:
    """The SQL filters that keep only devices inside their maximum age; none where tokens never age out."""
    return (ISSUED_AT > cutoff(now),) if ages_out() else ()


def view_status(device: MobileDevice, now: datetime) -> str:
    """The device's status as the device list shows it: ``EXPIRED_STATUS`` for an active one past its age."""
    if ages_out() and device.status == MobileDeviceStatus.ACTIVE.value and is_past_max_age(device, now):
        return EXPIRED_STATUS
    return device.status


def _revoke_and_record(db: Session, device_id: UUID, now: datetime, ip_address: Optional[str],
                       user_agent: Optional[str]) -> bool:
    device = revoke_active_device_if(db, device_id, now, due=lambda fresh: is_past_max_age(fresh, now))
    if device is None:
        return False
    device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=device.workspace_id, actor=None,
                       resource_id=device.id, ip_address=ip_address, user_agent=user_agent,
                       details={"reason": device_audit.REVOKED_TOKEN_MAX_AGE, "person": device.user_id,
                                "max_age_days": config.ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS})
    db.commit()
    return True


def revoke_aged_out(db: Session, device_id: UUID, now: datetime, *, ip_address: Optional[str],
                    user_agent: Optional[str]) -> bool:
    """Revoke the device and record why, committed together; whether this call revoked it.

    ``False`` when another request holds it or already revoked it, when it was enrolled again (no longer
    past its age), or on any database error (rolled back and logged by type: the next request tries
    again).
    """
    try:
        return _revoke_and_record(db, device_id, now, ip_address, user_agent)
    except SQLAlchemyError as failure:
        db.rollback()
        logger.error("Device %s past its maximum age could not be revoked: %s", device_id, type(failure).__name__)
        return False


__all__ = [
    "EXPIRED_STATUS",
    "ISSUED_AT",
    "ages_out",
    "cutoff",
    "is_past_max_age",
    "issued_at",
    "max_age",
    "revoke_aged_out",
    "view_status",
    "within_max_age_filters",
]
