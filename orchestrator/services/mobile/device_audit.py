"""PRD-277 US-601 (§5.12): a phone's pairing, its revocation and the push decision in ``audit_logs``.

Where the edition issues its own sessions (``issues_sessions``: enterprise), a device token is a
person's credential as a session is, so what happens to one is recorded as sessions are
(``core.auth.session_audit``): a pairing code made (``device:pairing_code_created``), a phone paired
with it (``device:paired``), a device revoked (``device:revoked``) and the install admin's answer to
"Send notifications through Expo's push service?" (``push:consent_changed``). ``local`` and ``saas``
write none of these (``records_devices``). The devices revoked when the install starts as another
edition, or with pairing switched off, are recorded in every edition (``services.mobile.edition_change``),
reason ``edition_changed`` or ``pairing_switched_off``; those an install admin revokes by ending their
person's sessions (``core.auth.people.end_access_of``), reason ``sessions_ended``.

Each row is staged in the caller's transaction and never committed here, so a change and its row
land together or not at all. A row holds the device's id, its person and why, the caller's address
and browser where there is a request; never a code, a token, a hash or a push key.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core import editions
from core.auth.session_audit import ACTOR_PERSON, ACTOR_SYSTEM, browser_fields
from core.workspaces.audit import AuditLog

DEVICE_RESOURCE = "mobile_device"
PUSH_CONSENT_RESOURCE = "push_consent"
PAIRING_CODE_CREATED = "device:pairing_code_created"
DEVICE_PAIRED = "device:paired"
DEVICE_REVOKED = "device:revoked"
PUSH_CONSENT_CHANGED = "push:consent_changed"
# Why a device was revoked: by a person (its own, or a workspace owner or admin); by an install admin
# ending its person's sessions; at the first start as another edition; at the first start with pairing off.
REVOKED_BY_PERSON = "revoked"
REVOKED_SESSIONS_ENDED = "sessions_ended"
REVOKED_EDITION_CHANGED = "edition_changed"
REVOKED_PAIRING_SWITCHED_OFF = "pairing_switched_off"


def records_devices() -> bool:
    """Whether this edition records device events: where it issues sessions (enterprise)."""
    return editions.current().issues_sessions


def stage(db: Session, *, action: str, workspace_id: Any, actor: Optional[int], resource_id: Any = None,
          details: Optional[Mapping[str, Any]] = None, ip_address: Optional[str] = None,
          user_agent: Optional[str] = None, resource_type: str = DEVICE_RESOURCE) -> None:
    """Add one ``audit_logs`` row to ``db``'s transaction; the caller commits it with the change.

    ``actor`` is the ``users.id`` that acted, or ``None`` for the install itself (``system``).
    """
    ip_address, user_agent = browser_fields(ip_address, user_agent)
    db.add(AuditLog(
        workspace_id=UUID(str(workspace_id)), user_id=actor,
        actor_type=ACTOR_PERSON if actor is not None else ACTOR_SYSTEM, action=action,
        resource_type=resource_type, resource_id=str(resource_id) if resource_id is not None else None,
        details=dict(details or {}), ip_address=ip_address, user_agent=user_agent,
    ))


__all__ = [
    "DEVICE_PAIRED",
    "DEVICE_RESOURCE",
    "DEVICE_REVOKED",
    "PAIRING_CODE_CREATED",
    "PUSH_CONSENT_CHANGED",
    "PUSH_CONSENT_RESOURCE",
    "REVOKED_BY_PERSON",
    "REVOKED_EDITION_CHANGED",
    "REVOKED_PAIRING_SWITCHED_OFF",
    "REVOKED_SESSIONS_ENDED",
    "records_devices",
    "stage",
]
