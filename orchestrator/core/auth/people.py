"""PRD-277 US-303 (§5.6): ending a person's access: their sessions, or the person.

* ``end_session`` ends one session; ``end_sessions_of`` ends every session of one person. Each
  writes one ``auth:session_revoked`` row.
* ``sign_out`` ends the caller's own session and writes ``auth:sign_out``.
* ``deactivate_person`` sets ``users.is_active`` false, ends every session of the person and
  revokes every device they hold, in every workspace (``services.mobile.devices``), all in ONE
  transaction, and writes one ``user:deactivated`` row naming how many of each it ended. The
  device lane already refuses a deactivated person; revoking the devices as well means the
  person's phones are told ``device_revoked`` and their push state is cleared.

Taking access away always lands, even when its audit row cannot be written
(``session_audit.commit_removal``). Each function commits. Who may call them is the caller's
check (``api/auth_sessions.py``: an install admin after step-up; sign-out: the session itself).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth import session_audit, sessions
from core.auth.system_role import UnknownPerson
from services.mobile.devices import revoke_persons_devices

UNKNOWN_PERSON_MESSAGE = "No such person."


@dataclass(frozen=True)
class Requester:
    """Who asked, from where: the ``audit_logs`` fields of the row the change writes."""

    user_id: Optional[int]
    workspace_id: Union[UUID, str, None]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass(frozen=True)
class Deactivation:
    """What deactivating a person ended."""

    user_id: int
    sessions_ended: int
    devices_revoked: int


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row(db: Session, requester: Requester, *, action: str, subject, resource_type: str, details: dict):
    return session_audit.session_ended_row(
        db, action=action, actor_user_id=requester.user_id, subject=subject, resource_type=resource_type,
        details=details, workspace_id=requester.workspace_id, ip_address=requester.ip_address,
        user_agent=requester.user_agent,
    )


def sign_out(db: Session, session_id: UUID, requester: Requester) -> bool:
    """End the caller's own session (``sign_out``), with one ``auth:sign_out`` row; whether it ended."""
    now = _now()
    return session_audit.commit_removal(
        db,
        lambda: sessions.revoke_session(db, session_id, sessions.REVOKED_SIGN_OUT, now),
        lambda ended: _row(db, requester, action=session_audit.SIGN_OUT, subject=session_id,
                           resource_type=session_audit.SESSION_RESOURCE, details={}),
        session_audit.SIGN_OUT,
    )


def end_session(db: Session, session_id: UUID, requester: Requester) -> bool:
    """End one session (an admin's), with one ``auth:session_revoked`` row; whether it was live."""
    now = _now()
    return session_audit.commit_removal(
        db,
        lambda: sessions.revoke_session(db, session_id, sessions.REVOKED_BY_ADMIN, now),
        lambda ended: _row(db, requester, action=session_audit.SESSION_REVOKED, subject=session_id,
                           resource_type=session_audit.SESSION_RESOURCE, details={"sessions": int(ended)}),
        session_audit.SESSION_REVOKED,
    )


def end_sessions_of(db: Session, user_id: int, requester: Requester) -> int:
    """End every session of ``user_id``, with one ``auth:session_revoked`` row; how many ended."""
    now = _now()
    return session_audit.commit_removal(
        db,
        lambda: sessions.revoke_sessions_of(db, user_id, sessions.REVOKED_BY_ADMIN, now),
        lambda ended: _row(db, requester, action=session_audit.SESSION_REVOKED, subject=user_id,
                           resource_type=session_audit.PERSON_RESOURCE, details={"sessions": ended}),
        session_audit.SESSION_REVOKED,
    )


def _lock_person(db: Session, user_id: int) -> None:
    """Lock the person's row until the transaction ends; ``UnknownPerson`` when there is none."""
    row = db.execute(text("SELECT id FROM users WHERE id = :id FOR UPDATE"), {"id": user_id}).fetchone()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)


def _deactivate(db: Session, user_id: int, revoked_by: Optional[int], now: datetime) -> Deactivation:
    """The three changes, in the caller's transaction: the person, their sessions, their devices."""
    _lock_person(db, user_id)
    db.execute(text("UPDATE users SET is_active = false, updated_at = now() WHERE id = :id"), {"id": user_id})
    ended = sessions.revoke_sessions_of(db, user_id, sessions.REVOKED_DEACTIVATED, now)
    revoked = revoke_persons_devices(db, user_id, revoked_by, now)
    return Deactivation(user_id=user_id, sessions_ended=ended, devices_revoked=revoked)


def deactivate_person(db: Session, user_id: int, requester: Requester) -> Deactivation:
    """Deactivate ``user_id``: the person, every session and every device in one commit, audited."""
    now = _now()
    return session_audit.commit_removal(
        db,
        lambda: _deactivate(db, user_id, requester.user_id, now),
        lambda done: _row(db, requester, action=session_audit.PERSON_DEACTIVATED, subject=user_id,
                          resource_type=session_audit.PERSON_RESOURCE,
                          details={"sessions": done.sessions_ended, "devices": done.devices_revoked}),
        session_audit.PERSON_DEACTIVATED,
    )


__all__ = [
    "Deactivation",
    "Requester",
    "deactivate_person",
    "end_session",
    "end_sessions_of",
    "sign_out",
]
