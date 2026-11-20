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

PRD-277 US-408 (§5.13) makes this the one module for people. It also holds:

* ``link_person``: the person a verified identity signs in as, found by ``(issuer, subject)`` or
  created (``core.auth.user_link.link_identity``), with ``email_verified`` exactly as the identity
  method verified it;
* ``person_is_active``: whether a person may sign in;
* ``set_membership``: a person's role in one workspace, written or, with ``if_absent``, written
  only when they have no membership there yet, so a role an admin set is never changed.

These three do not commit: sign-in commits them with its audit row (``core.auth.sign_in_flow``).
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
from core.auth.user_link import LinkedPerson, link_identity
from modules.policy.roles import WorkspaceRole
from services.mobile.devices import revoke_persons_devices

UNKNOWN_PERSON_MESSAGE = "No such person."
MEMBERSHIP_ROLES = tuple(role.value for role in WorkspaceRole)

_IS_ACTIVE = text("SELECT is_active FROM users WHERE id = :id")
_ADD_MEMBER = (
    "INSERT INTO workspace_members (workspace_id, user_id, role, is_active, joined_at) "
    "VALUES (CAST(:ws AS uuid), :uid, :role, true, now()) ON CONFLICT (workspace_id, user_id) "
)
_ADD_IF_ABSENT = text(_ADD_MEMBER + "DO NOTHING RETURNING id")
_ADD_OR_SET = text(_ADD_MEMBER + "DO UPDATE SET role = EXCLUDED.role, is_active = true RETURNING id")


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


def link_person(db: Session, *, issuer: str, subject: str, email_verified: bool,
                email: Optional[str] = None, name: Optional[str] = None) -> LinkedPerson:
    """The person ``(issuer, subject)`` signs in as, found or created; not committed.

    ``email_verified`` is the provider's own statement, from the verified ID token; ``link_identity``
    keeps an email only when it is True, and refuses (``IdentityLinkRefused``) an email another
    person holds. A repeat sign-in changes no row (PRD-277 §12, entry 8).
    """
    return link_identity(db, issuer=issuer, subject=subject, email_verified=email_verified,
                         email=email, name=name)


def person_is_active(db: Session, user_id: int) -> bool:
    """Whether ``user_id`` may sign in; ``UnknownPerson`` when there is no such row."""
    row = db.execute(_IS_ACTIVE, {"id": user_id}).fetchone()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)
    return row[0] is not False


def set_membership(db: Session, user_id: int, workspace_id: Union[UUID, str], role: str, *,
                   if_absent: bool = False) -> bool:
    """Give ``user_id`` ``role`` in ``workspace_id``; whether a row was written. Not committed.

    With ``if_absent`` an existing membership, active or not, is left exactly as it is.
    """
    if role not in MEMBERSHIP_ROLES:
        raise ValueError(f"a membership role is one of {', '.join(MEMBERSHIP_ROLES)}")
    statement = _ADD_IF_ABSENT if if_absent else _ADD_OR_SET
    written = db.execute(statement, {"ws": str(workspace_id), "uid": user_id, "role": role}).fetchone()
    return written is not None


__all__ = [
    "MEMBERSHIP_ROLES",
    "Deactivation",
    "Requester",
    "deactivate_person",
    "end_session",
    "end_sessions_of",
    "link_person",
    "person_is_active",
    "set_membership",
    "sign_out",
]
