"""PRD-277 US-303 (§5.6): ending a person's access: their sessions, or the person.

* ``end_session`` ends one session; ``end_sessions_of`` ends every session of one person. Each
  writes one ``auth:session_revoked`` row. ``end_access_of`` is an admin ending ANOTHER person's
  sessions: it also revokes every device they hold, in the same commit, with one ``device:revoked``
  row per device (reason ``sessions_ended``) and the counts on the ``auth:session_revoked`` row.
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

The Team page's routes (``api/team.py``) use it too, with the same statements they ran before:
``workspace_member`` (a membership by id in one workspace), ``change_member_role``,
``member_to_remove`` and ``end_membership``, and, accepting an invitation, ``person_for_invitee``
(the person, found by Clerk id or created) and ``join_from_invitation`` (the membership the
invitation grants; not committed). The rules of a membership change live here, not in the route:
the target is looked up in the workspace named and nowhere else (``MemberNotFound``), and a
workspace keeps its one owner: the owner's role is never changed, nobody is made owner, and the
owner's membership never ends (``MembershipRefused``). The two that change a row commit.

**People waiting for a workspace** (US-504). ``waiting_people`` lists those who signed in with an
identity and can open no workspace; ``give_workspace`` makes one of them a member of a workspace in
a role below owner, with its ``member:added_from_waiting`` audit row in the same commit. The route
(``api/team_waiting.py``) lets only an install admin with ``members:invite`` there do it, after
step-up.

**Install admins** (US-405). ``set_install_role`` gives a person an install-wide role
(``core.auth.system_role.set_system_role``, audited ``user:system_role_changed`` in the workspace the
admin acted from, else the default workspace). The install always keeps an active install admin:
``set_install_role`` taking the role away and ``deactivate_person`` both refuse
(``LastInstallAdmin``, nothing changed) when the person is the last one. Every active install
admin's row is locked in ``id`` order first, so two admins removing each other at once cannot both
succeed. ``locked_active_install_admins`` is that lock and count, which the first-admin rule uses
too (``core.auth.first_admin``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth import session_audit, sessions
from core.auth.system_role import (
    SystemRoleAuditFailed,
    SystemRoleChange,
    UnknownPerson,
    set_system_role,
    stored_system_role,
)
from core.auth.user_link import (PENDING_EMAIL_DOMAIN, LinkedPerson, link_identity, refuse_if_linked_before,
                                  user_id_for_invitee)
from core.workspaces.audit import AuditService
from core.workspaces.models import WorkspaceMember
from modules.policy.roles import ADMIN, SUPER_ADMIN, USER, WorkspaceRole
from services.mobile import device_audit
from services.mobile.devices import revoke_and_list_persons_devices, revoke_persons_devices

UNKNOWN_PERSON_MESSAGE = "No such person."
MEMBERSHIP_ROLES = tuple(role.value for role in WorkspaceRole)
# The install-wide roles that make an install admin (core.auth.roles.caller_is_admin).
INSTALL_ADMIN_ROLES = (ADMIN, SUPER_ADMIN)
LAST_INSTALL_ADMIN_MESSAGE = (
    "This person is the install's last active admin. Make another person an install admin first."
)
_LOCK_ACTIVE_ADMINS = text(
    "SELECT id FROM users WHERE system_role = ANY(:roles) AND is_active IS NOT FALSE ORDER BY id FOR UPDATE"
)


class LastInstallAdmin(RuntimeError):
    """The change would leave the install with no active install admin; nothing was changed."""


# A workspace's memberships (the Team page). The messages are the ones the page has always shown.
OWNER_ROLE = WorkspaceRole.OWNER.value
SETTABLE_MEMBER_ROLES = tuple(role.value for role in WorkspaceRole if role != WorkspaceRole.OWNER)
MEMBER_NOT_FOUND_MESSAGE = "Member not found"
OWNER_ROLE_FIXED_MESSAGE = "The owner's role cannot be changed."
ONE_OWNER_MESSAGE = "A workspace has one owner: make them an admin, editor or viewer."
OWNER_NOT_REMOVABLE_MESSAGE = "Cannot remove workspace owner"


class MemberNotFound(LookupError):
    """No such membership in the workspace named; nothing was changed."""


class MembershipRefused(ValueError):
    """A membership change a workspace does not allow (its one owner stays its one owner); nothing changed."""


_IS_ACTIVE = text("SELECT is_active FROM users WHERE id = :id")
_ADD_MEMBER = (
    "INSERT INTO workspace_members (workspace_id, user_id, role, is_active, joined_at) "
    "VALUES (CAST(:ws AS uuid), :uid, :role, true, now()) ON CONFLICT (workspace_id, user_id) "
)
_ADD_IF_ABSENT = text(_ADD_MEMBER + "DO NOTHING RETURNING id")
_ADD_OR_SET = text(_ADD_MEMBER + "DO UPDATE SET role = EXCLUDED.role, is_active = true RETURNING id")

# PRD-277 US-504: a person waiting for a workspace signed in with an identity (a user_identities
# row), is active, and can open no usable workspace: no active membership of one, and owns none.
# ``:only`` narrows it to one person (``give_workspace``'s check), or is NULL for everyone. Oldest
# first, so whoever has waited longest is on the first page; ``total`` counts every row before the
# page is cut (a window count is taken before LIMIT and OFFSET).
_WAITING_SQL = text(
    "SELECT u.id, u.email, u.name, u.created_at, COUNT(*) OVER () AS total FROM users u "
    "WHERE (CAST(:only AS integer) IS NULL OR u.id = CAST(:only AS integer)) "
    "AND u.is_active IS NOT FALSE "
    "AND EXISTS (SELECT 1 FROM user_identities ui WHERE ui.user_id = u.id) "
    "AND NOT EXISTS (SELECT 1 FROM workspace_members wm JOIN workspaces w ON w.id = wm.workspace_id "
    "  WHERE wm.user_id = u.id AND wm.is_active = true AND w.is_active IS NOT FALSE AND w.deleted_at IS NULL) "
    "AND NOT EXISTS (SELECT 1 FROM workspaces w WHERE w.owner_id = u.id "
    "  AND w.is_active IS NOT FALSE AND w.deleted_at IS NULL) "
    "ORDER BY u.created_at ASC, u.id ASC LIMIT :limit OFFSET :offset"
)
WAITING_PAGE_MAX = 100
GIVEN_A_WORKSPACE = "member:added_from_waiting"
NOT_WAITING_MESSAGE = "This person is not waiting for a workspace any more."
WORKSPACE_NOT_GIVEN_MESSAGE = "The person was not added: the change could not be recorded. Try again."


class NotWaiting(LookupError):
    """The person is not waiting for a workspace (in one already, deactivated, or never signed in); nothing changed."""


class WorkspaceNotGiven(RuntimeError):
    """The membership's audit row could not be written, so the membership was rolled back."""


@dataclass(frozen=True)
class WaitingPerson:
    """A person waiting for a workspace, as an install admin sees them."""

    user_id: int
    email: Optional[str]
    name: Optional[str]
    signed_in_first_at: Optional[datetime]


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


@dataclass(frozen=True)
class AccessEnded:
    """What an admin ending another person's access ended: their sessions, and their devices as (id, workspace)."""

    sessions_ended: int
    devices: Tuple[Tuple[UUID, UUID], ...]

    def __bool__(self) -> bool:
        return bool(self.sessions_ended or self.devices)


def _end_access(db: Session, user_id: int, revoked_by: Optional[int], now: datetime) -> AccessEnded:
    """The person's sessions and every device they hold, in the caller's transaction."""
    ended = sessions.revoke_sessions_of(db, user_id, sessions.REVOKED_BY_ADMIN, now)
    revoked = revoke_and_list_persons_devices(db, user_id, revoked_by, now)
    return AccessEnded(sessions_ended=ended, devices=tuple((row.id, row.workspace_id) for row in revoked))


def _record_access_ended(db: Session, requester: Requester, user_id: int, done: AccessEnded):
    """One ``device:revoked`` row per device (reason ``sessions_ended``), then the ``auth:session_revoked`` row.

    Staged before that row is written, so they are committed with it, or rolled back with it when it fails.
    """
    for device_id, workspace_id in done.devices:
        device_audit.stage(db, action=device_audit.DEVICE_REVOKED, workspace_id=workspace_id,
                           actor=requester.user_id, resource_id=device_id,
                           details={"reason": device_audit.REVOKED_SESSIONS_ENDED, "person": user_id},
                           ip_address=requester.ip_address, user_agent=requester.user_agent)
    return _row(db, requester, action=session_audit.SESSION_REVOKED, subject=user_id,
                resource_type=session_audit.PERSON_RESOURCE,
                details={"sessions": done.sessions_ended, "devices": len(done.devices)})


def end_access_of(db: Session, user_id: int, requester: Requester) -> AccessEnded:
    """An admin ends every session of ANOTHER person and revokes every device they hold, in one commit.

    A paired phone would otherwise keep the access the admin meant to end. Audited as above. A person
    ending their own sessions, and ending one session, use ``end_sessions_of`` and ``end_session``:
    their phones stay, and they manage them in Settings.
    """
    now = _now()
    return session_audit.commit_removal(
        db,
        lambda: _end_access(db, user_id, requester.user_id, now),
        lambda done: _record_access_ended(db, requester, user_id, done),
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


def locked_active_install_admins(db: Session) -> Tuple[int, ...]:
    """The ids of every active install admin, each row locked (in ``id`` order) until the
    transaction ends. The one way this install counts its admins: the lock comes with the count."""
    return tuple(row[0] for row in db.execute(_LOCK_ACTIVE_ADMINS, {"roles": list(INSTALL_ADMIN_ROLES)}))


def _assert_not_last_install_admin(db: Session, user_id: int) -> None:
    """``LastInstallAdmin`` when ``user_id`` is the install's only active install admin.

    Every active install admin's row is locked, in ``id`` order, until the transaction ends, so two
    admins removing each other at once are serialised and the second sees the first's change. A
    person who is no install admin locks nothing.
    """
    if (stored_system_role(db, user_id) or USER) not in INSTALL_ADMIN_ROLES:
        return
    admins = locked_active_install_admins(db)
    if admins == (user_id,):
        db.rollback()
        raise LastInstallAdmin(LAST_INSTALL_ADMIN_MESSAGE)


def deactivate_person(db: Session, user_id: int, requester: Requester) -> Deactivation:
    """Deactivate ``user_id``: the person, every session and every device in one commit, audited.

    ``LastInstallAdmin`` (nothing changed) when they are the install's last active install admin.
    """
    now = _now()
    _assert_not_last_install_admin(db, user_id)
    return session_audit.commit_removal(
        db,
        lambda: _deactivate(db, user_id, requester.user_id, now),
        lambda done: _row(db, requester, action=session_audit.PERSON_DEACTIVATED, subject=user_id,
                          resource_type=session_audit.PERSON_RESOURCE,
                          details={"sessions": done.sessions_ended, "devices": done.devices_revoked}),
        session_audit.PERSON_DEACTIVATED,
    )


def set_install_role(db: Session, user_id: int, role: str, requester: Requester) -> SystemRoleChange:
    """Give ``user_id`` the install-wide ``role`` (US-405), audited, and commit.

    ``user:system_role_changed`` is recorded in the workspace the admin acted from, else the
    install's default workspace (PRD-277 §12, entry 7). ``LastInstallAdmin`` (nothing changed) when
    the change would leave no active install admin; ``UnknownPerson`` for no such person.
    """
    workspace_id = session_audit.audit_workspace(requester.workspace_id)
    if workspace_id is None:
        raise SystemRoleAuditFailed(session_audit.NO_WORKSPACE_MESSAGE)
    if role not in INSTALL_ADMIN_ROLES:
        _assert_not_last_install_admin(db, user_id)
    return set_system_role(db, user_id=user_id, role=role, workspace_id=workspace_id,
                           actor_user_id=requester.user_id, ip_address=requester.ip_address,
                           user_agent=requester.user_agent)


def link_person(db: Session, *, issuer: str, subject: str, email_verified: bool,
                email: Optional[str] = None, name: Optional[str] = None,
                linked_before_as: Optional[str] = None) -> LinkedPerson:
    """The person ``(issuer, subject)`` signs in as, found or created; not committed.

    ``email_verified`` is the provider's own statement, from the verified ID token; ``link_identity``
    keeps an email only when it is True, and refuses (``IdentityLinkRefused``) an email another
    person holds, and a first sign-in of a person already linked as ``linked_before_as``. A repeat
    sign-in changes no row (PRD-277 §12, entry 8).
    """
    if linked_before_as is not None:
        refuse_if_linked_before(db, issuer=issuer, subject=subject, linked_before_as=linked_before_as)
    return link_identity(db, issuer=issuer, subject=subject, email_verified=email_verified,
                         email=email, name=name)


def person_is_active(db: Session, user_id: int) -> bool:
    """Whether ``user_id`` may sign in; ``UnknownPerson`` when there is no such row."""
    row = db.execute(_IS_ACTIVE, {"id": user_id}).fetchone()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)
    return row[0] is not False


def workspace_member(db: Session, workspace_id: Union[UUID, str], member_id: int) -> Optional[WorkspaceMember]:
    """The membership ``member_id`` in ``workspace_id``, or None (the Team page's lookup)."""
    return db.query(WorkspaceMember).filter(
        WorkspaceMember.id == member_id,
        WorkspaceMember.workspace_id == workspace_id,
    ).first()


def _member_in(db: Session, workspace_id: Union[UUID, str], member_id: int) -> WorkspaceMember:
    """The membership ``member_id`` in ``workspace_id``; ``MemberNotFound`` for one in no workspace or another."""
    member = workspace_member(db, workspace_id, member_id)
    if member is None:
        raise MemberNotFound(MEMBER_NOT_FOUND_MESSAGE)
    return member


def change_member_role(db: Session, workspace_id: Union[UUID, str], member_id: int, role: str) -> str:
    """Give the membership ``member_id`` of ``workspace_id`` the role ``role`` and commit; the role it had.

    A workspace has one owner: the owner's role is never changed (``MembershipRefused``), nobody is
    made owner, and ``role`` must be one of ``SETTABLE_MEMBER_ROLES``. ``MemberNotFound`` when the
    membership is not in that workspace. Its active flag is untouched.
    """
    member = _member_in(db, workspace_id, member_id)
    if member.role == OWNER_ROLE:
        raise MembershipRefused(OWNER_ROLE_FIXED_MESSAGE)
    if str(role).strip().lower() == OWNER_ROLE:
        raise MembershipRefused(ONE_OWNER_MESSAGE)
    if role not in SETTABLE_MEMBER_ROLES:
        raise MembershipRefused(f"Invalid role: {role}")
    old_role = member.role
    member.role = role
    db.commit()
    return old_role


def member_to_remove(db: Session, workspace_id: Union[UUID, str], member_id: int) -> WorkspaceMember:
    """The membership ``member_id`` of ``workspace_id`` when it may end; never the owner's.

    ``MemberNotFound`` when it is not in that workspace; ``MembershipRefused`` for the owner.
    """
    member = _member_in(db, workspace_id, member_id)
    if member.role == OWNER_ROLE:
        raise MembershipRefused(OWNER_NOT_REMOVABLE_MESSAGE)
    return member


def end_membership(db: Session, workspace_id: Union[UUID, str], member_id: int) -> WorkspaceMember:
    """Deactivate the membership ``member_id`` of ``workspace_id`` (the person leaves) and commit.

    The same checks as ``member_to_remove``: never the owner's, never one in another workspace.
    """
    member = member_to_remove(db, workspace_id, member_id)
    member.is_active = False
    db.commit()
    return member


def person_for_invitee(db: Session, clerk_user_id: str, email: Optional[str], *, name: Optional[str] = None) -> int:
    """The person accepting an invitation, found by their Clerk id or created; not committed.

    ``core.auth.user_link.user_id_for_invitee``: an email another row holds is refused, never attached.
    """
    return user_id_for_invitee(db, clerk_user_id, email, name=name)


def join_from_invitation(db: Session, invitation: Any, user_id: int) -> bool:
    """Make ``user_id`` an active member in the invitation's role; True when they already were.

    Idempotent: an existing inactive row is reactivated. Marks the invitation accepted the first
    time. Not committed: the caller commits.
    """
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == invitation.workspace_id,
        WorkspaceMember.user_id == user_id,
    ).first()

    already_member = bool(member and member.is_active)
    if member:
        member.is_active = True
        member.role = invitation.role
    else:
        db.add(WorkspaceMember(
            workspace_id=invitation.workspace_id,
            user_id=user_id,
            role=invitation.role,
            invited_by=invitation.invited_by,
            invited_at=invitation.created_at,
            joined_at=datetime.utcnow(),
            is_active=True,
        ))

    if not invitation.accepted_at:
        invitation.accepted_at = datetime.utcnow()
        invitation.accepted_by_user_id = user_id
    return already_member


def _waiting_person(row: Any) -> WaitingPerson:
    email = row[1] if row[1] and not str(row[1]).lower().endswith(PENDING_EMAIL_DOMAIN) else None
    return WaitingPerson(user_id=int(row[0]), email=email, name=row[2] or None, signed_in_first_at=row[3])


@dataclass(frozen=True)
class WaitingPage:
    """One page of the people waiting, oldest first, and how many are waiting in all."""

    people: Tuple[WaitingPerson, ...]
    total: int


def _waiting_rows(db: Session, limit: int, offset: int) -> list:
    return db.execute(_WAITING_SQL, {"only": None, "limit": limit, "offset": offset}).fetchall()


def waiting_people(db: Session, limit: int = WAITING_PAGE_MAX, offset: int = 0) -> WaitingPage:
    """A page of the people waiting for a workspace (``_WAITING_SQL``), oldest first; never committed.

    ``limit`` is capped at ``WAITING_PAGE_MAX``. A page past the end has no row to carry the
    count, so the count is then read from the first row. A placeholder email (an identity whose
    provider verified no address) is shown as none.
    """
    limit = max(1, min(int(limit), WAITING_PAGE_MAX))
    offset = max(0, int(offset))
    rows = _waiting_rows(db, limit, offset)
    if rows:
        total = int(rows[0][4])
    else:
        first = _waiting_rows(db, 1, 0) if offset else []
        total = int(first[0][4]) if first else 0
    return WaitingPage(people=tuple(_waiting_person(row) for row in rows), total=total)


def give_workspace(db: Session, user_id: int, workspace_id: Union[UUID, str], role: str,
                   requester: Requester) -> None:
    """Make a waiting person an active member of ``workspace_id`` in ``role``, audited, and commit.

    ``MembershipRefused`` for a role outside ``SETTABLE_MEMBER_ROLES`` (nobody is made owner);
    ``UnknownPerson`` for no such person; ``NotWaiting`` when they are deactivated, have no identity
    or are in a workspace already (an admin's later change is never overwritten). The person's row
    is locked first, so two admins placing the same person at once are serialised. The membership
    and its ``member:added_from_waiting`` row commit together, or neither does
    (``WorkspaceNotGiven``).
    """
    if role not in SETTABLE_MEMBER_ROLES:
        raise MembershipRefused(f"Invalid role: {role}")
    _lock_person(db, user_id)
    if db.execute(_WAITING_SQL, {"only": user_id, "limit": 1, "offset": 0}).fetchone() is None:
        db.rollback()
        raise NotWaiting(NOT_WAITING_MESSAGE)
    set_membership(db, user_id, workspace_id, role)
    entry = AuditService(db).log(
        workspace_id=str(workspace_id), user_id=requester.user_id, action=GIVEN_A_WORKSPACE,
        resource_type="member", resource_id=str(user_id), details={"role": role},
        ip_address=requester.ip_address, user_agent=requester.user_agent,
    )
    if entry is None:  # the log rolled the transaction back, the membership with it
        raise WorkspaceNotGiven(WORKSPACE_NOT_GIVEN_MESSAGE)


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
    "INSTALL_ADMIN_ROLES",
    "LAST_INSTALL_ADMIN_MESSAGE",
    "MEMBERSHIP_ROLES",
    "OWNER_ROLE",
    "SETTABLE_MEMBER_ROLES",
    "AccessEnded",
    "Deactivation",
    "LastInstallAdmin",
    "MemberNotFound",
    "MembershipRefused",
    "NotWaiting",
    "Requester",
    "WAITING_PAGE_MAX",
    "WaitingPage",
    "WaitingPerson",
    "WorkspaceNotGiven",
    "change_member_role",
    "deactivate_person",
    "end_membership",
    "end_access_of",
    "end_session",
    "end_sessions_of",
    "give_workspace",
    "join_from_invitation",
    "link_person",
    "locked_active_install_admins",
    "member_to_remove",
    "person_for_invitee",
    "person_is_active",
    "set_install_role",
    "set_membership",
    "sign_out",
    "waiting_people",
    "workspace_member",
]
