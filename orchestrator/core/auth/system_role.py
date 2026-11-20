"""PRD-277 US-203: an install-wide role the install stores for itself, not a Clerk claim.

``users.system_role`` (``core.models.user_identities.UserSystemRoleColumns``) holds
``super_admin``, ``admin`` or ``user``, or NULL when none was set. It is read only when the
edition signs people in with OIDC (``editions.current().sign_in == SIGN_IN_OIDC``). In
``saas`` the role comes from Clerk's claims (``core.auth.clerk.ClerkAuth.extract_user_info``)
and this column is never read; in ``local`` the one operator is ``super_admin`` and nothing
reads it either. The OIDC session lane that reads it is built in Waves 3 and 4.

Changing it is audited as a workspace role change is (``api/team.py``, ``member:role_changed``):
one ``audit_logs`` row through ``AuditService.log``, committed in the same transaction as the
change. When that row cannot be written, both are rolled back and ``SystemRoleAuditFailed`` is
raised, so no change goes unrecorded. Who may call ``set_system_role`` (an install admin, after
step-up) arrives with US-405; this module adds no route.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union
from uuid import UUID

from sqlalchemy import text

from core import editions
from core.models.user_identities import SYSTEM_ROLES
from core.workspaces.audit import AuditService

logger = logging.getLogger(__name__)

SYSTEM_ROLE_CHANGED_ACTION = "user:system_role_changed"
SYSTEM_ROLE_RESOURCE = "user"
ACTOR_PERSON = "user"
ACTOR_SYSTEM = "system"

NOT_STORED_MESSAGE = "This edition does not store install-wide roles."
UNKNOWN_ROLE_MESSAGE = "That is not an install-wide role."
UNKNOWN_PERSON_MESSAGE = "No such person."
AUDIT_FAILED_MESSAGE = "The role was not changed: its audit entry could not be written."


class SystemRoleNotStored(RuntimeError):
    """This edition takes install-wide roles from elsewhere (Clerk's claims, or the operator)."""


class UnknownPerson(LookupError):
    """No ``users`` row has the id given."""


class SystemRoleAuditFailed(RuntimeError):
    """The audit row could not be written, so the change was rolled back with it."""


@dataclass(frozen=True)
class SystemRoleChange:
    """What ``set_system_role`` found and left: the person, their role before and after."""

    user_id: int
    old_role: Optional[str]
    new_role: Optional[str]

    @property
    def changed(self) -> bool:
        """Whether the stored role moved (and so an audit row was written)."""
        return self.old_role != self.new_role


def stores_system_role() -> bool:
    """Whether this edition keeps install-wide roles in ``users.system_role`` (OIDC sign-in only)."""
    return editions.current().sign_in == editions.SIGN_IN_OIDC


def stored_system_role(db, user_id: int) -> Optional[str]:
    """The install-wide role stored for ``user_id`` when the edition signs people in with OIDC.

    None when none is stored, when there is no such person, and in every other edition, where
    the database is not read at all.
    """
    if not stores_system_role():
        return None
    row = db.execute(text("SELECT system_role FROM users WHERE id = :id"), {"id": user_id}).fetchone()
    return row[0] if row else None


def _locked_role(db, user_id: int) -> Optional[str]:
    """The person's stored role, their row locked until the caller's transaction ends."""
    row = db.execute(
        text("SELECT system_role FROM users WHERE id = :id FOR UPDATE"), {"id": user_id},
    ).fetchone()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)
    return row[0]


def _audit(db, change: SystemRoleChange, workspace_id: Union[UUID, str], actor_user_id: Optional[int],
           ip_address: Optional[str], user_agent: Optional[str]) -> None:
    """Write the change's one ``audit_logs`` row, which commits the change with it, or raise."""
    entry = AuditService(db).log(
        workspace_id=str(workspace_id),
        user_id=actor_user_id,
        action=SYSTEM_ROLE_CHANGED_ACTION,
        resource_type=SYSTEM_ROLE_RESOURCE,
        resource_id=str(change.user_id),
        details={"old_role": change.old_role, "new_role": change.new_role},
        ip_address=ip_address,
        user_agent=user_agent,
        actor_type=ACTOR_PERSON if actor_user_id is not None else ACTOR_SYSTEM,
    )
    if entry is None:
        logger.error("users.id=%s: system role change rolled back, its audit row failed", change.user_id)
        raise SystemRoleAuditFailed(AUDIT_FAILED_MESSAGE)


def set_system_role(
    db,
    *,
    user_id: int,
    role: Optional[str],
    workspace_id: Union[UUID, str],
    actor_user_id: Optional[int],
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> SystemRoleChange:
    """Store ``role`` (or NULL) as ``user_id``'s install-wide role, audited, and commit.

    Only in an edition that signs people in with OIDC (``SystemRoleNotStored`` elsewhere).
    ``role`` is one of ``SYSTEM_ROLES`` or None (``ValueError`` otherwise). The audit row is
    recorded in ``workspace_id``, the workspace the change was made from, with the acting
    person (``actor_user_id``; None records the system as the actor), the old and the new role.
    When the role is already ``role`` nothing is written and nothing is audited, and the
    session is committed all the same, which releases the row lock. Who may call this is the
    caller's check.

    Every path that reaches the row ends the caller's transaction: a change is committed with
    its audit row by ``AuditService.log`` (which rolls both back when the row fails), and a
    no-op is committed here. Call it with nothing else pending, as
    ``scripts/mint_pairing_code.py`` calls the same service.
    """
    if not stores_system_role():
        raise SystemRoleNotStored(NOT_STORED_MESSAGE)
    if role is not None and role not in SYSTEM_ROLES:
        raise ValueError(UNKNOWN_ROLE_MESSAGE)
    change = SystemRoleChange(user_id=user_id, old_role=_locked_role(db, user_id), new_role=role)
    if not change.changed:
        db.commit()  # ends the transaction as a change does, releasing the row lock
        return change
    db.execute(
        text("UPDATE users SET system_role = :role, updated_at = now() WHERE id = :id"),
        {"role": role, "id": user_id},
    )
    _audit(db, change, workspace_id, actor_user_id, ip_address, user_agent)
    return change


__all__ = [
    "SYSTEM_ROLE_CHANGED_ACTION",
    "SystemRoleAuditFailed",
    "SystemRoleChange",
    "SystemRoleNotStored",
    "UnknownPerson",
    "set_system_role",
    "stored_system_role",
    "stores_system_role",
]
