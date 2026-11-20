"""PRD-277 US-405 (§5.4, D-6): the first admins, ``ENTERPRISE_BOOTSTRAP_ADMINS``.

``grant_if_listed`` is the first-admin rule ``core.auth.sign_in_flow`` calls on every sign-in, after
the access rule has let the person in and a deactivated person has been refused. A person the list
names (``core.auth.bootstrap_admins``: by subject, or by an email the token marks verified in the
standard ``email`` claim) is granted:

* at their FIRST sign-in (the person was created by this sign-in), always;
* at any later sign-in only while the install has NO active install admin, counted under the lock
  the last-admin guard takes (``people.locked_active_install_admins``). So a list mistyped at first
  and corrected later still works. Once an install admin exists, the list does nothing for a person
  who already exists: an install admin does that in Automatos (``POST
  /api/v1/auth/people/{user_id}/system-role``).

The grant:

* makes them an install admin: ``users.system_role`` is ``admin``;
* gives them a membership of the default workspace (``DEFAULT_WORKSPACE_ID``): ``owner`` when the
  workspace has no owner yet, and they become ``workspaces.owner_id``; ``admin`` when it already has
  one, which stays its one owner. The workspace row is locked while that is decided, so two first
  admins at once never make two owners. The migration Job makes the workspace with no owner
  (``core.seeds.seed_default_workspace``), which this also calls, so an install whose Job has not
  run yet still works;
* is audited: one ``auth:bootstrap_admin`` row naming whether a subject or an email matched, and the
  workspace role given.

Nothing is committed here: the grants and their row land with the sign-in's own commit, or are
rolled back with it (a hook that fails afterwards takes the whole grant with it).

A list that cannot be read (start-up refuses it, so only a value changed after start-up) grants
nobody, and says so in the log without quoting it.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.auth import bootstrap_admins, people, session_audit
from core.auth.sign_in_methods import VerifiedIdentity
from core.seeds.seed_default_workspace import ensure_default_workspace
from modules.policy.roles import ADMIN, WorkspaceRole

logger = logging.getLogger(__name__)

OWNER_ROLE = WorkspaceRole.OWNER.value
ADMIN_MEMBER_ROLE = WorkspaceRole.ADMIN.value
_MAKE_ADMIN = text("UPDATE users SET system_role = :role, updated_at = now() WHERE id = :id")
_TAKE_OWNERSHIP = text("UPDATE workspaces SET owner_id = :id WHERE id = CAST(:ws AS uuid) AND owner_id IS NULL")
# The workspace's row, locked until the sign-in commits, and whether anyone else already owns it.
_OWNED_BY_ANOTHER = text(
    "SELECT (w.owner_id IS NOT NULL AND w.owner_id <> :id) OR EXISTS ("
    "SELECT 1 FROM workspace_members m WHERE m.workspace_id = w.id AND m.role = :owner AND m.user_id <> :id) "
    "FROM workspaces w WHERE w.id = CAST(:ws AS uuid) FOR UPDATE OF w"
)


def _listed_as(identity: VerifiedIdentity) -> Optional[str]:
    try:
        entries = bootstrap_admins.parse_entries(config.ENTERPRISE_BOOTSTRAP_ADMINS)
    except bootstrap_admins.MalformedEntry:
        logger.error("ENTERPRISE_BOOTSTRAP_ADMINS cannot be read; nobody was made a first admin")
        return None
    return bootstrap_admins.names_person(entries, identity.subject, identity.claims)


def _workspace_role(db: Session, user_id: int, workspace_id: str) -> str:
    """``owner`` when nobody else owns the (locked) default workspace, else ``admin``."""
    owned = db.execute(_OWNED_BY_ANOTHER, {"id": user_id, "ws": workspace_id, "owner": OWNER_ROLE}).scalar()
    return ADMIN_MEMBER_ROLE if owned else OWNER_ROLE


def grant_if_listed(db: Session, identity: VerifiedIdentity, user_id: int, *, created: bool,
                    ip_address: Optional[str], user_agent: Optional[str]) -> Optional[str]:
    """Make a listed person an install admin with a membership of the default workspace; not committed.

    ``created``: this sign-in created the person. A person who already existed is granted only
    while the install has no active install admin. The kind of entry that named them (``sub`` or
    ``email``), or None when nothing was granted.
    """
    matched_by = _listed_as(identity)
    if matched_by is None:
        return None
    if not created and people.locked_active_install_admins(db):
        return None
    workspace_id = session_audit.audit_workspace()
    if workspace_id is None:
        logger.error("users.id=%s is a first admin, but DEFAULT_WORKSPACE_ID is unset; nothing granted", user_id)
        return None
    ensure_default_workspace(db, workspace_id)
    role = _workspace_role(db, user_id, workspace_id)
    db.execute(_MAKE_ADMIN, {"role": ADMIN, "id": user_id})
    people.set_membership(db, user_id, workspace_id, role)
    if role == OWNER_ROLE:
        db.execute(_TAKE_OWNERSHIP, {"id": user_id, "ws": workspace_id})
    session_audit.stage_bootstrap_admin(db, user_id=user_id, workspace_id=workspace_id, matched_by=matched_by,
                                        workspace_role=role, ip_address=ip_address, user_agent=user_agent)
    logger.info("users.id=%s became a first admin (matched by %s, workspace role %s)", user_id, matched_by, role)
    return matched_by


__all__ = ["ADMIN_MEMBER_ROLE", "OWNER_ROLE", "grant_if_listed"]
