"""PRD-277 US-405 (§5.4, D-6): the first admins, ``ENTERPRISE_BOOTSTRAP_ADMINS``, at first sign-in.

``grant_if_listed`` is the first-admin rule ``core.auth.sign_in_flow`` calls on a FIRST sign-in only
(the person was created by this sign-in), after the access rule has let the person in. A person the
list names (``core.auth.bootstrap_admins``: by subject, or by an email the token marks verified):

* becomes an install admin: ``users.system_role`` is ``admin``;
* becomes owner of the default workspace (``DEFAULT_WORKSPACE_ID``): a ``workspace_members`` row with
  the ``owner`` role, and ``workspaces.owner_id`` when the workspace has no owner yet (the migration
  Job makes it with none, ``core.seeds.seed_default_workspace``, which this also calls, so an install
  whose Job has not run yet still works);
* is audited: one ``auth:bootstrap_admin`` row naming whether a subject or an email matched.

Nothing is committed here: the grants and their row land with the sign-in's own commit, or are
rolled back with it. A person who signed in before being listed is not a first sign-in, so the list
never changes them: an install admin does that in Automatos (``POST
/api/v1/auth/people/{user_id}/system-role``).

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
_MAKE_ADMIN = text("UPDATE users SET system_role = :role, updated_at = now() WHERE id = :id")
_TAKE_OWNERSHIP = text("UPDATE workspaces SET owner_id = :id WHERE id = CAST(:ws AS uuid) AND owner_id IS NULL")


def _listed_as(identity: VerifiedIdentity) -> Optional[str]:
    try:
        entries = bootstrap_admins.parse_entries(config.ENTERPRISE_BOOTSTRAP_ADMINS)
    except bootstrap_admins.MalformedEntry:
        logger.error("ENTERPRISE_BOOTSTRAP_ADMINS cannot be read; nobody was made a first admin")
        return None
    return bootstrap_admins.names_person(entries, identity.subject, identity.claims)


def grant_if_listed(db: Session, identity: VerifiedIdentity, user_id: int, *, ip_address: Optional[str],
                    user_agent: Optional[str]) -> Optional[str]:
    """Make a listed person an install admin and the default workspace's owner; not committed.

    The kind of entry that named them (``sub`` or ``email``), or None when the list names nobody
    here, or there is no default workspace to own.
    """
    matched_by = _listed_as(identity)
    if matched_by is None:
        return None
    workspace_id = session_audit.audit_workspace()
    if workspace_id is None:
        logger.error("users.id=%s is a first admin, but DEFAULT_WORKSPACE_ID is unset; nothing granted", user_id)
        return None
    ensure_default_workspace(db, workspace_id)
    db.execute(_MAKE_ADMIN, {"role": ADMIN, "id": user_id})
    people.set_membership(db, user_id, workspace_id, OWNER_ROLE)
    db.execute(_TAKE_OWNERSHIP, {"id": user_id, "ws": workspace_id})
    session_audit.stage_bootstrap_admin(db, user_id=user_id, workspace_id=workspace_id, matched_by=matched_by,
                                        ip_address=ip_address, user_agent=user_agent)
    logger.info("users.id=%s became a first admin (matched by %s)", user_id, matched_by)
    return matched_by


__all__ = ["OWNER_ROLE", "grant_if_listed"]
