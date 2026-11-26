"""#1657: an install admin's Sync with Automatos Cloud, revocation Check now and revocation lift in ``audit_logs``.

PRD-277 says every person's action is in ``audit_logs``. These three are install-wide admin actions that
used to reach only the server log, so each now writes one row through ``AuditService.log``:

* ``marketplace:cloud_sync`` (``api/marketplace_cloud_sync.py``): a Sync run with the button, ``ok`` with
  the whole Sync's counts, the installs moved and the updates that failed, or ``failed``;
* ``themes:revocation_pull`` (``api/admin_theme_revocation_sync.py``): a Check now, ``ok`` with the rows the
  pull wrote, or ``failed``;
* ``themes:revocation_lifted`` (``api/admin_theme_revocations.py``): a revocation lifted here, with its
  ``origin`` (``local``, or ``feed`` while revocation sync is off or stale) and its level.

A row holds who acted (``users.id``), the browser's address and agent (``session_audit.browser_fields``) and
the outcome as counts: never an item's contents, a URL, an error's text, a token or a key. It goes in the
workspace the admin acted from, else the install's default workspace (``session_audit.audit_workspace``), as
``people.set_install_role`` does for an install-wide role.

When the row cannot be written:

* **A Sync or Check now stands.** It has already committed what it copied from Automatos (a Sync in several
  commits, so there is nothing to roll back as one), and undoing a revocation pull would drop protections the
  install was told to apply. The failure is logged at ERROR naming the action and the person
  (:func:`record_after`), and the answer is unchanged.
* **A lift is rolled back with its row** (``people.give_workspace``'s rule): lifting a revocation loosens a
  control, so it never lands unrecorded. :func:`record` answers None and the router refuses the lift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.auth.principal import resolve_user_pk
from core.auth.session_audit import ACTOR_PERSON, ACTOR_SYSTEM, NO_WORKSPACE_MESSAGE, audit_workspace, browser_fields
from core.security.client_ip import client_ip
from core.workspaces.audit import AuditLog, AuditService

logger = logging.getLogger(__name__)

CLOUD_SYNC = "marketplace:cloud_sync"
REVOCATION_PULL = "themes:revocation_pull"
REVOCATION_LIFTED = "themes:revocation_lifted"
CLOUD_SYNC_RESOURCE = "marketplace_cloud_sync"
REVOCATION_SYNC_RESOURCE = "theme_revocation_sync"
REVOCATION_RESOURCE = "theme_revocation"
OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"


@dataclass(frozen=True)
class AdminRequest:
    """Who acted, from where: the ``audit_logs`` fields of the row the action writes."""

    user_id: Optional[int]
    workspace_id: Any
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


def admin_request(db: Session, ctx: Any, request: Any) -> AdminRequest:
    """The caller's ``users.id``, the workspace they acted from and their browser's address and agent."""
    return AdminRequest(
        user_id=resolve_user_pk(db, ctx), workspace_id=getattr(ctx, "workspace_id", None),
        ip_address=client_ip(request), user_agent=request.headers.get("user-agent"),
    )


def record(db: Session, who: AdminRequest, *, action: str, resource_type: str, resource_id: Any = None,
           details: Optional[Dict[str, Any]] = None) -> Optional[AuditLog]:
    """Write one row, committing whatever ``db`` holds with it; None when it could not be written, in which
    case the transaction is rolled back (``AuditService.log``), whatever it held."""
    workspace = audit_workspace(who.workspace_id)
    if workspace is None:
        db.rollback()
        logger.error("Audit %s not written: %s", action, NO_WORKSPACE_MESSAGE)
        return None
    ip_address, user_agent = browser_fields(who.ip_address, who.user_agent)
    return AuditService(db).log(
        workspace_id=UUID(workspace), user_id=who.user_id, action=action, resource_type=resource_type,
        resource_id=None if resource_id is None else str(resource_id), details=details or {},
        ip_address=ip_address, user_agent=user_agent,
        actor_type=ACTOR_PERSON if who.user_id is not None else ACTOR_SYSTEM,
    )


def record_after(db: Session, ctx: Any, request: Any, *, action: str, resource_type: str,
                 details: Dict[str, Any]) -> None:
    """The row for an action already committed (a Sync, a Check now): written, or an ERROR logged; the
    action stands either way, even when the caller's ``users.id`` can't be read (:func:`admin_request`)."""
    try:
        who = admin_request(db, ctx, request)
    except Exception:  # noqa: BLE001 — logged; the action has committed, so its answer must not become a 500
        logger.exception("Audit %s not written: who acted could not be read; the action itself stands", action)
        return
    if record(db, who, action=action, resource_type=resource_type, details=details) is None:
        logger.error("Audit %s by users.id=%s could not be written; the action itself stands", action, who.user_id)


__all__ = [
    "AdminRequest", "CLOUD_SYNC", "CLOUD_SYNC_RESOURCE", "OUTCOME_FAILED", "OUTCOME_OK", "REVOCATION_LIFTED",
    "REVOCATION_PULL", "REVOCATION_RESOURCE", "REVOCATION_SYNC_RESOURCE", "admin_request", "record", "record_after",
]
