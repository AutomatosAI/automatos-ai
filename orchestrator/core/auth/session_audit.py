"""PRD-277 §5.12, US-304: sign-ins, refusals, sign-outs and ended sessions in ``audit_logs``.

Every event is one row through ``AuditService.log``, with the person's ``users.id`` where one is
known, the browser's address and agent, and never a token, a hash, a code or a secret: a refusal
records only its reason, one of ``REFUSAL_REASONS``, and anything else is a ``ValueError``.

The actions (§5.12) are all named here. Wave 3 writes ``auth:sign_in`` (``issue_session``),
``auth:sign_in_refused`` (a deactivated person; Wave 4's callback adds the other reasons),
``auth:sign_out`` and ``auth:session_revoked``, and ``user:deactivated`` for a person an admin
deactivates. ``auth:step_up``, ``auth:identity_linked`` and ``auth:bootstrap_admin`` are written by
the Wave 4 code whose events they are.

Which workspace a row goes in: ``audit_logs.workspace_id`` is required, and signing in or out
belongs to no workspace. Such a row goes in the workspace the caller names, else the install's
default workspace (``DEFAULT_WORKSPACE_ID``, ``audit_workspace``). An admin's action goes in the
workspace the admin acted from.

The browser's address and agent come from the request, so ``browser_fields`` makes them fit
before any row is written: the agent is cut to its column's size, and the address is kept only
when it is an IP address that fits, else none. Every row goes through it (``_write``), so an
oversized header can never make a row fail.

What happens when a row cannot be written:

* **Granting access** (a sign-in) is rolled back with it: ``AuditService.log`` commits the session
  and its row together, or rolls both back, and ``SessionAuditFailed`` is raised. Nobody gets in
  unrecorded.
* **Taking access away** (sign-out, an ended session, a deactivation) always lands: when the row
  fails, ``commit_removal`` applies the removal again on its own, commits it, and logs an ERROR
  naming the action. Nobody keeps access because a bookkeeping row failed.
* **A refusal** changes nothing, so a failed row is logged as an ERROR and the refusal stands.
"""
from __future__ import annotations

import ipaddress
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from uuid import UUID

from config import config
from core.auth.user_link import REFUSED_EMAIL_CONFLICT
from core.models.auth_sessions import IP_ADDRESS_MAX, USER_AGENT_MAX
from core.workspaces.audit import AuditLog, AuditService

logger = logging.getLogger(__name__)

T = TypeVar("T")

SIGN_IN = "auth:sign_in"
SIGN_IN_REFUSED = "auth:sign_in_refused"
SIGN_OUT = "auth:sign_out"
SESSION_REVOKED = "auth:session_revoked"
STEP_UP = "auth:step_up"
IDENTITY_LINKED = "auth:identity_linked"
# A workspace role change is already audited under this action (api/team.py); a sign-in hook's is too.
MEMBERSHIP_CHANGED = "member:role_changed"
MEMBER_RESOURCE = "member"
BOOTSTRAP_ADMIN = "auth:bootstrap_admin"
PERSON_DEACTIVATED = "user:deactivated"
SESSION_ACTIONS = (SIGN_IN, SIGN_IN_REFUSED, SIGN_OUT, SESSION_REVOKED, STEP_UP, IDENTITY_LINKED, BOOTSTRAP_ADMIN)

# Why a sign-in was refused (§5.12): the access rule, an email on another person, a deactivated
# person, an ID token that did not verify.
REFUSED_ACCESS_RULE = "access_rule"
REFUSED_DEACTIVATED = "deactivated"
REFUSED_INVALID_TOKEN = "invalid_token"
REFUSAL_REASONS = (REFUSED_ACCESS_RULE, REFUSED_EMAIL_CONFLICT, REFUSED_DEACTIVATED, REFUSED_INVALID_TOKEN)

SESSION_RESOURCE = "auth_session"
PERSON_RESOURCE = "user"
ACTOR_PERSON = "user"
ACTOR_SYSTEM = "system"

AUDIT_FAILED_MESSAGE = "You were not signed in: the sign-in could not be recorded. Try again."
NO_WORKSPACE_MESSAGE = "no workspace to record it in (DEFAULT_WORKSPACE_ID is unset)"


class SessionAuditFailed(RuntimeError):
    """A sign-in's audit row could not be written, so the sign-in was rolled back with it."""


def audit_workspace(workspace_id: Union[UUID, str, None] = None) -> Optional[str]:
    """The workspace a sign-in event is recorded in: the one given, else ``DEFAULT_WORKSPACE_ID``."""
    chosen = workspace_id or config.DEFAULT_WORKSPACE_ID
    try:
        return str(UUID(str(chosen).strip())) if chosen else None
    except ValueError:
        return None


def _ip_or_none(value: Optional[str]) -> Optional[str]:
    """``value`` when it is an IP address that fits its column, else None."""
    text = (value or "").strip()
    if not text or len(text) > IP_ADDRESS_MAX:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def browser_fields(ip_address: Optional[str], user_agent: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """The browser's address and agent as a row may hold them: a real address or none; the agent cut to size."""
    agent = (user_agent or "").strip()[:USER_AGENT_MAX]
    return _ip_or_none(ip_address), agent or None


def _write(db, *, action: str, workspace_id: Union[UUID, str, None], user_id: Optional[int],
           resource_type: str, resource_id: Any, details: Dict[str, Any],
           ip_address: Optional[str], user_agent: Optional[str]) -> Optional[object]:
    """One ``audit_logs`` row (committing whatever the session holds with it), or None."""
    ip_address, user_agent = browser_fields(ip_address, user_agent)
    workspace = audit_workspace(workspace_id)
    if workspace is None:
        db.rollback()
        logger.error("Audit %s not written: %s", action, NO_WORKSPACE_MESSAGE)
        return None
    return AuditService(db).log(
        workspace_id=workspace, user_id=user_id, action=action, resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None, details=details,
        ip_address=ip_address, user_agent=user_agent,
        actor_type=ACTOR_PERSON if user_id is not None else ACTOR_SYSTEM,
    )


def record_sign_in(db, *, user_id: int, session_id: UUID, ip_address: Optional[str],
                   user_agent: Optional[str], workspace_id: Union[UUID, str, None] = None) -> None:
    """``auth:sign_in``, committed with the new session, or both rolled back and ``SessionAuditFailed``."""
    entry = _write(db, action=SIGN_IN, workspace_id=workspace_id, user_id=user_id,
                   resource_type=SESSION_RESOURCE, resource_id=session_id, details={},
                   ip_address=ip_address, user_agent=user_agent)
    if entry is None:
        logger.error("users.id=%s: sign-in rolled back, its audit row failed", user_id)
        raise SessionAuditFailed(AUDIT_FAILED_MESSAGE)


def record_sign_in_refused(db, *, reason: str, user_id: Optional[int] = None, ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None, workspace_id: Union[UUID, str, None] = None) -> None:
    """``auth:sign_in_refused`` with its reason and nothing else; a failed row is logged, never raised."""
    if reason not in REFUSAL_REASONS:
        raise ValueError(f"unknown sign-in refusal reason {reason!r}")
    entry = _write(db, action=SIGN_IN_REFUSED, workspace_id=workspace_id, user_id=user_id,
                   resource_type=PERSON_RESOURCE, resource_id=user_id, details={"reason": reason},
                   ip_address=ip_address, user_agent=user_agent)
    if entry is None:
        logger.error("A refused sign-in (%s) could not be recorded", reason)


def record_identity_linked(db, *, user_id: int, details: Dict[str, Any], ip_address: Optional[str],
                           user_agent: Optional[str], workspace_id: Union[UUID, str, None] = None) -> None:
    """``auth:identity_linked`` for a first sign-in, committed with the new person and link.

    Granting access: when the row cannot be written, the person and the link are rolled back with
    it and ``SessionAuditFailed`` is raised. ``details`` names what was granted, never a claim.
    """
    entry = _write(db, action=IDENTITY_LINKED, workspace_id=workspace_id, user_id=user_id,
                   resource_type=PERSON_RESOURCE, resource_id=user_id, details=dict(details),
                   ip_address=ip_address, user_agent=user_agent)
    if entry is None:
        logger.error("users.id=%s: first sign-in rolled back, its audit row failed", user_id)
        raise SessionAuditFailed(AUDIT_FAILED_MESSAGE)


def stage_membership_set(db, *, user_id: int, workspace_id: Union[UUID, str], role: str, hook: str,
                         ip_address: Optional[str], user_agent: Optional[str]) -> None:
    """``member:role_changed`` for a membership a sign-in hook set, added to the sign-in's transaction.

    Not committed here: the row lands with the sign-in that made the change, or is rolled back with
    it, so a membership is never kept without its row.
    """
    ip_address, user_agent = browser_fields(ip_address, user_agent)
    db.add(AuditLog(workspace_id=UUID(str(workspace_id)), user_id=user_id, actor_type=ACTOR_SYSTEM,
                    action=MEMBERSHIP_CHANGED, resource_type=MEMBER_RESOURCE, resource_id=str(user_id),
                    details={"new_role": role, "set_by": "sign_in_hook", "hook": hook},
                    ip_address=ip_address, user_agent=user_agent))


def commit_removal(db, remove: Callable[[], T], record: Callable[[T], Optional[object]], action: str) -> T:
    """Apply ``remove``, then ``record`` its audit row, committing both; the removal always lands.

    ``record`` returns the row, or None when ``AuditService.log`` failed and rolled the removal
    back with it. Then ``remove`` runs again on its own and is committed, and an ERROR names the
    action. ``remove`` must be safe to repeat (it ends only what has not ended). When it ended
    nothing (``False`` or ``0``) there is no event: nothing is recorded, and the transaction is
    committed all the same, which releases any lock it took.
    """
    removed = remove()
    if not removed:
        db.commit()
        return removed
    if record(removed) is not None:
        return removed
    logger.error("Audit %s failed; the removal was committed without its row", action)
    removed = remove()
    db.commit()
    return removed


def session_ended_row(db, *, action: str, actor_user_id: Optional[int], subject: Any, resource_type: str,
                      details: Dict[str, Any], workspace_id: Union[UUID, str, None], ip_address: Optional[str],
                      user_agent: Optional[str]) -> Optional[object]:
    """The row for a sign-out, an ended session or a deactivation (``commit_removal``'s ``record``)."""
    return _write(db, action=action, workspace_id=workspace_id, user_id=actor_user_id,
                  resource_type=resource_type, resource_id=subject, details=details,
                  ip_address=ip_address, user_agent=user_agent)


__all__ = [
    "BOOTSTRAP_ADMIN",
    "IDENTITY_LINKED",
    "MEMBERSHIP_CHANGED",
    "PERSON_DEACTIVATED",
    "PERSON_RESOURCE",
    "REFUSAL_REASONS",
    "REFUSED_ACCESS_RULE",
    "REFUSED_DEACTIVATED",
    "REFUSED_EMAIL_CONFLICT",
    "REFUSED_INVALID_TOKEN",
    "SESSION_ACTIONS",
    "SESSION_RESOURCE",
    "SESSION_REVOKED",
    "SIGN_IN",
    "SIGN_IN_REFUSED",
    "SIGN_OUT",
    "STEP_UP",
    "SessionAuditFailed",
    "audit_workspace",
    "browser_fields",
    "commit_removal",
    "stage_membership_set",
    "record_identity_linked",
    "record_sign_in",
    "record_sign_in_refused",
    "session_ended_row",
]
