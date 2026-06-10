"""Workspace-member handlers for PlatformActionExecutor (PRD-143 S11).

The administration surface, deliberately operator-tier (the Rev 2 inversion):
list/invite/role-set/remove workspace members. Safety is gates-and-logs, not
exclusion — role-set and remove are ``destructive`` (executor confirmation
backstop + audit), every lookup is workspace-filtered (tenant isolation), and
invite delegates to ``core.workspaces.invitations.invite_member_to_workspace``
— the flow extracted from POST /api/workspaces/{id}/team/invite into the
invitation service layer — so router and tool share one implementation.

Audit principal: handlers run without a request context, so workspace-audit
rows are attributed to the workspace owner (the principal Auto acts for —
same inheritance model as the executor's ``_workspace_has_admin_owner``),
with ``via: platform_tool`` in the details. ``workspace_id`` comes from the
executor context, never the params. Clerk-org sync (a web-session concern)
is not performed here; the DB row is the source of truth.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _workspace_owner_user_id(db: Session, workspace_id: UUID) -> Optional[int]:
    """Resolve the workspace owner's internal users.id (the audit principal)."""
    from core.workspaces.models import WorkspaceMember

    owner = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.role == "owner",
            WorkspaceMember.is_active == True,  # noqa: E712
        )
        .first()
    )
    return owner.user_id if owner else None


def _audit(db: Session, workspace_id: UUID, user_id: Optional[int], action: str,
           resource_id: Any = None, details: Optional[dict] = None) -> None:
    """Best-effort workspace audit entry (audit_logs.user_id is NOT NULL)."""
    if user_id is None:
        logger.warning(
            "[members] no owner principal for workspace %s — skipping AuditService "
            "entry for %s (tool_execution_logs still records the call)",
            workspace_id, action,
        )
        return
    try:
        from core.workspaces.audit import AuditService

        AuditService(db).log(
            workspace_id=str(workspace_id),
            user_id=user_id,
            action=action,
            resource_type="member",
            resource_id=str(resource_id) if resource_id is not None else None,
            details={**(details or {}), "via": "platform_tool"},
        )
    except Exception as exc:
        logger.error("[members] audit write failed for %s: %s", action, exc)


async def list_members(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List active members of this workspace with their roles."""
    try:
        from core.models.core import User
        from core.workspaces.models import WorkspaceMember

        members = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.is_active == True,  # noqa: E712
            )
            .all()
        )

        out = []
        for m in members:
            user = db.query(User).filter(User.id == m.user_id).first()
            out.append({
                "member_id": m.id,
                "user_id": m.user_id,
                "email": user.email if user else None,
                "name": user.name if user else None,
                "role": m.role,
                "joined_at": m.joined_at.isoformat() if m.joined_at else None,
            })

        return {"success": True, "members": out, "count": len(out)}
    except Exception as exc:
        logger.error("[members] list_members failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def invite_member(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Invite a new member — delegates to the canonical router-extracted flow."""
    email = (params.get("email") or "").strip()
    role = params.get("role") or "viewer"
    if not email:
        return {"success": False, "error": "email is required"}

    try:
        inviter_internal_id = _workspace_owner_user_id(db, workspace_id)
        if inviter_internal_id is None:
            return {
                "success": False,
                "error": "No active workspace owner found to attribute the invitation to.",
            }

        from core.workspaces.invitations import invite_member_to_workspace

        invitation = await invite_member_to_workspace(
            db=db,
            workspace_id=str(workspace_id),
            email=email,
            role=role,
            inviter_internal_id=inviter_internal_id,
        )

        # The token is the invite secret (embedded in the Clerk email link) —
        # it must never be surfaced into the LLM context.
        return {
            "success": True,
            "invitation": {
                "id": invitation.id,
                "email": invitation.email,
                "role": invitation.role,
                "status": "pending",
                "expires_at": invitation.expires_at.isoformat() if invitation.expires_at else None,
            },
            "message": f"Invitation email sent to {email} (role: {role}).",
        }
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        db.rollback()
        logger.error("[members] invite_member failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def set_member_role(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Change a member's workspace role. Owner's role can never be changed."""
    member_id = params.get("member_id")
    new_role = params.get("role")
    if member_id is None or not new_role:
        return {"success": False, "error": "member_id and role are required"}
    try:
        member_id = int(member_id)
    except (TypeError, ValueError):
        return {"success": False, "error": f"member_id must be an integer, got {member_id!r}"}

    try:
        from core.workspaces.models import WorkspaceMember
        from core.workspaces.permissions import WorkspaceRole

        member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.id == member_id,
                WorkspaceMember.workspace_id == workspace_id,
            )
            .first()
        )
        if not member:
            return {"success": False, "error": "Member not found in this workspace"}

        if member.role == WorkspaceRole.OWNER.value:
            return {
                "success": False,
                "error": "Cannot change the workspace owner's role. Transfer ownership instead.",
            }

        valid_roles = [r.value for r in WorkspaceRole if r != WorkspaceRole.OWNER]
        if new_role not in valid_roles:
            return {"success": False, "error": f"Invalid role: {new_role}. Must be one of {valid_roles}"}

        old_role = member.role
        member.role = new_role
        db.commit()

        _audit(
            db, workspace_id, _workspace_owner_user_id(db, workspace_id),
            "member:role_changed", resource_id=member.id,
            details={"old_role": old_role, "new_role": new_role},
        )

        return {"success": True, "member_id": member.id, "old_role": old_role, "new_role": new_role}
    except Exception as exc:
        db.rollback()
        logger.error("[members] set_member_role failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def remove_member(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Deactivate a member's workspace membership. The owner can never be removed."""
    member_id = params.get("member_id")
    if member_id is None:
        return {"success": False, "error": "member_id is required"}
    try:
        member_id = int(member_id)
    except (TypeError, ValueError):
        return {"success": False, "error": f"member_id must be an integer, got {member_id!r}"}

    try:
        from core.workspaces.models import WorkspaceMember
        from core.workspaces.permissions import WorkspaceRole

        member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.id == member_id,
                WorkspaceMember.workspace_id == workspace_id,
            )
            .first()
        )
        if not member:
            return {"success": False, "error": "Member not found in this workspace"}

        if member.role == WorkspaceRole.OWNER.value:
            return {"success": False, "error": "Cannot remove the workspace owner"}

        member.is_active = False
        db.commit()

        _audit(
            db, workspace_id, _workspace_owner_user_id(db, workspace_id),
            "member:removed", resource_id=member.id,
            details={"removed_user_id": member.user_id},
        )

        return {"success": True, "member_id": member.id, "removed_user_id": member.user_id}
    except Exception as exc:
        db.rollback()
        logger.error("[members] remove_member failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
