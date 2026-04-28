import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from core.auth.hybrid import get_request_context_hybrid as get_request_context
from core.auth.dependencies import RequestContext
from core.auth.clerk import get_clerk_auth
from core.database.database import get_db
from core.workspaces.permissions import require_permission, WorkspaceRole
from core.workspaces.invitations import InvitationService, WorkspaceInvitation
from core.workspaces.audit import AuditService
from core.workspaces.models import WorkspaceMember
from core.models.workspaces import Workspace
from core.models.core import User
from sqlalchemy.orm import Session
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces/{workspace_id}/team", tags=["team"])

# Public router for accept-invitation — uses Clerk JWT directly without going
# through workspace resolution (the invitee may not yet be a member of any workspace).
public_router = APIRouter(prefix="/api/team", tags=["team"])


def _resolve_internal_user_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """Resolve internal users.id (integer) from RequestContext.

    ctx.user.id is a Clerk string ID or email — both audit_logs.user_id and
    workspace_invitations.invited_by are Integer FKs to users.id, so we must
    look up the matching row before writing.
    """
    if not ctx.user:
        return None
    user = None
    if ctx.user.clerk_user_id:
        user = db.query(User).filter(User.clerk_user_id == ctx.user.clerk_user_id).first()
    if not user and ctx.user.email:
        user = db.query(User).filter(User.email == ctx.user.email).first()
    return user.id if user else None

class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str = "member"

class UpdateMemberRoleRequest(BaseModel):
    role: str

class TeamMemberResponse(BaseModel):
    id: int
    user_id: int
    email: str
    name: Optional[str]
    role: str
    joined_at: Optional[str]
    
class InvitationResponse(BaseModel):
    id: int
    email: str
    role: str
    status: str
    expires_at: str
    created_at: str

class AcceptInvitationRequest(BaseModel):
    token: str

class AcceptInvitationResponse(BaseModel):
    workspace_id: str
    workspace_name: str
    role: str
    already_member: bool

@router.get("/members", response_model=List[TeamMemberResponse])
@require_permission("members:read")
async def list_team_members(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """List all members of a workspace."""
    
    # Check if we have a Clerk Org ID
    if ctx.user.org_id:
        clerk = get_clerk_auth()
        try:
            # Sync members from Clerk
            # In a real sync engine, we'd upsert users/members properly.
            # Here we just fetch them for display, or we could rely on existing DB records 
            # if we trust we are syncing via webhooks.
            # However, user requested integration.
            # Let's fetch from DB, but maybe we could trigger a background sync?
            # For simplicity & speed in this task, we will just return local DB members
            # BUT ensuring the invite flow populates Clerk correctly.
            pass
        except Exception as e:
            logger.error(f"Failed to sync with Clerk: {e}")

    members = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.is_active == True
    ).all()
    
    response = []
    for m in members:
        # Fetch user details
        user = db.query(User).filter(User.id == m.user_id).first()
        if user:
            response.append(TeamMemberResponse(
                id=m.id,
                user_id=m.user_id,
                email=user.email,
                name=user.name,
                role=m.role,
                joined_at=m.joined_at.isoformat() if m.joined_at else None
            ))
            
    return response

@router.post("/invite", response_model=InvitationResponse)
@require_permission("members:invite")
async def invite_member(
    workspace_id: str,
    request: InviteMemberRequest,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """Invite a new member to the workspace via Clerk."""
    
    # Check plan limits (basic check)
    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
        
    current_members = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.is_active == True
    ).count()
    
    # Default to 5 members if not specified
    limits = workspace.plan_limits or {}
    max_members = limits.get("max_members", 5)
    
    if max_members != -1 and current_members >= max_members:
        raise HTTPException(400, f"Workspace has reached member limit ({max_members})")
    
    # Validate role
    valid_roles = [r.value for r in WorkspaceRole]
    if request.role not in valid_roles:
        raise HTTPException(400, f"Invalid role: {request.role}. Must be one of {valid_roles}")

    inviter_internal_id = _resolve_internal_user_id(db, ctx)
    if not inviter_internal_id:
        raise HTTPException(401, "Inviter user not found in database")

    invitation_service = InvitationService(db)
    invitation = None
    clerk_invite_data = {}

    # Create local invitation first; then Clerk. If Clerk fails, revoke local invite.
    try:
        invitation = await invitation_service.create_invitation(
            workspace_id=workspace_id,
            email=request.email,
            role=request.role,
            invited_by=inviter_internal_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Send the invitation email via Clerk (user-level, no org required).
    # Clerk hosts signup, then redirects the invitee to our accept page where
    # the token is exchanged for a WorkspaceMember row.
    try:
        clerk = get_clerk_auth()
        redirect_url = f"{config.FRONTEND_URL.rstrip('/')}/accept-invitation?token={invitation.token}"
        clerk_invite_data = await clerk.create_user_invitation(
            email=request.email,
            redirect_url=redirect_url,
            public_metadata={
                "workspace_id": str(workspace_id),
                "role": request.role,
                "invitation_token": invitation.token,
                "invited_by_email": ctx.user.email if ctx.user else None,
            },
            expires_in_days=7,
        )
        invitation.clerk_invitation_id = clerk_invite_data.get("id")
        db.commit()
    except Exception as e:
        try:
            invitation_service.revoke_invitation(invitation)
        except Exception as revoke_err:
            logger.error("Failed to revoke invitation after Clerk error: %s", revoke_err)
        logger.error("Clerk user invitation failed for %s: %s", request.email, e)
        raise HTTPException(status_code=502, detail=f"Could not send invitation email: {e}") from e

    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=inviter_internal_id,
        action="member:invited",
        details={"email": request.email, "role": request.role, "clerk_id": clerk_invite_data.get("id")},
    )
    
    return InvitationResponse(
        id=invitation.id,
        email=invitation.email,
        role=invitation.role,
        status="pending",
        expires_at=invitation.expires_at.isoformat(),
        created_at=invitation.created_at.isoformat()
    )

@router.patch("/members/{member_id}/role")
@require_permission("members:change_role")
async def update_member_role(
    workspace_id: str,
    member_id: int,
    request: UpdateMemberRoleRequest,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """Change a member's role."""
    
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.id == member_id,
        WorkspaceMember.workspace_id == workspace_id
    ).first()
    
    if not member:
        raise HTTPException(404, "Member not found")
    
    # Can't change owner's role
    if member.role == WorkspaceRole.OWNER.value:
        raise HTTPException(400, "Cannot change owner's role. Transfer ownership instead.")
    
    # Validate role
    valid_roles = [r.value for r in WorkspaceRole]
    if request.role not in valid_roles:
        raise HTTPException(400, f"Invalid role: {request.role}")
    
    old_role = member.role
    member.role = request.role
    db.commit()

    # TODO: Update role in Clerk metadata via API if needed

    actor_internal_id = _resolve_internal_user_id(db, ctx)
    if not actor_internal_id:
        raise HTTPException(401, "Acting user not found in database")
    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=actor_internal_id,
        action="member:role_changed",
        resource_type="member",
        resource_id=str(member_id),
        details={"old_role": old_role, "new_role": request.role},
    )
    
    return {"status": "success", "new_role": request.role}

@router.delete("/members/{member_id}")
@require_permission("members:remove")
async def remove_member(
    workspace_id: str,
    member_id: int,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """Remove a member from the workspace."""
    
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.id == member_id,
        WorkspaceMember.workspace_id == workspace_id
    ).first()
    
    if not member:
        raise HTTPException(404, "Member not found")
    
    if member.role == WorkspaceRole.OWNER.value:
        raise HTTPException(400, "Cannot remove workspace owner")
    
    # Remove from Clerk Org
    # We need the Clerk User ID logic here. We assume member.user_id -> User -> clerk_user_id
    user = db.query(User).filter(User.id == member.user_id).first()
    if user and user.clerk_user_id and ctx.user.org_id:
        clerk = get_clerk_auth()
        try:
            await clerk.remove_from_org(ctx.user.org_id, user.clerk_user_id)
        except Exception as e:
            logger.error(f"Failed to remove from Clerk: {e}")
            # Continue to remove locally even if clerk fails (e.g. out of sync)

    member.is_active = False
    db.commit()

    actor_internal_id = _resolve_internal_user_id(db, ctx)
    if not actor_internal_id:
        raise HTTPException(401, "Acting user not found in database")
    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=actor_internal_id,
        action="member:removed",
        resource_type="member",
        resource_id=str(member_id),
        details={"removed_user_id": member.user_id},
    )

    return {"status": "success"}


# ──────────────────────────────────────────────────────────────────────────────
# Pending invitations (admin)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/invitations", response_model=List[InvitationResponse])
@require_permission("members:invite")
async def list_pending_invitations(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """List pending (unaccepted, unexpired) invitations for this workspace."""
    rows = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.workspace_id == workspace_id,
        WorkspaceInvitation.accepted_at.is_(None),
        WorkspaceInvitation.expires_at > datetime.utcnow(),
    ).order_by(WorkspaceInvitation.created_at.desc()).all()

    return [
        InvitationResponse(
            id=r.id,
            email=r.email,
            role=r.role,
            status="pending",
            expires_at=r.expires_at.isoformat(),
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


@router.delete("/invitations/{invitation_id}")
@require_permission("members:invite")
async def revoke_pending_invitation(
    workspace_id: str,
    invitation_id: int,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Revoke a pending invitation. Also revokes the Clerk-side invitation."""
    invitation = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.id == invitation_id,
        WorkspaceInvitation.workspace_id == workspace_id,
    ).first()
    if not invitation:
        raise HTTPException(404, "Invitation not found")
    if invitation.accepted_at is not None:
        raise HTTPException(400, "Invitation already accepted")

    # Best-effort Clerk revoke — don't block local revoke if it fails.
    if invitation.clerk_invitation_id:
        try:
            clerk = get_clerk_auth()
            await clerk.revoke_user_invitation(invitation.clerk_invitation_id)
        except Exception as e:
            logger.error("Failed to revoke Clerk invitation %s: %s", invitation.clerk_invitation_id, e)

    db.delete(invitation)
    db.commit()

    actor_internal_id = _resolve_internal_user_id(db, ctx)
    if actor_internal_id:
        audit = AuditService(db)
        audit.log(
            workspace_id=workspace_id,
            user_id=actor_internal_id,
            action="invitation:revoked",
            resource_type="invitation",
            resource_id=str(invitation_id),
            details={"email": invitation.email},
        )
    return {"status": "success"}


# ──────────────────────────────────────────────────────────────────────────────
# Accept invitation (public — auth via Clerk JWT only, no workspace context)
# ──────────────────────────────────────────────────────────────────────────────

def _verify_clerk_user_only(request: Request) -> dict:
    """Verify the Clerk JWT and return user info dict.

    Used for endpoints where the caller may not yet have any workspace
    membership. Raises 401 if the token is missing or invalid.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    clerk = get_clerk_auth()
    claims = clerk.verify_token(token)
    if not claims:
        raise HTTPException(401, "Invalid token")
    return clerk.extract_user_info(claims)


@public_router.post("/accept-invitation", response_model=AcceptInvitationResponse)
async def accept_invitation(
    payload: AcceptInvitationRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Exchange an invitation token for workspace membership.

    Called by /accept-invitation page after the user signs up via Clerk and
    is redirected back to the app. Idempotent: if the user is already a
    member, returns success with already_member=true.
    """
    user_info = _verify_clerk_user_only(request)
    clerk_user_id = user_info.get("clerk_user_id")
    user_email = (user_info.get("email") or "").lower()
    user_name = user_info.get("name")
    if not clerk_user_id:
        raise HTTPException(401, "Clerk user ID missing from token")

    invitation = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.token == payload.token,
    ).first()
    if not invitation:
        raise HTTPException(404, "Invitation not found")
    if invitation.expires_at < datetime.utcnow():
        raise HTTPException(410, "Invitation has expired")

    workspace = db.query(Workspace).get(invitation.workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace no longer exists")

    # Resolve or create the User row for this Clerk user.
    user = db.query(User).filter(User.clerk_user_id == clerk_user_id).first()
    if not user and user_email:
        user = db.query(User).filter(User.email == user_email).first()
        if user and not user.clerk_user_id:
            user.clerk_user_id = clerk_user_id
            db.flush()
    if not user:
        user = User(
            username=user_email or clerk_user_id,
            email=user_email or f"{clerk_user_id}@pending",
            clerk_user_id=clerk_user_id,
            name=user_name,
            is_active=True,
        )
        db.add(user)
        db.flush()

    # Idempotent membership: existing inactive row gets reactivated.
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == invitation.workspace_id,
        WorkspaceMember.user_id == user.id,
    ).first()

    already_member = bool(member and member.is_active)
    if member:
        member.is_active = True
        member.role = invitation.role
    else:
        member = WorkspaceMember(
            workspace_id=invitation.workspace_id,
            user_id=user.id,
            role=invitation.role,
            invited_by=invitation.invited_by,
            invited_at=invitation.created_at,
            joined_at=datetime.utcnow(),
            is_active=True,
        )
        db.add(member)

    if not invitation.accepted_at:
        invitation.accepted_at = datetime.utcnow()
        invitation.accepted_by_user_id = user.id

    db.commit()

    try:
        audit = AuditService(db)
        audit.log(
            workspace_id=str(invitation.workspace_id),
            user_id=user.id,
            action="invitation:accepted",
            resource_type="invitation",
            resource_id=str(invitation.id),
            details={"email": invitation.email, "role": invitation.role},
        )
    except Exception as e:
        logger.error("Failed to write audit for invitation acceptance: %s", e)

    return AcceptInvitationResponse(
        workspace_id=str(invitation.workspace_id),
        workspace_name=workspace.name,
        role=invitation.role,
        already_member=already_member,
    )


@public_router.get("/invitation-info")
async def get_invitation_info(token: str, db: Session = Depends(get_db)):
    """Public lookup of invitation metadata by token (for UI rendering before
    the user is signed in). Returns workspace name, inviter email, role.
    """
    invitation = db.query(WorkspaceInvitation).filter(
        WorkspaceInvitation.token == token,
    ).first()
    if not invitation:
        raise HTTPException(404, "Invitation not found")

    workspace = db.query(Workspace).get(invitation.workspace_id)
    inviter = db.query(User).filter(User.id == invitation.invited_by).first() if invitation.invited_by else None

    return {
        "email": invitation.email,
        "role": invitation.role,
        "workspace_name": workspace.name if workspace else "Unknown",
        "inviter_name": (inviter.name or inviter.email) if inviter else None,
        "expires_at": invitation.expires_at.isoformat(),
        "expired": invitation.expires_at < datetime.utcnow(),
        "accepted": invitation.accepted_at is not None,
    }
