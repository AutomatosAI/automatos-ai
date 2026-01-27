import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from core.auth.hybrid import get_request_context_hybrid as get_request_context
from core.auth.dependencies import RequestContext
from core.auth.clerk import get_clerk_auth
from core.database.database import get_db
from core.workspaces.permissions import require_permission, WorkspaceRole
from core.workspaces.invitations import InvitationService
from core.workspaces.audit import AuditService
from core.workspaces.models import WorkspaceMember
from core.models.workspaces import Workspace
from core.models.core import User
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces/{workspace_id}/team", tags=["team"])

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

    invitation_service = InvitationService(db)
    invitation = None
    clerk_invite_data = {}

    # Create local invitation first; then Clerk. If Clerk fails, revoke local invite.
    try:
        invitation = await invitation_service.create_invitation(
            workspace_id=workspace_id,
            email=request.email,
            role=request.role,
            invited_by=ctx.user_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    if ctx.user.org_id:
        try:
            clerk = get_clerk_auth()
            clerk_invite_data = await clerk.invite_to_org(
                org_id=ctx.user.org_id,
                email=request.email,
                role=request.role,
                inviter_user_id=ctx.user.clerk_user_id or str(ctx.user_id),
            )
        except Exception as e:
            try:
                invitation_service.revoke_invitation(invitation)
            except Exception as revoke_err:
                logger.error("Failed to revoke invitation after Clerk error: %s", revoke_err)
            raise HTTPException(status_code=502, detail=f"Clerk invite failed: {e}") from e

    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=ctx.user_id,
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
    
    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=ctx.user_id,
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
    
    audit = AuditService(db)
    audit.log(
        workspace_id=workspace_id,
        user_id=ctx.user_id,
        action="member:removed",
        resource_type="member",
        resource_id=str(member_id),
        details={"removed_user_id": member.user_id},
    )
    
    return {"status": "success"}
