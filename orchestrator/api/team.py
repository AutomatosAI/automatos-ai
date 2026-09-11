import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from core.auth.hybrid import get_request_context_hybrid as get_request_context
from core.auth.dependencies import RequestContext
from core.auth.actor import resolve_internal_user_id
from core.auth.clerk import get_clerk_auth
from core.database.database import get_db
from core.auth.workspace_permission import require_workspace_permission, workspace_permission_granted
from modules.policy.roles import WorkspaceRole
from core.workspaces.invitations import WorkspaceInvitation, invite_member_to_workspace
from core.workspaces.audit import AuditService
from core.workspaces.models import WorkspaceMember
from core.models.workspaces import Workspace
from core.models.composio_cache import ToolExecutionLog
from core.models.core import Chat, User
from core.utils.timestamps import utc_iso
from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces/{workspace_id}/team", tags=["team"])

# Public router for accept-invitation — uses Clerk JWT directly without going
# through workspace resolution (the invitee may not yet be a member of any workspace).
public_router = APIRouter(prefix="/api/team", tags=["team"])


# The canonical resolver lives in core.auth.actor; the module-level name is kept
# so every call site in this file (and its tests) reads unchanged.
_resolve_internal_user_id = resolve_internal_user_id

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
    # 2026-09-11 — activity THIS workspace already records, surfaced per member.
    # Field additions only: same route, so the route manifest is unchanged.
    # Timestamps carry an explicit UTC offset (core/utils/timestamps) — the
    # page renders elapsed time, which an offset-less string gets wrong by the
    # viewer's own UTC offset.
    #
    # Deliberately NOT here: ``users.last_sign_in``. That stamp is platform-wide
    # (core/auth/hybrid stamps every authenticated request, whichever workspace
    # it was for), so surfacing it on a per-workspace page would tell workspace
    # A's admins when a shared user was last active in workspace B. The team
    # page reports what happened HERE; the super-admin console keeps the
    # cross-tenant view.
    #
    # The three activity fields are shown only to callers holding
    # ``audit:view`` (owner + admin in modules/policy/roles.py): per-teammate
    # recency and volume is audit-class telemetry, and ``members:read`` is held
    # by every role down to ``viewer``. For everyone else they are ``None`` —
    # distinct from ``0``, which means "visible, and nothing happened".
    last_active_at: Optional[str] = None   # max(last chat HERE, last tool run HERE)
    tool_runs_30d: Optional[int] = None    # tool_execution_logs in THIS workspace, last 30 days
    chats_30d: Optional[int] = None        # chats in THIS workspace, last 30 days
    invited_by_email: Optional[str] = None  # roster metadata, like joined_at: every member sees it
    
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

ACTIVITY_WINDOW_DAYS = 30
# Owner + admin only (modules/policy/roles.py). Reused rather than minted: the
# activity columns answer "who did what, when", which is what audit:view guards.
ACTIVITY_PERMISSION = "audit:view"


def _activity_by_user(db, user_col, ts_col, ws_col, workspace_id, user_ids, since):
    """``{user_id: (count_in_window, latest_ever)}`` for ONE activity table.

    Scoped to the requested workspace on purpose — a member's activity in some
    OTHER workspace must never show up on this team page. The count is
    windowed (``since``); the latest is all-time, so "last active" does not go
    blank the day the window rolls past someone's last action. One grouped
    query per table, never one per member.
    """
    rows = (
        db.query(
            user_col,
            func.count().filter(ts_col >= since),
            func.max(ts_col),
        )
        .filter(ws_col == workspace_id, user_col.in_(user_ids))
        .group_by(user_col)
        .all()
    )
    return {uid: (int(count or 0), latest) for uid, count, latest in rows}


@router.get(
    "/members",
    response_model=List[TeamMemberResponse],
    dependencies=[Depends(require_workspace_permission("members:read"))],
)
async def list_team_members(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """List all members of a workspace."""
    
    members = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.is_active == True
    ).all()
    if not members:
        return []

    member_ids = {m.user_id for m in members}
    inviter_ids = {m.invited_by for m in members if m.invited_by is not None}
    # ONE users query covers the members and whoever invited them — this used to
    # be a query per member.
    users = db.query(User).filter(User.id.in_(member_ids | inviter_ids)).all()
    user_by_id = {u.id: u for u in users}

    # Editors and viewers get the roster and nothing else — the aggregate
    # queries are not even issued for them.
    show_activity = workspace_permission_granted(db, ctx, ACTIVITY_PERMISSION)
    chats: dict = {}
    tools: dict = {}
    if show_activity:
        since = datetime.utcnow() - timedelta(days=ACTIVITY_WINDOW_DAYS)
        chats = _activity_by_user(
            db, Chat.user_id, Chat.created_at, Chat.workspace_id, workspace_id, member_ids, since,
        )
        tools = _activity_by_user(
            db, ToolExecutionLog.user_id, ToolExecutionLog.executed_at,
            ToolExecutionLog.workspace_id, workspace_id, member_ids, since,
        )

    response = []
    for m in members:
        user = user_by_id.get(m.user_id)
        if not user:
            continue  # a member row whose users row is gone — skipped before too
        if show_activity:
            chat_count, chat_last = chats.get(m.user_id, (0, None))
            tool_count, tool_last = tools.get(m.user_id, (0, None))
            # Both naive-UTC DateTime columns, so they compare directly. The
            # global sign-in stamp is left out on purpose — see TeamMemberResponse.
            seen = [t for t in (chat_last, tool_last) if t is not None]
            last_active = utc_iso(max(seen)) if seen else None
        else:
            chat_count = tool_count = None
            last_active = None
        inviter = user_by_id.get(m.invited_by) if m.invited_by is not None else None
        response.append(TeamMemberResponse(
            id=m.id,
            user_id=m.user_id,
            email=user.email,
            # NULL for every Clerk-provisioned user today (names were never
            # synced into users.name); the page derives a display label. Not
            # invented here.
            name=user.name,
            role=m.role,
            joined_at=m.joined_at.isoformat() if m.joined_at else None,
            last_active_at=last_active,
            tool_runs_30d=tool_count,
            chats_30d=chat_count,
            invited_by_email=inviter.email if inviter else None,
        ))

    return response

@router.post(
    "/invite",
    response_model=InvitationResponse,
    dependencies=[Depends(require_workspace_permission("members:invite"))],
)
async def invite_member(
    workspace_id: str,
    request: InviteMemberRequest,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db)
):
    """Invite a new member to the workspace via Clerk."""

    inviter_internal_id = _resolve_internal_user_id(db, ctx)
    if not inviter_internal_id:
        raise HTTPException(401, "Inviter user not found in database")

    try:
        invitation = await invite_member_to_workspace(
            db=db,
            workspace_id=workspace_id,
            email=request.email,
            role=request.role,
            inviter_internal_id=inviter_internal_id,
            inviter_email=ctx.user.email if ctx.user else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return InvitationResponse(
        id=invitation.id,
        email=invitation.email,
        role=invitation.role,
        status="pending",
        expires_at=invitation.expires_at.isoformat(),
        created_at=invitation.created_at.isoformat()
    )

@router.patch(
    "/members/{member_id}/role",
    dependencies=[Depends(require_workspace_permission("members:change_role"))],
)
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

@router.delete(
    "/members/{member_id}",
    dependencies=[Depends(require_workspace_permission("members:remove"))],
)
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

@router.get(
    "/invitations",
    response_model=List[InvitationResponse],
    dependencies=[Depends(require_workspace_permission("members:invite"))],
)
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


@router.delete(
    "/invitations/{invitation_id}",
    dependencies=[Depends(require_workspace_permission("members:invite"))],
)
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
