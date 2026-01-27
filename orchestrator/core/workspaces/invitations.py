from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from core.database.base import Base
# Import models to ensure ForeignKeys work
from core.models.core import User
# Note: WorkspaceMember is now in core.workspaces.models (to be created) or we assume it is in core.models.workspaces?
# The plan said "WorkspaceMember model if not present". 
# The research said it was in core/workspaces/models.py.
# But core/workspaces/models.py didn't exist before this task.
# I will need to define WorkspaceMember here or in a separate models file in this module.
# Let's check imports - I need to be careful about circular imports.
# For now, I will assume WorkspaceMember is defined in core.models.workspaces (which I viewed earlier and it only had Workspace).
# I will likely need to add WorkspaceMember to core/workspaces/models.py 
# OR add it to core/models/workspaces.py.
# The plan says "Create core/workspaces module".
# I'll define Invitation logic here, but I might need to import WorkspaceMember dynamically or definition.

import secrets
from datetime import datetime, timedelta

class WorkspaceInvitation(Base):
    """Pending invitations to join a workspace."""
    __tablename__ = "workspace_invitations"
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    
    # Invitation details
    email = Column(String(255), nullable=False)
    role = Column(String(50), default="member", nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    
    # Status tracking
    invited_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    accepted_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())

class InvitationService:
    """Handle workspace invitations."""
    
    TOKEN_EXPIRY_DAYS = 7
    
    def __init__(self, db, email_service=None):
        self.db = db
        self.email_service = email_service
    
    async def create_invitation(
        self,
        workspace_id: str,
        email: str,
        role: str,
        invited_by: int
    ) -> WorkspaceInvitation:
        """Create a new invitation and optionally send email."""
        
        # Check if already a member - Logic requires WorkspaceMember model
        # from core.workspaces.models import WorkspaceMember # Deferred import to avoid circular dependency issues if any
        # actually, I need to define WorkspaceMember somewhere first.
        
        # For now, implementing the core logic of invitation creation.
        # Membership check should be done before calling this or inside this if I can import the model.
        
        # Check for pending invitation
        pending = self.db.query(WorkspaceInvitation).filter(
            WorkspaceInvitation.workspace_id == workspace_id,
            WorkspaceInvitation.email == email,
            WorkspaceInvitation.accepted_at.is_(None),
            WorkspaceInvitation.expires_at > datetime.utcnow()
        ).first()
        
        if pending:
            raise ValueError(f"A pending invitation already exists for {email}")
        
        # Create invitation
        invitation = WorkspaceInvitation(
            workspace_id=workspace_id,
            email=email,
            role=role,
            token=secrets.token_urlsafe(32),
            invited_by=invited_by,
            expires_at=datetime.utcnow() + timedelta(days=self.TOKEN_EXPIRY_DAYS)
        )
        
        self.db.add(invitation)
        self.db.commit()
        
        # Send email (optional)
        if self.email_service:
            await self.email_service.send_invitation_email(
                email=email,
                invitation_token=invitation.token,
                workspace_name="Workspace" # Need to fetch workspace name or pass it
            )
        
        return invitation
    
    async def accept_invitation(self, token: str, user_id: int):
        """
        Accept an invitation. 
        Returns the invitation object. Caller is responsible for creating the WorkspaceMember.
        """
        
        invitation = self.db.query(WorkspaceInvitation).filter(
            WorkspaceInvitation.token == token,
            WorkspaceInvitation.accepted_at.is_(None)
        ).first()
        
        if not invitation:
            raise ValueError("Invitation not found or already used")
        
        if invitation.expires_at < datetime.utcnow():
            raise ValueError("Invitation has expired")
            
        return invitation

    def mark_accepted(self, invitation: WorkspaceInvitation, user_id: int):
        """Mark invitation as accepted."""
        invitation.accepted_at = datetime.utcnow()
        invitation.accepted_by_user_id = user_id
        self.db.commit()
