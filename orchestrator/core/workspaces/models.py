from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from core.database.base import Base


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member_workspace_user"),)

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(50), default="member")  # 'owner' | 'member' | 'admin' | 'editor' | 'viewer'

    invited_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    invited_at = Column(DateTime, nullable=True)
    joined_at = Column(DateTime, nullable=True)

    is_active = Column(Boolean, default=True)
    
    # Relationships will be defined in core/models/workspaces.py or core/models/core.py where other relationships are
    # or we can define them here if we import User/Workspace.
    # To keep it simple and avoid circular imports, we just define the columns here and
    # let SQLAlchemy registry handle string-based relationships if defined elsewhere,
    # or define them here if safe.
