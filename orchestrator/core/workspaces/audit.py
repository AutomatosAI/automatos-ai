from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from core.database.base import Base
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class AuditLog(Base):
    """Track all important actions within a workspace."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # What happened
    action = Column(String(100), nullable=False)  # 'agent:created', 'member:invited', etc.
    resource_type = Column(String(50), nullable=True)  # 'agent', 'workflow', etc.
    resource_id = Column(String(255), nullable=True)  # ID of affected resource
    resource_name = Column(String(255), nullable=True)  # Human-readable name
    
    # Additional context
    details = Column(JSONB, default={})  # Any extra info
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())

class AuditService:
    """Service for creating audit log entries."""

    def __init__(self, db):
        self.db = db

    def log(
        self,
        workspace_id: str,
        user_id: int,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Create an audit log entry (sync)."""
        try:
            entry = AuditLog(
                workspace_id=workspace_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                resource_name=resource_name,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self.db.add(entry)
            self.db.commit()
            logger.info("Audit: %s by user %s in workspace %s", action, user_id, workspace_id)
            return entry
        except Exception:
            self.db.rollback()
            logger.exception("Failed to create audit log")
            return None
