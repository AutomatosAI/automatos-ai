from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from core.database.base import Base
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class AuditLog(Base):
    """Track all important actions within a workspace.

    PRD-181 S1 (EU-AI-Act Art.12): this table is the per-tenant record-keeping
    substrate for *every* policy verdict, not just human-initiated member
    actions. Because agent / heartbeat / scheduled tool calls have **no human
    principal**, ``user_id`` is nullable — a NULL user is a non-human actor,
    and ``actor_type`` + ``details.actor_type`` carry the true actor (agent id,
    system). Recording those calls is the whole point of Art.12; the old
    pattern of skipping the audit when no user was present would leave the
    autonomous surfaces unlogged.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    # Nullable (PRD-181 S1): a non-human actor (agent / system / scheduled) has
    # no user row. SET NULL on user deletion so a GDPR user-erasure never
    # orphans an audit row via a dangling FK (the row survives with actor
    # context preserved in `details`).
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Who acted — 'user' | 'agent' | 'system' (PRD-181 S1). The fine-grained
    # identity (agent id, tool trace) lives in `details`.
    actor_type = Column(String(20), nullable=False, default="user", server_default="user")

    # What happened
    action = Column(String(100), nullable=False)  # 'agent:created', 'policy:deny', etc.
    resource_type = Column(String(50), nullable=True)  # 'agent', 'tool', etc.
    resource_id = Column(String(255), nullable=True)  # ID of affected resource
    resource_name = Column(String(255), nullable=True)  # Human-readable name

    # Additional context
    details = Column(JSONB, default={})  # Any extra info
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        # PRD-196 S3: the audit-log read view (workspace-scoped, newest-first)
        # and the S5 retention sweep (created_at < cutoff) both scan by these two
        # columns. Migration prd196_audit_logs_ws_created_idx creates the same
        # index in prod; declared here so create_all-based test DBs match.
        Index("ix_audit_logs_workspace_created", "workspace_id", "created_at"),
    )

class AuditService:
    """Service for creating audit log entries."""

    def __init__(self, db):
        self.db = db

    def log(
        self,
        workspace_id: str,
        user_id: Optional[int],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        actor_type: str = "user",
    ):
        """Create an audit log entry (sync).

        ``user_id`` may be ``None`` for a non-human actor (agent / system /
        scheduled). ``actor_type`` records which — the fine-grained identity
        (agent id, tool trace) belongs in ``details``. Caller does NOT skip the
        write when there is no human principal (PRD-181 S1 / Art.12).
        """
        try:
            entry = AuditLog(
                workspace_id=workspace_id,
                user_id=user_id,
                actor_type=actor_type,
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
            logger.info(
                "Audit: %s by %s=%s in workspace %s",
                action, actor_type, user_id, workspace_id,
            )
            return entry
        except Exception:
            self.db.rollback()
            logger.exception("Failed to create audit log")
            return None
