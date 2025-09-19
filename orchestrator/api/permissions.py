

"""
Permissions Management API Routes
==================================

Role-based access control for agent-tool permissions with assignment management,
security validation, and audit logging.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from database.database import get_db
from database.models import Tool, Agent, AgentToolPermission, PermissionAuditLog
from utils.logging_adapter import set_request_id
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/permissions", tags=["Permissions Management"])

# ====================================
# Pydantic Models
# ====================================

class AgentToolAssignmentRequest(BaseModel):
    agent_id: int = Field(..., description="Agent ID")
    tool_id: int = Field(..., description="Tool ID")
    environment: str = Field(..., description="Environment")
    permissions: List[str] = Field(default_factory=list, description="Specific permissions")
    expires_at: Optional[datetime] = Field(None, description="Permission expiration")

class BulkAssignmentRequest(BaseModel):
    assignments: List[AgentToolAssignmentRequest] = Field(..., description="List of assignments")
    validate_conflicts: bool = Field(default=True, description="Validate conflicts before applying")

class PermissionResponse(BaseModel):
    id: int
    agent_id: int
    agent_name: str
    agent_type: str
    tool_id: int
    tool_name: str
    tool_category: str
    environment: str
    permissions: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class PermissionMatrix(BaseModel):
    agent_types: Dict[str, List[str]]  # agent_type -> allowed tools
    current_assignments: Dict[str, List[int]]  # agent_id -> tool_ids
    conflicts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]

class PermissionConflict(BaseModel):
    agent_id: int
    agent_name: str
    agent_type: str
    tool_id: int
    tool_name: str
    conflict_type: str
    reason: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str

class AuditLogEntry(BaseModel):
    id: int
    agent_id: int
    agent_name: str
    tool_id: int
    tool_name: str
    action: str
    environment: str
    user_id: Optional[str]
    timestamp: datetime
    details: Optional[Dict[str, Any]]

# ====================================
# Permission Matrix Configuration
# ====================================

# Define which agent types can access which tools
AGENT_TOOL_PERMISSIONS = {
    "code_architect": [
        "GitHub", "GitLab", "Bitbucket", "Jira", "Linear", "Asana", 
        "Docker", "Kubernetes", "Jenkins"
    ],
    "security_expert": [
        "GitHub", "GitLab", "SonarQube", "Snyk", "OWASP ZAP",
        "Vault", "1Password", "AWS IAM"
    ],
    "performance_optimizer": [
        "DataDog", "New Relic", "Grafana", "Prometheus", "Docker",
        "Kubernetes", "AWS CloudWatch", "Azure Monitor"
    ],
    "data_analyst": [
        "Google Analytics", "Mixpanel", "Tableau", "Power BI",
        "PostgreSQL", "MongoDB", "Snowflake"
    ],
    "infrastructure_manager": [
        "AWS S3", "AWS EC2", "AWS Lambda", "Azure", "GCP",
        "Docker", "Kubernetes", "Terraform", "Ansible", "Slack"
    ],
    "custom": [
        "GitHub", "Jira", "Slack", "Gmail", "Google Calendar",
        "Trello", "Notion", "Zapier"
    ]
}

# ====================================
# Helper Functions
# ====================================

def validate_permission(agent_type: str, tool_name: str) -> bool:
    """Check if an agent type is allowed to use a specific tool"""
    allowed_tools = AGENT_TOOL_PERMISSIONS.get(agent_type, [])
    return tool_name in allowed_tools

def create_permission_audit(
    db: Session,
    agent_id: int,
    tool_id: int,
    action: str,
    environment: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Create a permission audit log entry"""
    audit_entry = PermissionAuditLog(
        agent_id=agent_id,
        tool_id=tool_id,
        action=action,
        environment=environment,
        user_id=user_id,
        details=details or {},
        timestamp=datetime.utcnow()
    )
    db.add(audit_entry)
    return audit_entry

def detect_conflicts(db: Session, assignments: List[AgentToolAssignmentRequest]) -> List[PermissionConflict]:
    """Detect conflicts in permission assignments"""
    conflicts = []
    
    for assignment in assignments:
        agent = db.query(Agent).filter(Agent.id == assignment.agent_id).first()
        tool = db.query(Tool).filter(Tool.id == assignment.tool_id).first()
        
        if not agent or not tool:
            continue
        
        # Check if agent type has permission for this tool
        if not validate_permission(agent.agent_type, tool.name):
            conflicts.append(PermissionConflict(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                tool_id=tool.id,
                tool_name=tool.name,
                conflict_type="permission_denied",
                reason=f"{agent.agent_type} agents are not authorized to use {tool.name}",
                severity="high",
                recommendation=f"Remove {tool.name} from {agent.name} or change agent type"
            ))
        
        # Check if tool is properly configured
        if not tool.is_configured:
            conflicts.append(PermissionConflict(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                tool_id=tool.id,
                tool_name=tool.name,
                conflict_type="tool_not_configured",
                reason=f"{tool.name} is not properly configured",
                severity="medium",
                recommendation=f"Configure {tool.name} before assigning to agents"
            ))
        
        # Check for environment mismatches
        if assignment.environment not in (tool.supported_environments or []):
            conflicts.append(PermissionConflict(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                tool_id=tool.id,
                tool_name=tool.name,
                conflict_type="environment_mismatch",
                reason=f"{tool.name} doesn't support {assignment.environment} environment",
                severity="high",
                recommendation=f"Use supported environments: {', '.join(tool.supported_environments or [])}"
            ))
    
    return conflicts

# ====================================
# API Endpoints
# ====================================

@router.get("/matrix", response_model=PermissionMatrix)
async def get_permission_matrix(db: Session = Depends(get_db)):
    """
    Get the complete permission matrix showing agent types, allowed tools,
    current assignments, and any conflicts.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Get current assignments
        assignments_query = db.query(
            AgentToolPermission.agent_id,
            AgentToolPermission.tool_id
        ).filter(AgentToolPermission.is_active == True)
        
        current_assignments = {}
        for agent_id, tool_id in assignments_query.all():
            agent_key = f"agent_{agent_id}"
            if agent_key not in current_assignments:
                current_assignments[agent_key] = []
            current_assignments[agent_key].append(tool_id)
        
        # Detect conflicts in current assignments
        all_assignments = []
        for agent_key, tool_ids in current_assignments.items():
            agent_id = int(agent_key.split('_')[1])
            for tool_id in tool_ids:
                all_assignments.append(AgentToolAssignmentRequest(
                    agent_id=agent_id,
                    tool_id=tool_id,
                    environment="production"  # Default for conflict detection
                ))
        
        conflicts = detect_conflicts(db, all_assignments)
        
        # Generate recommendations
        recommendations = []
        agent_types_without_tools = []
        
        for agent_type, allowed_tools in AGENT_TOOL_PERMISSIONS.items():
            agents_of_type = db.query(Agent).filter(Agent.agent_type == agent_type).count()
            if agents_of_type > 0:
                # Check if any agents of this type have tool assignments
                assigned_tools = db.query(func.count(AgentToolPermission.id)).join(
                    Agent, AgentToolPermission.agent_id == Agent.id
                ).filter(
                    Agent.agent_type == agent_type,
                    AgentToolPermission.is_active == True
                ).scalar() or 0
                
                if assigned_tools == 0:
                    agent_types_without_tools.append(agent_type)
        
        for agent_type in agent_types_without_tools:
            recommendations.append({
                "type": "missing_assignments",
                "agent_type": agent_type,
                "suggestion": f"Consider assigning tools to {agent_type} agents",
                "available_tools": AGENT_TOOL_PERMISSIONS[agent_type]
            })
        
        matrix = PermissionMatrix(
            agent_types=AGENT_TOOL_PERMISSIONS,
            current_assignments=current_assignments,
            conflicts=[conflict.dict() for conflict in conflicts],
            recommendations=recommendations
        )
        
        logger.info(f"Retrieved permission matrix with {len(current_assignments)} assignments and {len(conflicts)} conflicts")
        return matrix
        
    except Exception as e:
        logger.error(f"Error retrieving permission matrix: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve permission matrix: {str(e)}")

@router.get("/assignments", response_model=List[PermissionResponse])
async def list_assignments(
    agent_id: Optional[int] = Query(None, description="Filter by agent ID"),
    tool_id: Optional[int] = Query(None, description="Filter by tool ID"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    active_only: bool = Query(True, description="Only show active assignments"),
    db: Session = Depends(get_db)
):
    """List all agent-tool assignments with optional filtering"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        query = db.query(
            AgentToolPermission,
            Agent.name.label("agent_name"),
            Agent.agent_type.label("agent_type"),
            Tool.name.label("tool_name"),
            Tool.category.label("tool_category")
        ).join(
            Agent, AgentToolPermission.agent_id == Agent.id
        ).join(
            Tool, AgentToolPermission.tool_id == Tool.id
        )
        
        if agent_id:
            query = query.filter(AgentToolPermission.agent_id == agent_id)
        
        if tool_id:
            query = query.filter(AgentToolPermission.tool_id == tool_id)
        
        if environment:
            query = query.filter(AgentToolPermission.environment == environment)
        
        if active_only:
            query = query.filter(AgentToolPermission.is_active == True)
        
        results = query.all()
        
        assignments = []
        for permission, agent_name, agent_type, tool_name, tool_category in results:
            assignments.append(PermissionResponse(
                id=permission.id,
                agent_id=permission.agent_id,
                agent_name=agent_name,
                agent_type=agent_type,
                tool_id=permission.tool_id,
                tool_name=tool_name,
                tool_category=tool_category,
                environment=permission.environment,
                permissions=permission.permissions or [],
                is_active=permission.is_active,
                expires_at=permission.expires_at,
                created_at=permission.created_at,
                updated_at=permission.updated_at
            ))
        
        logger.info(f"Retrieved {len(assignments)} permission assignments")
        return assignments
        
    except Exception as e:
        logger.error(f"Error listing assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list assignments: {str(e)}")

@router.post("/assign")
async def assign_tool_to_agent(
    assignment: AgentToolAssignmentRequest,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    force: bool = Query(False, description="Force assignment even if conflicts exist"),
    db: Session = Depends(get_db)
):
    """Assign a tool to an agent with permission validation"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Verify agent and tool exist
        agent = db.query(Agent).filter(Agent.id == assignment.agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        tool = db.query(Tool).filter(Tool.id == assignment.tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Check for existing assignment
        existing = db.query(AgentToolPermission).filter(
            and_(
                AgentToolPermission.agent_id == assignment.agent_id,
                AgentToolPermission.tool_id == assignment.tool_id,
                AgentToolPermission.environment == assignment.environment
            )
        ).first()
        
        if existing and existing.is_active:
            raise HTTPException(
                status_code=400, 
                detail=f"Agent {agent.name} already has access to {tool.name} in {assignment.environment}"
            )
        
        # Validate permissions unless forced
        if not force:
            conflicts = detect_conflicts(db, [assignment])
            if conflicts:
                high_severity_conflicts = [c for c in conflicts if c.severity in ['high', 'critical']]
                if high_severity_conflicts:
                    conflict_messages = [f"{c.conflict_type}: {c.reason}" for c in high_severity_conflicts]
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Permission conflicts detected",
                            "conflicts": conflict_messages,
                            "hint": "Use force=true to override conflicts"
                        }
                    )
        
        # Create or reactivate assignment
        if existing:
            existing.is_active = True
            existing.permissions = assignment.permissions
            existing.expires_at = assignment.expires_at
            existing.updated_at = datetime.utcnow()
            permission = existing
        else:
            permission = AgentToolPermission(
                agent_id=assignment.agent_id,
                tool_id=assignment.tool_id,
                environment=assignment.environment,
                permissions=assignment.permissions,
                is_active=True,
                expires_at=assignment.expires_at,
                created_at=datetime.utcnow()
            )
            db.add(permission)
        
        # Create audit log
        create_permission_audit(
            db, assignment.agent_id, assignment.tool_id, "assign",
            assignment.environment, user_id,
            {
                "permissions": assignment.permissions,
                "expires_at": assignment.expires_at.isoformat() if assignment.expires_at else None,
                "forced": force
            }
        )
        
        db.commit()
        db.refresh(permission)
        
        logger.info(f"Assigned {tool.name} to {agent.name} in {assignment.environment}")
        return {
            "message": f"Successfully assigned {tool.name} to {agent.name}",
            "assignment_id": permission.id,
            "agent_name": agent.name,
            "tool_name": tool.name,
            "environment": assignment.environment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning tool to agent: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to assign tool: {str(e)}")

@router.post("/bulk-assign")
async def bulk_assign_tools(
    bulk_request: BulkAssignmentRequest,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """Bulk assign multiple tools to agents with conflict validation"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        results = {
            "successful": [],
            "failed": [],
            "conflicts": [],
            "summary": {
                "total": len(bulk_request.assignments),
                "success_count": 0,
                "error_count": 0,
                "conflict_count": 0
            }
        }
        
        # Validate conflicts if requested
        if bulk_request.validate_conflicts:
            conflicts = detect_conflicts(db, bulk_request.assignments)
            if conflicts:
                results["conflicts"] = [conflict.dict() for conflict in conflicts]
                results["summary"]["conflict_count"] = len(conflicts)
                
                # Filter out assignments with high-severity conflicts
                conflict_pairs = {(c.agent_id, c.tool_id) for c in conflicts if c.severity in ['high', 'critical']}
                filtered_assignments = [
                    a for a in bulk_request.assignments 
                    if (a.agent_id, a.tool_id) not in conflict_pairs
                ]
                
                logger.warning(f"Filtered {len(bulk_request.assignments) - len(filtered_assignments)} assignments due to conflicts")
                bulk_request.assignments = filtered_assignments
        
        # Process each assignment
        for assignment in bulk_request.assignments:
            try:
                # Individual assignment logic (similar to assign_tool_to_agent)
                agent = db.query(Agent).filter(Agent.id == assignment.agent_id).first()
                tool = db.query(Tool).filter(Tool.id == assignment.tool_id).first()
                
                if not agent or not tool:
                    results["failed"].append({
                        "assignment": assignment.dict(),
                        "error": f"Agent or tool not found"
                    })
                    continue
                
                # Check existing assignment
                existing = db.query(AgentToolPermission).filter(
                    and_(
                        AgentToolPermission.agent_id == assignment.agent_id,
                        AgentToolPermission.tool_id == assignment.tool_id,
                        AgentToolPermission.environment == assignment.environment
                    )
                ).first()
                
                if existing and existing.is_active:
                    results["failed"].append({
                        "assignment": assignment.dict(),
                        "error": "Assignment already exists"
                    })
                    continue
                
                # Create assignment
                if existing:
                    existing.is_active = True
                    existing.permissions = assignment.permissions
                    existing.expires_at = assignment.expires_at
                    existing.updated_at = datetime.utcnow()
                    permission = existing
                else:
                    permission = AgentToolPermission(
                        agent_id=assignment.agent_id,
                        tool_id=assignment.tool_id,
                        environment=assignment.environment,
                        permissions=assignment.permissions,
                        is_active=True,
                        expires_at=assignment.expires_at,
                        created_at=datetime.utcnow()
                    )
                    db.add(permission)
                
                # Create audit log
                create_permission_audit(
                    db, assignment.agent_id, assignment.tool_id, "bulk_assign",
                    assignment.environment, user_id
                )
                
                results["successful"].append({
                    "assignment": assignment.dict(),
                    "agent_name": agent.name,
                    "tool_name": tool.name
                })
                
            except Exception as e:
                results["failed"].append({
                    "assignment": assignment.dict(),
                    "error": str(e)
                })
        
        # Update summary
        results["summary"]["success_count"] = len(results["successful"])
        results["summary"]["error_count"] = len(results["failed"])
        
        db.commit()
        
        logger.info(f"Bulk assignment completed: {results['summary']['success_count']} successful, {results['summary']['error_count']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Error in bulk assignment: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process bulk assignment: {str(e)}")

@router.delete("/revoke")
async def revoke_tool_access(
    agent_id: int = Query(..., description="Agent ID"),
    tool_id: int = Query(..., description="Tool ID"),
    environment: str = Query(..., description="Environment"),
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """Revoke an agent's access to a tool"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Find the permission
        permission = db.query(AgentToolPermission).filter(
            and_(
                AgentToolPermission.agent_id == agent_id,
                AgentToolPermission.tool_id == tool_id,
                AgentToolPermission.environment == environment,
                AgentToolPermission.is_active == True
            )
        ).first()
        
        if not permission:
            raise HTTPException(
                status_code=404, 
                detail="No active permission found for this agent-tool-environment combination"
            )
        
        # Get agent and tool info for response
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        
        # Deactivate permission
        permission.is_active = False
        permission.updated_at = datetime.utcnow()
        
        # Create audit log
        create_permission_audit(
            db, agent_id, tool_id, "revoke", environment, user_id
        )
        
        db.commit()
        
        logger.info(f"Revoked {tool.name if tool else 'unknown tool'} access from {agent.name if agent else 'unknown agent'}")
        return {
            "message": f"Successfully revoked access to {tool.name if tool else 'tool'} from {agent.name if agent else 'agent'}",
            "agent_name": agent.name if agent else None,
            "tool_name": tool.name if tool else None,
            "environment": environment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking tool access: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to revoke access: {str(e)}")

@router.get("/audit", response_model=List[AuditLogEntry])
async def get_permission_audit_logs(
    agent_id: Optional[int] = Query(None, description="Filter by agent ID"),
    tool_id: Optional[int] = Query(None, description="Filter by tool ID"),
    action: Optional[str] = Query(None, description="Filter by action"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    limit: int = Query(100, description="Maximum number of logs to return"),
    db: Session = Depends(get_db)
):
    """Get permission audit logs with optional filtering"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        query = db.query(
            PermissionAuditLog,
            Agent.name.label("agent_name"),
            Tool.name.label("tool_name")
        ).join(
            Agent, PermissionAuditLog.agent_id == Agent.id
        ).join(
            Tool, PermissionAuditLog.tool_id == Tool.id
        )
        
        if agent_id:
            query = query.filter(PermissionAuditLog.agent_id == agent_id)
        
        if tool_id:
            query = query.filter(PermissionAuditLog.tool_id == tool_id)
        
        if action:
            query = query.filter(PermissionAuditLog.action == action)
        
        if environment:
            query = query.filter(PermissionAuditLog.environment == environment)
        
        results = query.order_by(PermissionAuditLog.timestamp.desc()).limit(limit).all()
        
        audit_entries = []
        for log, agent_name, tool_name in results:
            audit_entries.append(AuditLogEntry(
                id=log.id,
                agent_id=log.agent_id,
                agent_name=agent_name,
                tool_id=log.tool_id,
                tool_name=tool_name,
                action=log.action,
                environment=log.environment,
                user_id=log.user_id,
                timestamp=log.timestamp,
                details=log.details
            ))
        
        logger.info(f"Retrieved {len(audit_entries)} permission audit log entries")
        return audit_entries
        
    except Exception as e:
        logger.error(f"Error retrieving permission audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}")

# ====================================
# Health Check
# ====================================

@router.get("/health")
async def permissions_health_check():
    """Health check endpoint for permissions management system"""
    return {
        "status": "healthy",
        "service": "permissions-management",
        "timestamp": datetime.utcnow().isoformat(),
        "rbac": "active",
        "audit_logging": "enabled",
        "version": "1.0.0"
    }
