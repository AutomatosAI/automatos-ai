"""
MCP Tools API
=============

Phase 3: Skills & Tools Integration
Manage MCP (Model Context Protocol) tools that agents can use.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func
import logging

from database.database import get_db
from database.models import Agent, MCPTool, AgentToolAssignment
# Import from models.py file (not models/ directory)
import sys
from pathlib import Path
models_file = Path(__file__).parent.parent / "models.py"
sys.path.insert(0, str(models_file.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("models_file", models_file)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)

MCPToolCreate = models_module.MCPToolCreate
MCPToolUpdate = models_module.MCPToolUpdate
MCPToolResponse = models_module.MCPToolResponse
AgentToolAssignmentCreate = models_module.AgentToolAssignmentCreate
AgentToolAssignmentResponse = models_module.AgentToolAssignmentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp-tools", tags=["MCP Tools"])

# ===================================================================
# MCP TOOLS CRUD
# ===================================================================

@router.get("/")
async def list_mcp_tools(
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    search: Optional[str] = Query(None, description="Search in name, description"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all MCP tools with optional filters"""
    try:
        query = db.query(MCPTool)
        
        # Apply filters
        if status:
            query = query.filter(MCPTool.status == status)
        if category:
            query = query.filter(MCPTool.category == category)
        if provider:
            query = query.filter(MCPTool.provider == provider)
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    MCPTool.name.ilike(search_pattern),
                    MCPTool.description.ilike(search_pattern),
                    MCPTool.provider.ilike(search_pattern)
                )
            )
        
        tools = query.offset(skip).limit(limit).all()
        
        # Manually convert to dict to handle metadata field
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "mcp_server_url": t.mcp_server_url,
                "capabilities": t.capabilities or {},
                "credentials_schema": t.credentials_schema or {},
                "status": t.status,
                "provider": t.provider,
                "version": t.version,
                "icon": t.icon,
                "category": t.category,
                "tags": t.tags or [],
                "metadata": t.tool_metadata or {},  # Map tool_metadata -> metadata
                "created_at": t.created_at,
                "updated_at": t.updated_at,
                "created_by": t.created_by
            }
            for t in tools
        ]
        
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}")
async def get_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    """Get single MCP tool by ID"""
    tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    return {
        "id": tool.id,
        "name": tool.name,
        "description": tool.description,
        "mcp_server_url": tool.mcp_server_url,
        "capabilities": tool.capabilities or {},
        "credentials_schema": tool.credentials_schema or {},
        "status": tool.status,
        "provider": tool.provider,
        "version": tool.version,
        "icon": tool.icon,
        "category": tool.category,
        "tags": tool.tags or [],
        "metadata": tool.tool_metadata or {},
        "created_at": tool.created_at,
        "updated_at": tool.updated_at,
        "created_by": tool.created_by
    }

@router.post("/", response_model=MCPToolResponse)
async def create_mcp_tool(
    tool_data: MCPToolCreate,
    db: Session = Depends(get_db)
):
    """Create new MCP tool"""
    try:
        # Check if tool with same name exists
        existing = db.query(MCPTool).filter(MCPTool.name == tool_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Tool with this name already exists")
        
        # Create tool
        tool = MCPTool(
            name=tool_data.name,
            description=tool_data.description,
            mcp_server_url=tool_data.mcp_server_url,
            capabilities=tool_data.capabilities or {},
            credentials_schema=tool_data.credentials_schema or {},
            status=tool_data.status or 'active',
            provider=tool_data.provider,
            version=tool_data.version,
            icon=tool_data.icon,
            category=tool_data.category,
            tags=tool_data.tags or [],
            tool_metadata=tool_data.tool_metadata or {},
            created_by="api"
        )
        
        db.add(tool)
        db.commit()
        db.refresh(tool)
        
        logger.info(f"Created MCP tool: {tool.name} (ID: {tool.id})")
        return tool
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating MCP tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{tool_id}", response_model=MCPToolResponse)
async def update_mcp_tool(
    tool_id: int,
    tool_data: MCPToolUpdate,
    db: Session = Depends(get_db)
):
    """Update MCP tool"""
    try:
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Update fields
        update_data = tool_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(tool, field, value)
        
        db.commit()
        db.refresh(tool)
        
        logger.info(f"Updated MCP tool: {tool.name} (ID: {tool.id})")
        return tool
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating MCP tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{tool_id}")
async def delete_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    """Delete MCP tool"""
    try:
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Check if tool is assigned to any agents
        assignments = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.tool_id == tool_id,
            AgentToolAssignment.enabled == True
        ).count()
        
        if assignments > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete tool. It is assigned to {assignments} agent(s). Disable assignments first."
            )
        
        db.delete(tool)
        db.commit()
        
        logger.info(f"Deleted MCP tool: {tool.name} (ID: {tool.id})")
        return {"status": "success", "message": f"Tool '{tool.name}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting MCP tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{tool_id}/test")
async def test_mcp_tool_connection(
    tool_id: int,
    test_params: Dict[str, Any] = Body(default={}),
    db: Session = Depends(get_db)
):
    """Test MCP tool connection"""
    try:
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # TODO: Implement actual MCP server connection test
        # For now, return mock success if tool has valid URL
        if not tool.mcp_server_url:
            raise HTTPException(status_code=400, detail="Tool has no MCP server URL configured")
        
        return {
            "status": "success",
            "message": f"Connection to {tool.name} successful",
            "tool_id": tool.id,
            "tool_name": tool.name,
            "mcp_server_url": tool.mcp_server_url,
            "capabilities_count": len(tool.capabilities.get("methods", [])) if tool.capabilities else 0,
            "test_mode": True  # Indicates this is a mock test
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing MCP tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories/list")
async def list_tool_categories(db: Session = Depends(get_db)):
    """Get list of all tool categories with counts"""
    try:
        # Get distinct categories with counts
        categories = db.query(
            MCPTool.category,
            func.count(MCPTool.id).label('count')
        ).filter(
            MCPTool.category.isnot(None)
        ).group_by(
            MCPTool.category
        ).all()
        
        return [
            {"name": cat, "count": count}
            for cat, count in categories
        ]
        
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assignments", response_model=List[AgentToolAssignmentResponse])
async def get_all_tool_assignments(
    enabled_only: bool = Query(True, description="Only return enabled assignments"),
    db: Session = Depends(get_db)
):
    """Get all tool assignments across all agents"""
    try:
        query = db.query(AgentToolAssignment).options(
            joinedload(AgentToolAssignment.tool),
            joinedload(AgentToolAssignment.agent)
        )
        
        if enabled_only:
            query = query.filter(AgentToolAssignment.enabled == True)
        
        assignments = query.all()
        return assignments
        
    except Exception as e:
        logger.error(f"Error getting all tool assignments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_tools_stats(db: Session = Depends(get_db)):
    """Get tool statistics summary"""
    try:
        total = db.query(func.count(MCPTool.id)).scalar()
        active = db.query(func.count(MCPTool.id)).filter(MCPTool.status == 'active').scalar()
        assigned = db.query(func.count(func.distinct(AgentToolAssignment.tool_id))).filter(
            AgentToolAssignment.enabled == True
        ).scalar()
        
        return {
            "total_tools": total,
            "active_tools": active,
            "assigned_tools": assigned,
            "unassigned_tools": total - assigned
        }
        
    except Exception as e:
        logger.error(f"Error getting tool stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
# AGENT-TOOL ASSIGNMENT ENDPOINTS
# ===================================================================

@router.get("/agents/{agent_id}/tools", response_model=List[AgentToolAssignmentResponse])
async def get_agent_tools(
    agent_id: int,
    enabled_only: bool = Query(True, description="Only return enabled tools"),
    db: Session = Depends(get_db)
):
    """Get all tools assigned to an agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get tool assignments
        query = db.query(AgentToolAssignment).options(
            joinedload(AgentToolAssignment.tool)
        ).filter(AgentToolAssignment.agent_id == agent_id)
        
        if enabled_only:
            query = query.filter(AgentToolAssignment.enabled == True)
        
        assignments = query.all()
        return assignments
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/tools/{tool_id}", response_model=AgentToolAssignmentResponse)
async def assign_tool_to_agent(
    agent_id: int,
    tool_id: int,
    assignment_data: AgentToolAssignmentCreate = Body(default=AgentToolAssignmentCreate(tool_id=0)),
    db: Session = Depends(get_db)
):
    """Assign tool to agent with permissions"""
    try:
        # Enhanced validation: Check if agent exists and is active
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.status != "active":
            raise HTTPException(status_code=400, detail=f"Agent '{agent.name}' is not active (status: {agent.status})")
        
        # Enhanced validation: Check if tool exists and is active
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        if tool.status != "active":
            raise HTTPException(status_code=400, detail=f"Tool '{tool.name}' is not active (status: {tool.status})")
        
        # Enhanced validation: Check for circular dependencies or conflicts
        if agent_id == tool_id:  # This shouldn't happen but let's be safe
            raise HTTPException(status_code=400, detail="Invalid assignment: agent and tool cannot have the same ID")
        
        # Check if assignment already exists
        existing = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == tool_id
        ).first()
        
        if existing:
            # Update existing assignment
            existing.enabled = assignment_data.enabled
            existing.permissions = assignment_data.permissions or {}
            existing.configuration = assignment_data.configuration or {}
            db.commit()
            db.refresh(existing)
            
            # Load tool relationship
            assignment_with_tool = db.query(AgentToolAssignment).options(
                joinedload(AgentToolAssignment.tool)
            ).filter(AgentToolAssignment.id == existing.id).first()
            
            logger.info(f"Updated tool assignment: Agent {agent_id} - Tool {tool_id}")
            return assignment_with_tool
        else:
            # Create new assignment
            assignment = AgentToolAssignment(
                agent_id=agent_id,
                tool_id=tool_id,
                enabled=assignment_data.enabled,
                permissions=assignment_data.permissions or {"read": True, "write": True, "execute": True},
                configuration=assignment_data.configuration or {}
            )
            
            db.add(assignment)
            db.commit()
            db.refresh(assignment)
            
            # Load tool relationship
            assignment_with_tool = db.query(AgentToolAssignment).options(
                joinedload(AgentToolAssignment.tool)
            ).filter(AgentToolAssignment.id == assignment.id).first()
            
            logger.info(f"Created tool assignment: Agent {agent_id} - Tool {tool_id}")
            return assignment_with_tool
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error assigning tool to agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}/tools/{tool_id}")
async def remove_tool_from_agent(
    agent_id: int,
    tool_id: int,
    db: Session = Depends(get_db)
):
    """Remove tool assignment from agent"""
    try:
        assignment = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == tool_id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=404, detail="Tool assignment not found")
        
        db.delete(assignment)
        db.commit()
        
        logger.info(f"Removed tool assignment: Agent {agent_id} - Tool {tool_id}")
        return {"status": "success", "message": "Tool removed from agent"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing tool from agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/agents/{agent_id}/tools/{tool_id}/permissions")
async def update_tool_permissions(
    agent_id: int,
    tool_id: int,
    permissions: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Update tool permissions for agent"""
    try:
        assignment = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == tool_id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=404, detail="Tool assignment not found")
        
        assignment.permissions = permissions
        db.commit()
        
        logger.info(f"Updated tool permissions: Agent {agent_id} - Tool {tool_id}")
        return {"status": "success", "message": "Permissions updated", "permissions": permissions}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating tool permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/usage/logs")
async def get_tool_usage_logs(
    tool_id: Optional[int] = Query(None),
    agent_id: Optional[int] = Query(None),
    success_only: Optional[bool] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get tool usage logs with optional filters"""
    try:
        ToolUsageLog = models.ToolUsageLog
        
        query = db.query(ToolUsageLog)
        
        if tool_id:
            query = query.filter(ToolUsageLog.tool_id == tool_id)
        if agent_id:
            query = query.filter(ToolUsageLog.agent_id == agent_id)
        if success_only is not None:
            query = query.filter(ToolUsageLog.success == success_only)
        
        logs = query.order_by(ToolUsageLog.created_at.desc()).offset(skip).limit(limit).all()
        
        return [{
            "id": log.id,
            "execution_id": log.execution_id,
            "agent_id": log.agent_id,
            "tool_id": log.tool_id,
            "method_called": log.method_called,
            "success": log.success,
            "execution_time_ms": log.execution_time_ms,
            "error_message": log.error_message,
            "created_at": log.created_at
        } for log in logs]
        
    except Exception as e:
        logger.error(f"Error getting tool usage logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

