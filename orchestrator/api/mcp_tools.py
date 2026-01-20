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

from core.database.database import get_db
from core.models import Agent, MCPTool, AgentToolAssignment, MCPToolCreate, MCPToolUpdate, MCPToolResponse, AgentToolAssignmentCreate, AgentToolAssignmentResponse
from modules.tools.services.adapter_client import AdapterClient

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
    skip: int = Query(0, ge=0, description="Number of items to skip (for pagination)"),
    limit: int = Query(20, ge=1, le=10000, description="Number of items per page (max 10000)"),
    db: Session = Depends(get_db)
):
    """List all MCP tools with pagination and optional filters"""
    try:
        adapter_client = AdapterClient()
        if adapter_client.is_configured():
            adapter_tools = await adapter_client.list_tools()
            items = _merge_adapter_tools(adapter_tools, db)
        else:
            items = _load_local_tools(db)

        # Apply filters
        filtered = _apply_filters(items, status, category, provider, search)
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit else 1
        paged = filtered[skip : skip + limit]

        return {
            "data": paged,
            "pagination": {
                "total": total,
                "skip": skip,
                "limit": limit,
                "pages": pages,
                "current_page": (skip // limit) + 1 if limit > 0 else 1,
            },
        }
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}")
async def get_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    """Get single MCP tool by ID"""
    try:
        # Try to find by integer ID first (local DB)
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if tool:
             return _local_tool_to_response(tool)

        # If not found locally by ID, check adapter
        adapter_client = AdapterClient()
        if adapter_client.is_configured():
            adapter_tools = await adapter_client.list_tools()
            items = _merge_adapter_tools(adapter_tools, db)
            for item in items:
                if item["id"] == tool_id:
                    return item
            
        raise HTTPException(status_code=404, detail="Tool not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading MCP tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        update_data = tool_data.model_dump(exclude_unset=True, by_alias=False)
        logger.info(f"Updating tool {tool_id} with data: {update_data}")

        allowed_fields = {"status", "metadata", "tool_metadata"}
        for field, value in update_data.items():
            if field not in allowed_fields:
                continue
            if field in {"metadata", "tool_metadata"}:
                setattr(tool, "tool_metadata", value)
            else:
                setattr(tool, field, value)
        
        db.commit()
        db.refresh(tool)
        
        logger.info(f"✅ Updated MCP tool: {tool.name} (ID: {tool.id}, Status: {tool.status})")
        return tool
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating MCP tool {tool_id}: {str(e)}", exc_info=True)
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
        adapter_client = AdapterClient()
        if adapter_client.is_configured():
            adapter_tools = await adapter_client.list_tools()
            items = _merge_adapter_tools(adapter_tools, db)
            counts: Dict[str, int] = {}
            for item in items:
                cat = item.get("category") or "uncategorized"
                counts[cat] = counts.get(cat, 0) + 1
            return [{"name": name, "count": count} for name, count in counts.items()]

        categories = db.query(
            MCPTool.category,
            func.count(MCPTool.id).label("count"),
        ).filter(MCPTool.category.isnot(None)).group_by(MCPTool.category).all()
        return [{"name": cat, "count": count} for cat, count in categories]
        
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
        adapter_client = AdapterClient()
        if adapter_client.is_configured():
            adapter_tools = await adapter_client.list_tools()
            items = _merge_adapter_tools(adapter_tools, db)
            total = len(items)
            active = sum(1 for item in items if item.get("status") == "active")
        else:
            total = db.query(func.count(MCPTool.id)).scalar()
            active = (
                db.query(func.count(MCPTool.id))
                .filter(MCPTool.status == "active")
                .scalar()
            )
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


def _apply_filters(
    items: List[Dict[str, Any]],
    status: Optional[str],
    category: Optional[str],
    provider: Optional[str],
    search: Optional[str],
) -> List[Dict[str, Any]]:
    filtered = items
    if status:
        filtered = [item for item in filtered if item.get("status") == status]
    if category:
        filtered = [item for item in filtered if item.get("category") == category]
    if provider:
        filtered = [item for item in filtered if item.get("provider") == provider]
    if search:
        query = search.lower()
        filtered = [
            item
            for item in filtered
            if query in (item.get("name") or "").lower()
            or query in (item.get("description") or "").lower()
            or query in (item.get("provider") or "").lower()
        ]
    return sorted(filtered, key=lambda item: item.get("name") or "")


def _local_tool_to_response(tool: MCPTool) -> Dict[str, Any]:
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
        "logo": tool.logo,
        "category": tool.category,
        "tags": tool.tags or [],
        "metadata": tool.tool_metadata or {},
        "credential_usage": "tool",
        "created_at": tool.created_at,
        "updated_at": tool.updated_at,
        "created_by": tool.created_by,
    }


def _load_local_tools(db: Session) -> List[Dict[str, Any]]:
    tools = db.query(MCPTool).order_by(MCPTool.name).all()
    return [_local_tool_to_response(tool) for tool in tools]


def _merge_adapter_tools(adapter_tools: List[Dict[str, Any]], db: Session) -> List[Dict[str, Any]]:
    local_tools = {tool.name: tool for tool in db.query(MCPTool).all()}
    created = False
    for adapter_tool in adapter_tools:
        name = adapter_tool.get("name")
        if not name:
            continue
        if name in local_tools:
            continue
        status = "active" if adapter_tool.get("enabled") else "inactive"
        new_tool = MCPTool(
            name=name,
            description=adapter_tool.get("description") or "",
            mcp_server_url=adapter_tool.get("mcp_server_url"),
            capabilities=(adapter_tool.get("metadata") or {}).get("capabilities") or {},
            credentials_schema=(adapter_tool.get("metadata") or {}).get("credentials_schema") or {},
            status=status,
            provider=adapter_tool.get("provider"),
            version=(adapter_tool.get("metadata") or {}).get("version"),
            icon=(adapter_tool.get("metadata") or {}).get("icon"),
            logo=(adapter_tool.get("metadata") or {}).get("logo"),
            category=adapter_tool.get("category"),
            tags=adapter_tool.get("tags") or [],
            tool_metadata={"adapter_tool_id": adapter_tool.get("id")},
            created_by="adapter-sync",
        )
        db.add(new_tool)
        local_tools[name] = new_tool
        created = True
    if created:
        db.commit()
        for tool in local_tools.values():
            db.refresh(tool)

    items: List[Dict[str, Any]] = []
    for adapter_tool in adapter_tools:
        name = adapter_tool.get("name")
        if not name:
            continue
        local = local_tools.get(name)
        adapter_metadata = adapter_tool.get("metadata") or {}
        status = local.status if local else ("active" if adapter_tool.get("enabled") else "inactive")
        items.append(
            {
                "id": local.id if local else adapter_tool.get("id"),
                "name": name,
                "description": adapter_tool.get("description") or "",
                "mcp_server_url": adapter_tool.get("mcp_server_url"),
                "capabilities": adapter_metadata.get("capabilities") or {},
                "credentials_schema": adapter_metadata.get("credentials_schema") or {},
                "status": status,
                "provider": adapter_tool.get("provider"),
                "version": adapter_metadata.get("version") or (local.version if local else None),
                "icon": adapter_metadata.get("icon") or (local.icon if local else None),
                "logo": adapter_metadata.get("logo") or (local.logo if local else None),
                "category": adapter_tool.get("category"),
                "tags": adapter_tool.get("tags") or [],
                "metadata": adapter_metadata,
                "credential_usage": "tool",
                "created_at": local.created_at if local else None,
                "updated_at": local.updated_at if local else None,
                "created_by": local.created_by if local else None,
            }
        )
    return items

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
        
        # Enhanced validation: Check if tool exists and is active (Lookup by PK)
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        if tool.status != "active":
            raise HTTPException(status_code=400, detail=f"Tool '{tool.name}' is not active (status: {tool.status})")
        
        # Use simple string ID from the tool model for the assignment
        clean_tool_id = tool.tool_id
        
        # Enhanced validation: Check for circular dependencies or conflicts
        if agent_id == tool_id:  # This shouldn't happen but let's be safe
            raise HTTPException(status_code=400, detail="Invalid assignment: agent and tool cannot have the same ID")
        
        # Check if assignment already exists
        existing = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == clean_tool_id
        ).first()
        
        if existing:
            # Update existing assignment
            existing.enabled = assignment_data.enabled
            existing.permissions = assignment_data.permissions or {}
            existing.configuration = assignment_data.configuration or {}
            db.commit()
            db.refresh(existing)
            
            # Load tool relationship - Re-query to ensure relations are loaded
            assignment_with_tool = db.query(AgentToolAssignment).options(
                joinedload(AgentToolAssignment.tool)
            ).filter(AgentToolAssignment.id == existing.id).first()
            
            logger.info(f"Updated tool assignment: Agent {agent_id} - Tool {clean_tool_id} (PK: {tool_id})")
            return assignment_with_tool
        else:
            # Create new assignment
            assignment = AgentToolAssignment(
                agent_id=agent_id,
                tool_id=clean_tool_id,
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
            
            logger.info(f"Created tool assignment: Agent {agent_id} - Tool {clean_tool_id} (PK: {tool_id})")
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
        # Resolve tool PK to string ID first
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
             raise HTTPException(status_code=404, detail="Tool not found")
        
        clean_tool_id = tool.tool_id

        assignment = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == clean_tool_id
        ).first()
        
        if not assignment:
            # Fallback: check if assignment exists with PK (legacy/corrupt data that wasn't migrated)
            assignment_pk = db.query(AgentToolAssignment).filter(
                AgentToolAssignment.agent_id == agent_id,
                AgentToolAssignment.tool_id == str(tool_id)
            ).first()
            if assignment_pk:
                assignment = assignment_pk
            else:
                raise HTTPException(status_code=404, detail="Tool assignment not found")
        
        db.delete(assignment)
        db.commit()
        
        logger.info(f"Removed tool assignment: Agent {agent_id} - Tool {clean_tool_id} (PK: {tool_id})")
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

