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
from core.composio.client import get_composio_client
from core.composio.entity_manager import EntityManager
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid

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
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """List all MCP tools with pagination and optional filters"""
    print(">>> API LIST MCP TOOLS HIT - DEBUG <<<")
    try:
        # Use Composio Client instead of legacy adapter
        composio_client = get_composio_client()
        try:
            # Get apps directly (they are the "tool groups")
            apps = composio_client.get_available_apps()
            items = _merge_composio_apps(apps, db)
        except Exception as e:
            logger.error(f"Failed to fetch Composio apps: {e}")
            # Fallback to local DB only if Composio fails
            items = _load_local_tools(db)

        # [ISOLATION FIX] Override status/enabled based on WorkspaceToolConfig
        from core.models.tool_assignments import WorkspaceToolConfig
        
        # Connected apps for this workspace (Composio connections)
        connected_app_names: set[str] = set()
        try:
            entity_manager = EntityManager(db)
            entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
            if entity:
                connections = entity_manager.get_entity_connections(entity["id"])
                connected_app_names = {
                    (c["app_name"] or "").lower()
                    for c in connections
                    if c.get("status") == "active"
                }
        except Exception as e:
            logger.warning(f"Failed to load connected apps for workspace: {e}")
        
        # specific to this workspace
        enabled_configs = db.query(WorkspaceToolConfig).filter(
            WorkspaceToolConfig.workspace_id == ctx.workspace_id,
            WorkspaceToolConfig.enabled == True
        ).all()
        
        logger.info(f"[MCP_TOOLS_DEBUG] Workspace: {ctx.workspace_id}, Enabled Configs Found: {len(enabled_configs)}")
        for cfg in enabled_configs:
            logger.info(f"[MCP_TOOLS_DEBUG] Config: {cfg.tool_id}, Enabled: {cfg.enabled}")

        enabled_tool_ids = {str(c.tool_id).lower() for c in enabled_configs} # adapter_tool_id / app_name
        
        for item in items:
            # Map item ID/Names to enablement
            # MCPTool.name usually matches the adapter tool_id/name
            # For Composio, the 'name' is the app name (e.g. "GITHUB")
            item_name = (item.get("name") or "").lower()
            is_connected = item_name in connected_app_names
            is_enabled = item_name in enabled_tool_ids or is_connected
            
            # If status filter is active, we might want to respect real enablement
            item["enabled"] = is_enabled
            item["status"] = "active" if is_enabled else "available" # 'available' means in marketplace but not enabled
            if is_connected:
                metadata = item.get("metadata") or {}
                if not metadata.get("action_count"):
                    try:
                        metadata["action_count"] = composio_client.get_app_action_count(item.get("name") or "")
                    except Exception as e:
                        logger.warning(f"Failed to fetch action count for {item.get('name')}: {e}")
                        metadata["action_count"] = 0
                item["metadata"] = metadata

        # Apply filters
        # If filtering by status='active', we now only return actually enabled tools
        filtered = _apply_filters(items, status, category, provider, search)
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit else 1
        paged = filtered[skip : skip + limit]

        # Populate action counts for visible apps if missing
        for item in paged:
            if item.get("provider") != "Composio":
                continue
            metadata = item.get("metadata") or {}
            if not metadata.get("action_count"):
                try:
                    metadata["action_count"] = composio_client.get_app_action_count(item.get("name") or "")
                except Exception as e:
                    logger.warning(f"Failed to fetch action count for {item.get('name')}: {e}")
                    metadata["action_count"] = 0
                item["metadata"] = metadata

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

@router.get("/{tool_id}/actions")
async def get_tool_actions(
    tool_id: int,
    db: Session = Depends(get_db)
):
    """Get all available actions (sub-tools) for a specific tool/app"""
    try:
        # Get tool name from DB or ID
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
             # Fallback: check if we can resolve ID to a composio app name directly? 
             # For now require tool to exist in DB (which it should if it was listed)
             raise HTTPException(status_code=404, detail="Tool not found")
        
        if tool.provider != "Composio":
            return [] # Local tools don't have dynamic actions via this API yet

        composio_client = get_composio_client()
        # tool.name is the app name (e.g. "gmail", "github")
        actions = composio_client.get_app_actions(tool.name)
        return actions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching actions for tool {tool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}")
async def get_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    """Get single MCP tool by ID"""
    try:
        # Try to find by integer ID first (local DB)
        if tool_id:
             # Try to find by integer ID first (local DB)
             tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
             if tool:
                 return _local_tool_to_response(tool)

        # If not found locally by ID, check Composio
        composio_client = get_composio_client()
        try:
            apps = composio_client.get_available_apps()
            items = _merge_composio_apps(apps, db)
            for item in items:
                if item["id"] == tool_id:
                    return item
        except Exception as e:
            logger.error(f"Failed to fetch from Composio: {e}")
            
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
        composio_client = get_composio_client()
        try:
            apps = composio_client.get_available_apps()
            items = _merge_composio_apps(apps, db)
            counts: Dict[str, int] = {}
            for item in items:
                tags = item.get("tags") or []
                if tags:
                    for tag in tags:
                        counts[tag] = counts.get(tag, 0) + 1
                else:
                    cat = item.get("category") or "uncategorized"
                    counts[cat] = counts.get(cat, 0) + 1
            return [{"name": name, "count": count} for name, count in counts.items()]
        except Exception:
             # Fallback
             pass

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
async def get_tools_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get tool statistics summary"""
    try:
        composio_client = get_composio_client()
        try:
            apps = composio_client.get_available_apps()
            items = _merge_composio_apps(apps, db)
            total = len(items)
            # Connected apps for this workspace
            connected_app_names: set[str] = set()
            try:
                entity_manager = EntityManager(db)
                entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
                if entity:
                    connections = entity_manager.get_entity_connections(entity["id"])
                    connected_app_names = {
                        (c["app_name"] or "").lower()
                        for c in connections
                        if c.get("status") == "active"
                    }
            except Exception as e:
                logger.warning(f"Failed to load connected apps for workspace stats: {e}")

            from core.models.tool_assignments import WorkspaceToolConfig
            enabled_configs = db.query(WorkspaceToolConfig).filter(
                WorkspaceToolConfig.workspace_id == ctx.workspace_id,
                WorkspaceToolConfig.enabled == True
            ).all()
            enabled_tool_ids = {str(c.tool_id).lower() for c in enabled_configs}

            active = 0
            categories = set()
            total_tools = 0
            for item in items:
                item_name = (item.get("name") or "").lower()
                is_connected = item_name in connected_app_names
                is_enabled = item_name in enabled_tool_ids or is_connected
                if is_enabled:
                    active += 1

                tags = item.get("tags") or []
                if tags:
                    for tag in tags:
                        categories.add(tag)
                else:
                    categories.add(item.get("category") or "uncategorized")

                if is_connected:
                    action_count = (item.get("metadata") or {}).get("action_count") or 0
                    if not action_count:
                        try:
                            action_count = composio_client.get_app_action_count(item.get("name") or "")
                        except Exception as e:
                            logger.warning(f"Failed to fetch action count for {item.get('name')}: {e}")
                            action_count = 0
                    total_tools += action_count
        except Exception:
            total = db.query(func.count(MCPTool.id)).scalar()
            active = (
                db.query(func.count(MCPTool.id))
                .filter(MCPTool.status == "active")
                .scalar()
            )
            categories = set(
                r[0] for r in db.query(MCPTool.category).filter(MCPTool.category.isnot(None)).all()
            )
            total_tools = 0
        assigned = db.query(func.count(func.distinct(AgentToolAssignment.tool_id))).filter(
            AgentToolAssignment.enabled == True
        ).scalar()
        
        return {
            "total_tools": total,
            "active_tools": active,
            "assigned_tools": assigned,
            "unassigned_tools": total - assigned,
            "connected_apps": active,
            "categories": len(categories),
            "apps_available": total,
            "tools_available": total_tools,
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
        category_lower = category.lower()
        filtered = [
            item for item in filtered
            if (item.get("category") or "").lower() == category_lower
            or any((tag or "").lower() == category_lower for tag in (item.get("tags") or []))
        ]
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
    # Ensure even fallback only returns Composio tools
    return [_local_tool_to_response(tool) for tool in tools if tool.provider == "Composio"]


def _merge_composio_apps(apps: List[Dict[str, Any]], db: Session) -> List[Dict[str, Any]]:
    """Merge Composio apps into tool list, creating DB entries if needed."""
    local_tools = {tool.name: tool for tool in db.query(MCPTool).all()}
    created = False
    
    for app in apps:
        name = app.get("name")
        if not name:
            continue
            
        # Create local mirror if not exists
        if name not in local_tools:
            # Default metadata for new tools
            meta = app.get("meta", {})
            
            new_tool = MCPTool(
                name=name,
                description=app.get("description") or "",
                mcp_server_url="composio://", # placeholder for Composio
                capabilities={"actions": []}, # To be populated by describe
                credentials_schema={},
                status="inactive", # Default to inactive until enabled
                provider="Composio",
                tool_id=name,
                version="1.0.0",
                icon=app.get("logo_url"),
                logo=app.get("logo_url"),
                category=(app.get("categories") or ["Integration"])[0],
                tags=app.get("categories") or [],
                tool_metadata={
                    "app_name": name,
                    "auth_schemes": app.get("auth_schemes", []),
                    "triggers": app.get("triggers", []),
                    "trigger_count": app.get("trigger_count", 0),
                    "action_count": app.get("action_count", 0),
                },
                created_by="composio-sync",
            )
            db.add(new_tool)
            local_tools[name] = new_tool
            created = True
            
    if created:
        db.commit()
        for tool in local_tools.values():
            db.refresh(tool)

    items: List[Dict[str, Any]] = []
    
    # Return merged list
    # Preference: Local DB state (which tracks ID and status), enriched with live Composio metadata
    for app in apps:
        name = app.get("name")
        local = local_tools.get(name)
        
        status = local.status if local else "inactive"
        
        items.append(
            {
                "id": local.id if local else 0, # Should always have local ID if we synced
                "name": name,
                "description": app.get("description") or (local.description if local else ""),
                "mcp_server_url": local.mcp_server_url if local else "composio://",
                "capabilities": local.capabilities if local else {},
                "credentials_schema": {}, # handled by Composio auth
                "status": status,
                "provider": "Composio",
                "version": local.version if local else "1.0.0",
                "icon": app.get("logo_url") or (local.icon if local else None),
                "logo": app.get("logo_url") or (local.logo if local else None),
                "category": ((app.get("categories") or ["Integration"])[0]) if app.get("categories") else (local.category if local else "Integration"),
                "tags": app.get("categories") or [],
                "metadata": {
                    **((local.tool_metadata or {}) if local else {}),
                    "auth_schemes": app.get("auth_schemes", []),
                    "triggers": app.get("triggers", []),
                    "trigger_count": app.get("trigger_count", 0),
                    "action_count": app.get("action_count", 0),
                },
                "credential_usage": "oauth",
                "created_at": local.created_at if local else None,
                "updated_at": local.updated_at if local else None,
                "created_by": local.created_by if local else None,
            }
        )
    
    # Also include non-Composio tools (e.g. database tools)
    for tool_name, tool in local_tools.items():
        # ONLY return Composio tools - hide legacy "Internal" or "Marketplace" tools
        if tool.provider == "Composio" and tool.name not in [app.get("name") for app in apps if app.get("name")]:
             # If it's a Composio tool in DB but not in current fetch (e.g. valid cache), include it
            items.append(_local_tool_to_response(tool))
            
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

