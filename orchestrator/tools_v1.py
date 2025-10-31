"""
Tool Management API v1
======================
PRD-17 Phase 3: API endpoints for tool registry and management

Provides RESTful API for:
- Listing and querying tools
- Tool recommendations for tasks
- Agent-tool assignments
- Tool execution
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from database.database import get_db
from models import MCPTool, AgentToolAssignment, Agent
from services.tool_registry import ToolRegistry, get_tool_registry
from services.tool_capability_mapper import ToolCapabilityMapper, get_tool_capability_mapper
from services.unified_tool_executor import UnifiedToolExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tools", tags=["Tools v1"])


# ===================================================================
# REQUEST/RESPONSE MODELS
# ===================================================================

class ToolInfo(BaseModel):
    """Tool information response"""
    name: str
    description: str
    category: str
    security_level: str
    parameters: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    status: Optional[str] = None


class ToolRecommendationRequest(BaseModel):
    """Request for tool recommendations"""
    task_description: str
    task_type: Optional[str] = None
    include_optional: bool = False


class ToolRecommendationResponse(BaseModel):
    """Tool recommendation response"""
    task_type: str
    confidence: float
    required_tools: List[str]
    optional_tools: List[str]
    specific_tools: List[str]
    rationale: str


class AgentToolsUpdate(BaseModel):
    """Update agent tool assignments"""
    tool_categories: List[str]
    specific_tools: Optional[List[str]] = None


class ToolExecutionRequest(BaseModel):
    """Execute a tool"""
    tool_name: str
    parameters: Dict[str, Any]


# ===================================================================
# TOOL REGISTRY ENDPOINTS
# ===================================================================

@router.get("/registry", response_model=List[ToolInfo])
async def list_all_tools(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status (active/inactive)"),
    db: Session = Depends(get_db)
):
    """
    List all registered tools in the system.
    
    Returns both built-in tools (research, file_ops, shell) and MCP tools.
    """
    try:
        tool_registry = get_tool_registry(db)
        
        # Get all tools
        all_tools = []
        
        # Add built-in tools from registry
        for tool_name, tool_spec in tool_registry.tools.items():
            if category and tool_spec.category != category:
                continue
            
            all_tools.append(ToolInfo(
                name=tool_spec.name,
                description=tool_spec.description,
                category=tool_spec.category,
                security_level=tool_spec.security_level,
                parameters=tool_spec.parameters,
                provider="Built-in",
                status="active"
            ))
        
        # Add MCP tools from database
        mcp_query = db.query(MCPTool)
        if status:
            mcp_query = mcp_query.filter(MCPTool.status == status)
        if category:
            mcp_query = mcp_query.filter(MCPTool.category == category)
        
        mcp_tools = mcp_query.all()
        for mcp_tool in mcp_tools:
            all_tools.append(ToolInfo(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                category=mcp_tool.category or "mcp",
                security_level="cautious",
                parameters=mcp_tool.capabilities,
                provider=mcp_tool.provider,
                status=mcp_tool.status
            ))
        
        logger.info(f"✅ Returned {len(all_tools)} tools (filtered: category={category}, status={status})")
        return all_tools
        
    except Exception as e:
        logger.error(f"❌ Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def list_tool_categories(db: Session = Depends(get_db)):
    """List all tool categories with counts"""
    try:
        tool_registry = get_tool_registry(db)
        
        categories = {}
        for tool_spec in tool_registry.tools.values():
            cat = tool_spec.category
            if cat not in categories:
                categories[cat] = {"name": cat, "count": 0, "tools": []}
            categories[cat]["count"] += 1
            categories[cat]["tools"].append(tool_spec.name)
        
        # Add MCP tool categories
        mcp_tools = db.query(MCPTool).filter(MCPTool.status == 'active').all()
        for mcp_tool in mcp_tools:
            cat = mcp_tool.category or "mcp"
            if cat not in categories:
                categories[cat] = {"name": cat, "count": 0, "tools": []}
            categories[cat]["count"] += 1
            categories[cat]["tools"].append(mcp_tool.name)
        
        return {
            "categories": list(categories.values()),
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/{tool_name}")
async def get_tool_details(
    tool_name: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific tool"""
    try:
        tool_registry = get_tool_registry(db)
        
        # Check built-in tools
        tool_spec = tool_registry.get_tool(tool_name)
        if tool_spec:
            return {
                "name": tool_spec.name,
                "description": tool_spec.description,
                "category": tool_spec.category,
                "security_level": tool_spec.security_level,
                "parameters": tool_spec.parameters,
                "returns": tool_spec.returns,
                "status": tool_spec.status,
                "provider": "Built-in"
            }
        
        # Check MCP tools
        mcp_tool = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
        if mcp_tool:
            return {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "category": mcp_tool.category,
                "security_level": "cautious",
                "capabilities": mcp_tool.capabilities,
                "credentials_schema": mcp_tool.credentials_schema,
                "mcp_server_url": mcp_tool.mcp_server_url,
                "status": mcp_tool.status,
                "provider": mcp_tool.provider,
                "version": mcp_tool.version,
                "metadata": mcp_tool.tool_metadata
            }
        
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting tool details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# TOOL RECOMMENDATION ENDPOINTS
# ===================================================================

@router.post("/recommend", response_model=ToolRecommendationResponse)
async def recommend_tools_for_task(
    request: ToolRecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze a task and recommend which tools are needed.
    
    Uses ToolCapabilityMapper to intelligently map tasks to required tools.
    """
    try:
        mapper = get_tool_capability_mapper(db)
        
        # Infer task type if not provided
        if not request.task_type:
            task_type = mapper.infer_task_type(request.task_description)
        else:
            task_type = request.task_type
        
        # Get recommendation
        recommendation = mapper.get_recommendation(
            task_description=request.task_description,
            task_type=task_type
        )
        
        if not recommendation:
            raise HTTPException(
                status_code=400,
                detail=f"Could not determine tools for task type: {task_type}"
            )
        
        return ToolRecommendationResponse(
            task_type=task_type,
            confidence=recommendation.confidence,
            required_tools=recommendation.required,
            optional_tools=recommendation.optional if request.include_optional else [],
            specific_tools=recommendation.specific,
            rationale=recommendation.rationale
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error recommending tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_task_tools(
    task_description: str,
    db: Session = Depends(get_db)
):
    """
    Analyze a task description and provide detailed tool analysis.
    
    Returns:
        - Inferred task type
        - Required tool categories
        - Recommended specific tools
        - Reasoning
    """
    try:
        mapper = get_tool_capability_mapper(db)
        
        # Infer task type
        task_type = mapper.infer_task_type(task_description)
        
        # Get tool categories
        required_categories = mapper.get_tool_categories(task_type, include_optional=False)
        optional_categories = mapper.get_tool_categories(task_type, include_optional=True)
        optional_categories = [c for c in optional_categories if c not in required_categories]
        
        # Get recommendation
        recommendation = mapper.get_recommendation(task_description, task_type)
        
        return {
            "task_description": task_description,
            "task_type": task_type,
            "confidence": recommendation.confidence if recommendation else 0.5,
            "required_categories": required_categories,
            "optional_categories": optional_categories,
            "specific_tools": recommendation.specific if recommendation else [],
            "rationale": recommendation.rationale if recommendation else "Default task analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# AGENT TOOL MANAGEMENT ENDPOINTS
# ===================================================================

@router.get("/agents/{agent_id}/tools")
async def get_agent_tools(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Get all tools assigned to a specific agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get tool assignments
        assignments = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id
        ).all()
        
        assigned_tools = []
        for assignment in assignments:
            tool = db.query(MCPTool).filter(MCPTool.id == assignment.tool_id).first()
            if tool:
                assigned_tools.append({
                    "tool_id": tool.id,
                    "tool_name": tool.name,
                    "enabled": assignment.enabled,
                    "permissions": assignment.permissions,
                    "assigned_at": assignment.assigned_at.isoformat()
                })
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "assigned_tools": assigned_tools,
            "total_count": len(assigned_tools)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}/tools")
async def update_agent_tools(
    agent_id: int,
    update: AgentToolsUpdate,
    db: Session = Depends(get_db)
):
    """
    Update tool assignments for an agent.
    
    Assigns tools based on categories (e.g., research, file_ops, mcp)
    and optionally specific tools.
    """
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get tools by categories
        tool_registry = get_tool_registry(db)
        tools_to_assign = set()
        
        for category in update.tool_categories:
            category_tools = tool_registry.get_tools_by_category(category)
            tools_to_assign.update([t.name for t in category_tools])
        
        # Add specific tools if provided
        if update.specific_tools:
            tools_to_assign.update(update.specific_tools)
        
        # Clear existing assignments (optional - could also merge)
        db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id
        ).delete()
        
        # Create new assignments for MCP tools
        assigned_count = 0
        for tool_name in tools_to_assign:
            mcp_tool = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
            if mcp_tool:
                assignment = AgentToolAssignment(
                    agent_id=agent_id,
                    tool_id=mcp_tool.id,
                    enabled=True,
                    permissions={"all": True}  # Default full permissions
                )
                db.add(assignment)
                assigned_count += 1
        
        db.commit()
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "categories": update.tool_categories,
            "assigned_tools": list(tools_to_assign),
            "assigned_count": assigned_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error updating agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/execute")
async def execute_tool_for_agent(
    agent_id: int,
    execution: ToolExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute a tool on behalf of an agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Execute via UnifiedToolExecutor
        tool_executor = UnifiedToolExecutor(db)
        result = await tool_executor.execute_tool(
            tool_name=execution.tool_name,
            parameters=execution.parameters,
            agent_id=agent_id
        )
        
        return {
            "agent_id": agent_id,
            "tool": execution.tool_name,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# STATISTICS & ANALYTICS
# ===================================================================

@router.get("/stats")
async def get_tool_statistics(db: Session = Depends(get_db)):
    """Get tool usage statistics"""
    try:
        tool_registry = get_tool_registry(db)
        
        # Count tools by category
        category_counts = {}
        for tool_spec in tool_registry.tools.values():
            cat = tool_spec.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Count MCP tools
        mcp_active = db.query(MCPTool).filter(MCPTool.status == 'active').count()
        mcp_total = db.query(MCPTool).count()
        
        # Count agent assignments
        total_assignments = db.query(AgentToolAssignment).count()
        enabled_assignments = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.enabled == True
        ).count()
        
        return {
            "built_in_tools": len(tool_registry.tools),
            "mcp_tools_active": mcp_active,
            "mcp_tools_total": mcp_total,
            "total_tools": len(tool_registry.tools) + mcp_total,
            "category_distribution": category_counts,
            "agent_assignments_total": total_assignments,
            "agent_assignments_enabled": enabled_assignments
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting tool stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

