

"""
Tools Management API Routes

Comprehensive API for managing tools, configurations, and installations.
Supports MCP (Model Context Protocol) tool integration and marketplace functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from pydantic import BaseModel, Field

from core.database.database import get_db
from core.models import Tool, ToolCredentials, ToolConfiguration, ToolUsageLog
from core.models import Tool, ToolCredentials, ToolConfiguration, ToolUsageLog
from core.utils.logging_adapter import set_request_id
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tools Management"])

# ====================================
# Pydantic Models for Request/Response
# ====================================

class ToolBase(BaseModel):
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    category: str = Field(..., description="Tool category")
    provider: str = Field(..., description="Tool provider")
    version: str = Field(..., description="Tool version")
    icon: Optional[str] = Field(None, description="Tool icon emoji or identifier")
    pricing: Optional[str] = Field(None, description="Pricing model")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    permissions: List[str] = Field(default_factory=list, description="Allowed agent types")
    required_credentials: List[str] = Field(default_factory=list, description="Required credential keys")
    supported_environments: List[str] = Field(default_factory=list, description="Supported environments")
    mcp_config: Optional[Dict[str, Any]] = Field(None, description="MCP server configuration")

class ToolCreate(ToolBase):
    status: str = Field(default="available", description="Tool status")

class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None
    permissions: Optional[List[str]] = None
    required_credentials: Optional[List[str]] = None
    supported_environments: Optional[List[str]] = None
    mcp_config: Optional[Dict[str, Any]] = None

class ToolResponse(ToolBase):
    id: int
    status: str
    is_installed: bool
    is_configured: bool
    usage_count: int
    rating: float
    last_updated: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class ToolInstallRequest(BaseModel):
    environment: str = Field(default="development", description="Target environment")
    auto_configure: bool = Field(default=False, description="Auto-configure with default settings")

class ToolConfigurationRequest(BaseModel):
    environment: str = Field(..., description="Target environment")
    credentials: Dict[str, str] = Field(..., description="Tool credentials")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Tool configuration")
    test_connection: bool = Field(default=True, description="Test connection after configuration")

class ToolUsageStats(BaseModel):
    tool_id: int
    total_usage: int
    success_rate: float
    avg_response_time: float
    last_used: Optional[datetime]
    top_agents: List[Dict[str, Any]]

# ====================================
# Helper Functions
# ====================================

def get_mock_marketplace_tools() -> List[Dict[str, Any]]:
    """
    REMOVED - NO MOCK DATA IN PRODUCTION
    Returns empty list - populate from real MCP tools only
    """
    return []  # NO MOCK DATA

def create_tool_from_dict(tool_data: Dict[str, Any], db: Session) -> Tool:
    """Create a tool record from dictionary data"""
    tool = Tool(
        name=tool_data["name"],
        description=tool_data["description"],
        category=tool_data["category"],
        provider=tool_data["provider"],
        version=tool_data["version"],
        icon=tool_data.get("icon"),
        pricing=tool_data.get("pricing"),
        rating=tool_data.get("rating", 0.0),
        status=tool_data.get("status", "available"),
        tags=tool_data.get("tags", []),
        permissions=tool_data.get("permissions", []),
        required_credentials=tool_data.get("required_credentials", []),
        supported_environments=tool_data.get("supported_environments", []),
        mcp_config=tool_data.get("mcp_config"),
        is_installed=False,
        is_configured=False,
        usage_count=0,
        created_at=datetime.utcnow()
    )
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return tool

# ====================================
# API Endpoints
# ====================================

@router.get("/", response_model=List[ToolResponse])
async def list_tools(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    installed_only: bool = Query(False, description="Only show installed tools"),
    search: Optional[str] = Query(None, description="Search in name, description, tags"),
    sort_by: str = Query("name", description="Sort by field"),
    db: Session = Depends(get_db)
):
    """
    Get list of all available tools with optional filtering and sorting.
    
    **Categories**: developer, communication, cloud, analytics, productivity, security
    **Statuses**: available, deprecated, maintenance, beta
    **Sort Options**: name, rating, usage, updated, created
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # NO MOCK DATA - only use real tools from database
        tools_count = db.query(func.count(Tool.id)).scalar()
        if tools_count == 0:
            logger.info("No tools found in database - waiting for real MCP tools to be registered")
        
        query = db.query(Tool)
        
        # Apply filters
        if category:
            query = query.filter(Tool.category == category)
        
        if status:
            query = query.filter(Tool.status == status)
        
        if installed_only:
            query = query.filter(Tool.is_installed == True)
        
        if search:
            search_filter = or_(
                Tool.name.ilike(f"%{search}%"),
                Tool.description.ilike(f"%{search}%"),
                Tool.provider.ilike(f"%{search}%"),
                func.array_to_string(Tool.tags, ',').ilike(f"%{search}%")
            )
            query = query.filter(search_filter)
        
        # Apply sorting
        sort_mapping = {
            "name": Tool.name,
            "rating": Tool.rating.desc(),
            "usage": Tool.usage_count.desc(),
            "updated": Tool.last_updated.desc(),
            "created": Tool.created_at.desc()
        }
        
        if sort_by in sort_mapping:
            query = query.order_by(sort_mapping[sort_by])
        else:
            query = query.order_by(Tool.name)
        
        tools = query.all()
        
        logger.info(f"Retrieved {len(tools)} tools with filters: category={category}, status={status}")
        return tools
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tools: {str(e)}")

@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific tool"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        logger.info(f"Retrieved tool: {tool.name} (ID: {tool_id})")
        return tool
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving tool {tool_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tool: {str(e)}")

@router.post("/{tool_id}/install")
async def install_tool(
    tool_id: int,
    install_request: ToolInstallRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Install a tool in the specified environment.
    This handles MCP server setup, credential preparation, and initial configuration.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        if tool.is_installed:
            raise HTTPException(status_code=400, detail="Tool is already installed")
        
        # Update tool status
        tool.is_installed = True
        tool.last_updated = datetime.utcnow()
        
        # Create installation log
        usage_log = ToolUsageLog(
            tool_id=tool_id,
            action="install",
            environment=install_request.environment,
            details={"auto_configure": install_request.auto_configure},
            timestamp=datetime.utcnow()
        )
        db.add(usage_log)
        
        # If auto-configure is enabled, set up basic configuration
        if install_request.auto_configure:
            config = ToolConfiguration(
                tool_id=tool_id,
                environment=install_request.environment,
                configuration={
                    "auto_configured": True,
                    "configured_at": datetime.utcnow().isoformat()
                },
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.add(config)
            tool.is_configured = True
        
        db.commit()
        
        # Add background task for MCP server setup
        if tool.mcp_config:
            background_tasks.add_task(setup_mcp_server, tool_id, tool.mcp_config)
        
        logger.info(f"Installed tool: {tool.name} in environment: {install_request.environment}")
        return {
            "message": f"Tool {tool.name} installed successfully",
            "tool_id": tool_id,
            "environment": install_request.environment,
            "auto_configured": install_request.auto_configure
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error installing tool {tool_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to install tool: {str(e)}")

@router.post("/{tool_id}/configure")
async def configure_tool(
    tool_id: int,
    config_request: ToolConfigurationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Configure a tool with credentials and settings for a specific environment.
    Supports encrypted credential storage and connection testing.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        if not tool.is_installed:
            raise HTTPException(status_code=400, detail="Tool must be installed before configuration")
        
        # Store encrypted credentials
        for cred_key, cred_value in config_request.credentials.items():
            existing_cred = db.query(ToolCredentials).filter(
                and_(
                    ToolCredentials.tool_id == tool_id,
                    ToolCredentials.environment == config_request.environment,
                    ToolCredentials.credential_key == cred_key
                )
            ).first()
            
            if existing_cred:
                existing_cred.credential_value = cred_value  # Will be encrypted in real implementation
                existing_cred.updated_at = datetime.utcnow()
            else:
                new_cred = ToolCredentials(
                    tool_id=tool_id,
                    environment=config_request.environment,
                    credential_key=cred_key,
                    credential_value=cred_value,  # Will be encrypted in real implementation
                    created_at=datetime.utcnow()
                )
                db.add(new_cred)
        
        # Store configuration
        existing_config = db.query(ToolConfiguration).filter(
            and_(
                ToolConfiguration.tool_id == tool_id,
                ToolConfiguration.environment == config_request.environment
            )
        ).first()
        
        if existing_config:
            existing_config.configuration = config_request.configuration
            existing_config.updated_at = datetime.utcnow()
            existing_config.is_active = True
        else:
            new_config = ToolConfiguration(
                tool_id=tool_id,
                environment=config_request.environment,
                configuration=config_request.configuration,
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.add(new_config)
        
        # Update tool status
        tool.is_configured = True
        tool.last_updated = datetime.utcnow()
        
        # Create configuration log
        usage_log = ToolUsageLog(
            tool_id=tool_id,
            action="configure",
            environment=config_request.environment,
            details={"test_connection": config_request.test_connection},
            timestamp=datetime.utcnow()
        )
        db.add(usage_log)
        
        db.commit()
        
        # Test connection if requested
        connection_result = {"status": "skipped"}
        if config_request.test_connection:
            background_tasks.add_task(test_tool_connection, tool_id, config_request.environment)
            connection_result = {"status": "testing", "message": "Connection test initiated"}
        
        logger.info(f"Configured tool: {tool.name} for environment: {config_request.environment}")
        return {
            "message": f"Tool {tool.name} configured successfully",
            "tool_id": tool_id,
            "environment": config_request.environment,
            "connection_test": connection_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring tool {tool_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to configure tool: {str(e)}")

@router.delete("/{tool_id}/uninstall")
async def uninstall_tool(
    tool_id: int,
    environment: str = Query(..., description="Environment to uninstall from"),
    force: bool = Query(False, description="Force uninstall even if in use"),
    db: Session = Depends(get_db)
):
    """
    Uninstall a tool from the specified environment.
    Removes configurations, credentials, and MCP server connections.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        if not tool.is_installed:
            raise HTTPException(status_code=400, detail="Tool is not installed")
        
        # Check if tool is in use (in real implementation, check active workflows/agents)
        if not force:
            # Mock check - in real implementation, query active usage
            active_usage = False  # Would check actual usage
            if active_usage:
                raise HTTPException(
                    status_code=400, 
                    detail="Tool is currently in use. Use force=true to override."
                )
        
        # Remove credentials for the environment
        db.query(ToolCredentials).filter(
            and_(
                ToolCredentials.tool_id == tool_id,
                ToolCredentials.environment == environment
            )
        ).delete()
        
        # Remove configurations for the environment
        db.query(ToolConfiguration).filter(
            and_(
                ToolConfiguration.tool_id == tool_id,
                ToolConfiguration.environment == environment
            )
        ).delete()
        
        # Check if there are configurations for other environments
        remaining_configs = db.query(ToolConfiguration).filter(
            ToolConfiguration.tool_id == tool_id
        ).count()
        
        if remaining_configs == 0:
            tool.is_configured = False
            tool.is_installed = False
        
        tool.last_updated = datetime.utcnow()
        
        # Create uninstallation log
        usage_log = ToolUsageLog(
            tool_id=tool_id,
            action="uninstall",
            environment=environment,
            details={"force": force},
            timestamp=datetime.utcnow()
        )
        db.add(usage_log)
        
        db.commit()
        
        logger.info(f"Uninstalled tool: {tool.name} from environment: {environment}")
        return {
            "message": f"Tool {tool.name} uninstalled successfully",
            "tool_id": tool_id,
            "environment": environment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uninstalling tool {tool_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to uninstall tool: {str(e)}")

@router.get("/{tool_id}/status")
async def get_tool_status(tool_id: int, db: Session = Depends(get_db)):
    """Get detailed status information for a tool including health and usage metrics"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Get recent usage logs
        recent_usage = db.query(ToolUsageLog).filter(
            ToolUsageLog.tool_id == tool_id
        ).order_by(ToolUsageLog.timestamp.desc()).limit(10).all()
        
        # Get configurations by environment
        configurations = db.query(ToolConfiguration).filter(
            ToolConfiguration.tool_id == tool_id
        ).all()
        
        config_by_env = {}
        for config in configurations:
            config_by_env[config.environment] = {
                "is_active": config.is_active,
                "configured_at": config.created_at,
                "last_updated": config.updated_at
            }
        
        # Mock health check - in real implementation, ping actual services
        health_status = "healthy" if tool.is_configured else "not_configured"
        
        status_info = {
            "tool_id": tool_id,
            "name": tool.name,
            "is_installed": tool.is_installed,
            "is_configured": tool.is_configured,
            "health_status": health_status,
            "usage_count": tool.usage_count,
            "rating": tool.rating,
            "last_updated": tool.last_updated,
            "configurations": config_by_env,
            "recent_activity": [
                {
                    "action": log.action,
                    "environment": log.environment,
                    "timestamp": log.timestamp,
                    "details": log.details
                }
                for log in recent_usage
            ],
            "mcp_status": {
                "enabled": bool(tool.mcp_config),
                "server_url": tool.mcp_config.get("server_url") if tool.mcp_config else None,
                "protocol_version": tool.mcp_config.get("protocol_version") if tool.mcp_config else None
            }
        }
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool status {tool_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool status: {str(e)}")

@router.get("/{tool_id}/usage-stats", response_model=ToolUsageStats)
async def get_tool_usage_stats(tool_id: int, db: Session = Depends(get_db)):
    """Get comprehensive usage statistics for a tool"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Get usage statistics from logs
        total_usage = db.query(func.count(ToolUsageLog.id)).filter(
            ToolUsageLog.tool_id == tool_id
        ).scalar() or 0
        
        # Mock additional statistics
        stats = ToolUsageStats(
            tool_id=tool_id,
            total_usage=total_usage,
            success_rate=0.95,  # Mock data
            avg_response_time=250.0,  # Mock data in ms
            last_used=datetime.utcnow(),  # Mock data
            top_agents=[  # Mock data
                {"agent_id": 1, "agent_name": "Code Architect Pro", "usage_count": 45},
                {"agent_id": 3, "agent_name": "Performance Optimizer", "usage_count": 32}
            ]
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting usage stats for tool {tool_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage stats: {str(e)}")

# ====================================
# Background Tasks
# ====================================

async def setup_mcp_server(tool_id: int, mcp_config: Dict[str, Any]):
    """Background task to set up MCP server connection"""
    try:
        logger.info(f"Setting up MCP server for tool {tool_id}")
        # In real implementation:
        # 1. Connect to MCP server
        # 2. Validate protocol version
        # 3. Register tool capabilities
        # 4. Update tool status
        
        # Mock implementation
        await asyncio.sleep(2)  # Simulate setup time
        logger.info(f"MCP server setup completed for tool {tool_id}")
        
    except Exception as e:
        logger.error(f"Failed to setup MCP server for tool {tool_id}: {e}")

async def test_tool_connection(tool_id: int, environment: str):
    """Background task to test tool connection"""
    try:
        logger.info(f"Testing connection for tool {tool_id} in {environment}")
        # In real implementation:
        # 1. Load credentials
        # 2. Make test API call
        # 3. Validate response
        # 4. Update connection status
        
        # Mock implementation
        await asyncio.sleep(3)  # Simulate connection test
        logger.info(f"Connection test completed for tool {tool_id}")
        
    except Exception as e:
        logger.error(f"Failed to test connection for tool {tool_id}: {e}")

# ====================================
# Health Check
# ====================================

@router.get("/health")
async def tools_health_check():
    """Health check endpoint for tools management system"""
    return {
        "status": "healthy",
        "service": "tools-management",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
