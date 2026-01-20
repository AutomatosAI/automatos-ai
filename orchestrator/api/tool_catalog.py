"""
Tool Catalog & Assignment API (PRD-35)
======================================

API endpoints for:
- Tool enablement (tenant level)
- Agent tool assignments
- Credential resolution (for Adapter callback)

Part of PRD-35: Tool Catalog & Registry Architecture
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models import Agent
from core.models.credentials import Credential, CredentialType
from core.models.tool_assignments import (
    AgentToolAssignmentResponse,
    AgentToolAssignBatchRequest,
    AgentToolAssignRequest,
    AgentToolResponse,
    AvailableToolResponse,
    CredentialResolveRequest,
    CredentialResolveResponse,
    TenantToolConfig,
    TenantToolConfigResponse,
    ToolEnableRequest,
)
from core.credentials.service import CredentialStore
from modules.tools.services.tool_access_service import get_tool_access_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["Tool Catalog"])


# =============================================================================
# Tool Enablement (Tenant Level)
# =============================================================================

@router.get("/available", response_model=List[AvailableToolResponse])
async def list_available_tools(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    tenant_id: UUID = Query(..., description="Tenant UUID"),
    db: Session = Depends(get_db)
):
    """
    List all tools available for enablement.
    
    Returns tools from the Adapter with tenant-specific enablement status.
    """
    try:
        service = get_tool_access_service(db)
        tools = await service.get_available_tools_for_tenant(tenant_id)
        
        # Apply filters
        if category:
            tools = [t for t in tools if t.get("category") == category]
        
        if search:
            search_lower = search.lower()
            tools = [
                t for t in tools
                if search_lower in (t.get("name") or "").lower()
                or search_lower in (t.get("description") or "").lower()
                or search_lower in (t.get("provider") or "").lower()
            ]
        
        return tools
        
    except Exception as e:
        logger.error(f"Error listing available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enabled", response_model=List[TenantToolConfigResponse])
async def list_enabled_tools(
    tenant_id: UUID = Query(..., description="Tenant UUID"),
    db: Session = Depends(get_db)
):
    """
    List all tools enabled for a tenant.
    """
    try:
        configs = db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.enabled == True
        ).all()
        
        return [
            TenantToolConfigResponse(
                id=c.id,
                tenant_id=c.tenant_id,
                adapter_tool_id=c.tool_id,
                adapter_tool_name=c.adapter_tool_name,
                enabled=c.enabled,
                credential_id=c.credential_id,
                credential_configured=c.credential_id is not None,
                configuration=c.configuration or {},
                created_at=c.created_at,
                updated_at=c.updated_at
            )
            for c in configs
        ]
        
    except Exception as e:
        logger.error(f"Error listing enabled tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable", response_model=TenantToolConfigResponse)
async def enable_tool(
    request: ToolEnableRequest,
    tenant_id: UUID = Query(..., description="Tenant UUID"),
    db: Session = Depends(get_db)
):
    """
    Enable a tool for a tenant with credentials.
    
    This creates a 1:1 mapping between the tool and a credential.
    """
    try:
        service = get_tool_access_service(db)
        
        # Get tool info from Adapter
        tool_def = await service.adapter_client.get_tool(request.tool_id)
        if not tool_def:
            raise HTTPException(status_code=404, detail=f"Tool '{request.tool_id}' not found in Adapter")
        
        tool_name = tool_def.get("name") or request.tool_id
        credential_type_name = tool_def.get("credential_type")
        
        # Find or create credential type
        credential_type = None
        if credential_type_name:
            credential_type = db.query(CredentialType).filter(
                CredentialType.name == credential_type_name
            ).first()
        
        # Create credential
        store = CredentialStore(db)
        
        # Generate credential name
        cred_name = f"{tool_name} - {tenant_id.hex[:8]}"
        
        credential = store.create_credential(
            name=cred_name,
            credential_type_id=credential_type.id if credential_type else None,
            data=request.credentials,
            environment="production",
            user_id=str(tenant_id),
            metadata={"tenant_id": str(tenant_id), "tool_id": request.tool_id}
        )
        
        # Enable tool with credential
        config = await service.enable_tool_for_tenant(
            tenant_id=tenant_id,
            adapter_tool_id=request.tool_id,
            adapter_tool_name=tool_name,
            credential_id=credential.id,
            configuration=request.configuration
        )
        
        logger.info(f"Enabled tool '{request.tool_id}' for tenant {tenant_id}")
        
        return TenantToolConfigResponse(
            id=config.id,
            tenant_id=config.tenant_id,
            adapter_tool_id=config.tool_id,
            adapter_tool_name=config.adapter_tool_name,
            enabled=config.enabled,
            credential_id=config.credential_id,
            credential_configured=True,
            configuration=config.configuration or {},
            created_at=config.created_at,
            updated_at=config.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling tool: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{adapter_tool_id}/disable")
async def disable_tool(
    adapter_tool_id: str,
    tenant_id: UUID = Query(..., description="Tenant UUID"),
    db: Session = Depends(get_db)
):
    """
    Disable a tool for a tenant.
    
    This does NOT delete the credential - user can re-enable later.
    """
    try:
        service = get_tool_access_service(db)
        
        if service.disable_tool_for_tenant(tenant_id, adapter_tool_id):
            return {"success": True, "message": f"Tool '{adapter_tool_id}' disabled"}
        else:
            raise HTTPException(status_code=404, detail="Tool configuration not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Agent Tool Assignments
# =============================================================================

@router.get("/agents/{agent_id}/tools", response_model=List[AgentToolResponse])
async def get_agent_tools(
    agent_id: int,
    tenant_id: Optional[UUID] = Query(None, description="Tenant UUID (optional, will use agent's tenant)"),
    db: Session = Depends(get_db)
):
    """
    Get all tools assigned to an agent.
    """
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Use agent's tenant if not provided (for backward compatibility)
        effective_tenant_id = tenant_id or getattr(agent, 'tenant_id', UUID('00000000-0000-0000-0000-000000000000'))
        
        service = get_tool_access_service(db)
        assignments = await service.get_agent_assignments(agent_id, effective_tenant_id)
        
        return [
            AgentToolResponse(
                id=a["id"],
                adapter_tool_id=a["adapter_tool_id"],
                name=a["name"],
                description=a.get("description"),
                category=a.get("category"),
                icon=a.get("icon"),
                enabled=a["enabled"],
                assigned_at=a.get("assigned_at")
            )
            for a in assignments
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/tools", response_model=AgentToolAssignmentResponse)
async def assign_tool_to_agent(
    agent_id: int,
    request: AgentToolAssignRequest,
    tenant_id: Optional[UUID] = Query(None, description="Tenant UUID (optional, will use agent's tenant)"),
    db: Session = Depends(get_db)
):
    """
    Assign a tool to an agent.
    
    The tool must first be enabled at the tenant level.
    """
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Use agent's tenant if not provided
        effective_tenant_id = tenant_id or getattr(agent, 'tenant_id', UUID('00000000-0000-0000-0000-000000000000'))
        
        service = get_tool_access_service(db)
        assignment, error = service.assign_tool_to_agent(
            agent_id=agent_id,
            tenant_id=effective_tenant_id,
            adapter_tool_id=request.tool_id,
            enabled=request.enabled
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return AgentToolAssignmentResponse(
            id=assignment.id,
            agent_id=assignment.agent_id,
            adapter_tool_id=assignment.tool_id,
            enabled=assignment.enabled,
            created_at=assignment.created_at,
            updated_at=assignment.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning tool to agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}/tools/{adapter_tool_id}")
async def remove_tool_from_agent(
    agent_id: int,
    adapter_tool_id: str,
    db: Session = Depends(get_db)
):
    """
    Remove a tool assignment from an agent.
    """
    try:
        service = get_tool_access_service(db)
        
        if service.remove_tool_from_agent(agent_id, adapter_tool_id):
            return {"success": True, "message": f"Tool '{adapter_tool_id}' removed from agent {agent_id}"}
        else:
            raise HTTPException(status_code=404, detail="Assignment not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing tool from agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}/tools/batch")
async def batch_update_agent_tools(
    agent_id: int,
    request: AgentToolAssignBatchRequest,
    tenant_id: Optional[UUID] = Query(None, description="Tenant UUID (optional, will use agent's tenant)"),
    db: Session = Depends(get_db)
):
    """
    Batch update agent tool assignments.
    
    Useful for the UI when toggling multiple tools at once.
    """
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Use agent's tenant if not provided
        effective_tenant_id = tenant_id or getattr(agent, 'tenant_id', UUID('00000000-0000-0000-0000-000000000000'))
        
        service = get_tool_access_service(db)
        
        # Convert to dict format
        assignments = [
            {"adapter_tool_id": a.tool_id, "enabled": a.enabled}
            for a in request.assignments
        ]
        
        updated, errors = service.batch_update_agent_assignments(
            agent_id=agent_id,
            tenant_id=effective_tenant_id,
            assignments=assignments
        )
        
        return {
            "success": True,
            "updated": updated,
            "errors": errors if errors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error batch updating agent tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Credential Resolution (for Adapter callback)
# =============================================================================

@router.post("/credentials/resolve", response_model=CredentialResolveResponse)
async def resolve_credential(
    request: CredentialResolveRequest,
    db: Session = Depends(get_db)
):
    """
    Resolve credentials for a tool execution.
    
    This endpoint is called by the Unified Adapter when executing tools
    in "hosted" credential mode. The Adapter passes tenant_id and tool_name,
    and we return the decrypted credential data.
    
    Authentication: Service token (Bearer)
    """
    try:
        # Find the tenant's tool config
        config = db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == request.tenant_id,
            TenantToolConfig.tool_id == request.tool_name,
            TenantToolConfig.enabled == True
        ).first()
        
        if not config:
            return CredentialResolveResponse(
                success=False,
                error={
                    "code": "TOOL_NOT_FOUND",
                    "message": f"Tool '{request.tool_name}' not enabled for tenant"
                }
            )
        
        if not config.credential_id:
            return CredentialResolveResponse(
                success=False,
                error={
                    "code": "CREDENTIAL_NOT_FOUND",
                    "message": f"No credential configured for tool '{request.tool_name}'"
                }
            )
        
        # Resolve the credential
        store = CredentialStore(db)
        
        try:
            credential_data = store.get_decrypted_credential(
                credential_id=config.credential_id,
                service_name=request.service_name
            )
        except Exception as e:
            logger.error(f"Failed to decrypt credential: {e}")
            return CredentialResolveResponse(
                success=False,
                error={
                    "code": "CREDENTIAL_DECRYPT_FAILED",
                    "message": "Failed to decrypt credential"
                }
            )
        
        # Get credential metadata
        credential = db.query(Credential).filter(Credential.id == config.credential_id).first()
        
        return CredentialResolveResponse(
            success=True,
            data=credential_data,
            meta={
                "credential_id": config.credential_id,
                "credential_type": credential.credential_type.name if credential and credential.credential_type else None,
                "environment": credential.environment if credential else request.environment,
                "resolved_at": str(credential.updated_at) if credential else None
            }
        )
        
    except Exception as e:
        logger.error(f"Error resolving credential: {e}")
        return CredentialResolveResponse(
            success=False,
            error={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )


# =============================================================================
# Utility Endpoints
# =============================================================================

@router.get("/categories")
async def list_tool_categories(
    tenant_id: UUID = Query(..., description="Tenant UUID"),
    db: Session = Depends(get_db)
):
    """
    Get list of all tool categories with counts.
    """
    try:
        service = get_tool_access_service(db)
        tools = await service.get_available_tools_for_tenant(tenant_id)
        
        # Count by category
        counts: Dict[str, int] = {}
        for tool in tools:
            cat = tool.get("category") or "other"
            counts[cat] = counts.get(cat, 0) + 1
        
        return [
            {"name": name, "count": count}
            for name, count in sorted(counts.items())
        ]
        
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
