"""
Enhanced Credentials Management API
===================================

Comprehensive credential management endpoints inspired by n8n.
Supports credential types, CRUD operations, testing, and audit logging.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Any, Optional
import logging
import uuid

from database.database import get_db
from models.credentials import (
    CredentialType,
    Credential,
    CredentialAuditLog,
    CredentialTypeCreate,
    CredentialTypeResponse,
    CredentialCreate,
    CredentialUpdate,
    CredentialResponse,
    CredentialTestRequest,
    CredentialTestResponse,
    CredentialAuditLogResponse,
    CredentialResolveRequest,
    CredentialResolveResponse
)
from services.credential_service import (
    CredentialStore,
    CredentialNotFoundError,
    CredentialValidationError
)
from services.encryption_service import EncryptionKeyError
from utils.logging_adapter import set_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/credentials", tags=["🔐 Credential Management"])


# ============================================================================
# Dependency Injection
# ============================================================================

def get_credential_store(db: Session = Depends(get_db)) -> CredentialStore:
    """Get credential store instance"""
    return CredentialStore(db)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ============================================================================
# Credential Type Endpoints
# ============================================================================

@router.get("/types", response_model=List[CredentialTypeResponse])
async def list_credential_types(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Only active types"),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    List all available credential types.
    Returns all 400+ credential type definitions for dynamic form generation.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        types = store.list_credential_types(category=category, active_only=active_only)
        return [CredentialTypeResponse.from_orm(t) for t in types]
    
    except Exception as e:
        logger.error(f"Failed to list credential types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types/categories", response_model=List[str])
async def list_credential_categories(
    store: CredentialStore = Depends(get_credential_store)
):
    """Get list of all credential categories"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        from credential_types.all_credential_types import get_all_categories
        return get_all_categories()
    
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types/{type_id}", response_model=CredentialTypeResponse)
async def get_credential_type(
    type_id: int,
    store: CredentialStore = Depends(get_credential_store)
):
    """Get a specific credential type by ID"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        cred_type = store.get_credential_type(type_id)
        if not cred_type:
            raise HTTPException(status_code=404, detail="Credential type not found")
        
        return CredentialTypeResponse.from_orm(cred_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential type {type_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential Instance Endpoints
# ============================================================================

@router.get("/", response_model=List[CredentialResponse])
async def list_credentials(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    search: Optional[str] = Query(None, description="Search term"),
    category: Optional[str] = Query(None, description="Filter by category"),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    List all stored credentials with pagination and search.
    Returns decrypted credential data for editing.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        credentials = store.list_credentials(
            skip=skip, 
            limit=limit, 
            search=search, 
            category=category
        )
        
        # Convert to response format with decrypted data
        result = []
        for cred in credentials:
            cred_dict = {
                "id": cred.id,
                "name": cred.name,
                "credential_type_id": cred.credential_type_id,
                "credential_type_name": cred.credential_type.name if cred.credential_type else None,
                "category": cred.credential_type.category if cred.credential_type else None,
                "description": cred.description,
                "is_active": cred.is_active,
                "created_at": cred.created_at,
                "updated_at": cred.updated_at,
                "credential_data": cred.encrypted_data  # This will be decrypted by the service
            }
            result.append(cred_dict)
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to list credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential(
    credential_id: int,
    store: CredentialStore = Depends(get_credential_store)
):
    """Get a specific credential by ID with decrypted data"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        credential = store.get_credential(credential_id)
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        # Return with decrypted data for editing
        cred_dict = {
            "id": credential.id,
            "name": credential.name,
            "credential_type_id": credential.credential_type_id,
            "credential_type_name": credential.credential_type.name if credential.credential_type else None,
            "category": credential.credential_type.category if credential.credential_type else None,
            "description": credential.description,
            "is_active": credential.is_active,
            "created_at": credential.created_at,
            "updated_at": credential.updated_at,
            "credential_data": credential.encrypted_data  # This will be decrypted by the service
        }
        
        return cred_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=CredentialResponse)
async def create_credential(
    credential_data: CredentialCreate,
    request: Request,
    store: CredentialStore = Depends(get_credential_store)
):
    """Create a new credential instance"""
    set_request_id(str(uuid.uuid4()))
    client_ip = get_client_ip(request)
    
    try:
        credential = store.create_credential(
            name=credential_data.name,
            credential_type_id=credential_data.credential_type_id,
            credential_data=credential_data.credential_data,
            description=credential_data.description,
            user_id="system",  # TODO: Get from auth
            client_ip=client_ip
        )
        
        # Auto-activate matching MCP servers
        from services.mcp_auto_activation import MCPAutoActivationService
        mcp_service = MCPAutoActivationService(store.db)
        mcp_service.activate_matching_servers(credential.credential_type.name)
        
        return CredentialResponse.from_orm(credential)
    
    except CredentialValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create credential: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    credential_data: CredentialUpdate,
    request: Request,
    store: CredentialStore = Depends(get_credential_store)
):
    """Update an existing credential"""
    set_request_id(str(uuid.uuid4()))
    client_ip = get_client_ip(request)
    
    try:
        credential = store.update_credential(
            credential_id=credential_id,
            name=credential_data.name,
            credential_data=credential_data.credential_data,
            description=credential_data.description,
            user_id="system",  # TODO: Get from auth
            client_ip=client_ip
        )
        
        return CredentialResponse.from_orm(credential)
    
    except CredentialNotFoundError:
        raise HTTPException(status_code=404, detail="Credential not found")
    except CredentialValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: int,
    request: Request,
    store: CredentialStore = Depends(get_credential_store)
):
    """Delete a credential"""
    set_request_id(str(uuid.uuid4()))
    client_ip = get_client_ip(request)
    
    try:
        # Get credential before deletion for MCP deactivation
        credential = store.get_credential(credential_id)
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        # Deactivate matching MCP servers
        from services.mcp_auto_activation import MCPAutoActivationService
        mcp_service = MCPAutoActivationService(store.db)
        mcp_service.deactivate_matching_servers(credential.credential_type.name)
        
        store.delete_credential(
            credential_id=credential_id,
            user_id="system",  # TODO: Get from auth
            client_ip=client_ip
        )
        
        return {"message": "Credential deleted successfully"}
    
    except CredentialNotFoundError:
        raise HTTPException(status_code=404, detail="Credential not found")
    except Exception as e:
        logger.error(f"Failed to delete credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential Testing Endpoints
# ============================================================================

@router.post("/{credential_id}/test", response_model=CredentialTestResponse)
async def test_credential(
    credential_id: int,
    test_request: CredentialTestRequest,
    store: CredentialStore = Depends(get_credential_store)
):
    """Test a credential by making an actual API call"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        credential = store.get_credential(credential_id)
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        # Perform the test
        test_result = store.test_credential(credential_id, test_request.test_data)
        
        return CredentialTestResponse(
            success=test_result.success,
            message=test_result.message,
            response_data=test_result.response_data,
            test_duration=test_result.test_duration
        )
    
    except CredentialNotFoundError:
        raise HTTPException(status_code=404, detail="Credential not found")
    except Exception as e:
        logger.error(f"Failed to test credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential Resolution Endpoints
# ============================================================================

@router.post("/resolve", response_model=CredentialResolveResponse)
async def resolve_credential(
    resolve_request: CredentialResolveRequest,
    store: CredentialStore = Depends(get_credential_store)
):
    """Resolve a credential by name and return decrypted data"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        credential_data = store.resolve_credential_by_name(
            credential_name=resolve_request.credential_name,
            user_id=resolve_request.user_id
        )
        
        return CredentialResolveResponse(
            credential_name=resolve_request.credential_name,
            credential_data=credential_data,
            resolved_at=datetime.utcnow()
        )
    
    except CredentialNotFoundError:
        raise HTTPException(status_code=404, detail="Credential not found")
    except Exception as e:
        logger.error(f"Failed to resolve credential {resolve_request.credential_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Audit Log Endpoints
# ============================================================================

@router.get("/audit/logs", response_model=List[CredentialAuditLogResponse])
async def get_audit_logs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of records to return"),
    credential_id: Optional[int] = Query(None, description="Filter by credential ID"),
    action: Optional[str] = Query(None, description="Filter by action"),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential audit logs"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        logs = store.get_audit_logs(
            skip=skip,
            limit=limit,
            credential_id=credential_id,
            action=action
        )
        
        # Convert to response format
        result = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "credential_id": log.credential_id,
                "credential_name": log.credential.name if log.credential else f"Credential #{log.credential_id}",
                "action": log.action,
                "user_id": log.user_id,
                "client_ip": log.client_ip,
                "details": log.details,
                "timestamp": log.timestamp or log.created_at,
                "audit_metadata": log.audit_metadata
            }
            result.append(log_dict)
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics Endpoints
# ============================================================================

@router.get("/stats/summary")
async def get_credential_stats(
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential statistics summary"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        stats = store.get_credential_stats()
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get credential stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))