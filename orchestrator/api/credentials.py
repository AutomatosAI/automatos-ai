"""
Enhanced Credentials Management API
===================================

Comprehensive credential management endpoints
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
    """Get a specific credential type with full schema definition"""
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


@router.get("/types/by-name/{type_name}", response_model=CredentialTypeResponse)
async def get_credential_type_by_name(
    type_name: str,
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential type by name (e.g., 'openai_api', 'postgres_credentials')"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        cred_type = store.get_credential_type_by_name(type_name)
        if not cred_type:
            raise HTTPException(status_code=404, detail=f"Credential type '{type_name}' not found")
        
        return CredentialTypeResponse.from_orm(cred_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential type '{type_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential CRUD Endpoints
# ============================================================================

@router.post("/", response_model=CredentialResponse)
async def create_credential(
    credential: CredentialCreate,
    request: Request,
    user_id: Optional[str] = Query(None, description="User ID"),
    store: CredentialStore = Depends(get_credential_store),
    db: Session = Depends(get_db)
):
    """
    Create a new credential with encryption.
    Credential values are immediately encrypted and never stored in plaintext.
    
    PRD-20: Auto-activates matching MCP servers when credential is created!
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Debug logging
        logger.info(f"Received credential: type={type(credential)}, credential_type_id={credential.credential_type_id}")
        logger.info(f"credential_data type: {type(credential.credential_data)}, value: {credential.credential_data}")
        
        ip_address = get_client_ip(request)
        
        created_cred = store.create_credential(
            credential_data=credential,
            user_id=user_id,
            ip_address=ip_address
        )
        
        # Build response (without decrypted values)
        cred_type = store.get_credential_type(created_cred.credential_type_id)
        
        # Extract field names from encrypted data
        decrypted = store.encryption_service.decrypt_dict(created_cred.encrypted_data)
        field_names = list(decrypted.keys())
        
        response = CredentialResponse(
            id=created_cred.id,
            name=created_cred.name,
            credential_type_id=created_cred.credential_type_id,
            credential_type_name=cred_type.name,
            credential_type_display_name=cred_type.display_name,
            environment=created_cred.environment,
            description=created_cred.description,
            tags=created_cred.tags or [],
            is_active=created_cred.is_active,
            expires_at=created_cred.expires_at,
            last_tested=created_cred.last_tested,
            test_status=created_cred.test_status,
            test_message=created_cred.test_message,
            created_by=created_cred.created_by,
            created_at=created_cred.created_at,
            updated_at=created_cred.updated_at,
            has_credentials=True,
            field_names=field_names
        )
        
        # PRD-20: AUTO-ACTIVATE matching MCP servers! 🚀
        try:
            from services.mcp_auto_activation import MCPAutoActivationService
            
            activation_service = MCPAutoActivationService(db)
            activation_result = await activation_service.activate_mcp_servers_for_credential(
                credential_id=created_cred.id,
                credential_type_name=cred_type.name
            )
            
            logger.info(f"🎉 Auto-activated {activation_result['activated_count']} MCP servers for credential '{created_cred.name}'")
            
        except Exception as e:
            # Don't fail credential creation if auto-activation fails
            logger.warning(f"⚠️  Auto-activation failed (credential still created): {e}")
        
        return response
    
    except CredentialValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EncryptionKeyError as e:
        raise HTTPException(status_code=500, detail=f"Encryption error: {e}")
    except Exception as e:
        import traceback
        logger.error(f"Failed to create credential: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_credentials(
    credential_type_id: Optional[int] = Query(None, description="Filter by type"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    active_only: bool = Query(True, description="Only active credentials"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in name or description"),
    skip: int = Query(0, ge=0, description="Number of items to skip (for pagination)"),
    limit: int = Query(20, ge=1, le=100, description="Number of items per page"),
    store: CredentialStore = Depends(get_credential_store),
    db: Session = Depends(get_db)
):
    """
    List all credentials with pagination (values are NEVER returned for security).
    Returns metadata only - use resolve endpoint to get actual values.
    
    PRD-20: Now with pagination support for 400+ credentials!
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Build query
        query = db.query(Credential)
        
        # Apply filters
        if credential_type_id:
            query = query.filter(Credential.credential_type_id == credential_type_id)
        if environment:
            query = query.filter(Credential.environment == environment)
        if active_only:
            query = query.filter(Credential.is_active == True)
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Credential.name.ilike(search_pattern),
                    Credential.description.ilike(search_pattern)
                )
            )
        if tags:
            tag_list = tags.split(',')
            for tag in tag_list:
                query = query.filter(Credential.tags.contains([tag]))
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        credentials = query.order_by(Credential.name).offset(skip).limit(limit).all()
        
        # Calculate pagination metadata
        pages = (total + limit - 1) // limit
        
        # Build responses
        responses = []
        for cred in credentials:
            cred_type = store.get_credential_type(cred.credential_type_id)
            
            # Get field names from credential type schema
            try:
                schema_fields = cred_type.schema_definition
                field_names = [field.get('name', '') for field in schema_fields if field.get('name')]
            except:
                field_names = []
            
            responses.append(CredentialResponse(
                id=cred.id,
                name=cred.name,
                credential_type_id=cred.credential_type_id,
                credential_type_name=cred_type.name,
                credential_type_display_name=cred_type.display_name,
                environment=cred.environment,
                description=cred.description,
                tags=cred.tags or [],
                is_active=cred.is_active,
                expires_at=cred.expires_at,
                last_tested=cred.last_tested,
                test_status=cred.test_status,
                test_message=cred.test_message,
                created_by=cred.created_by,
                created_at=cred.created_at,
                updated_at=cred.updated_at,
                has_credentials=True,
                field_names=field_names
            ))
        
        # Return paginated response
        return {
            "items": responses,
            "total": total,
            "skip": skip,
            "limit": limit,
            "pages": pages,
            "current_page": (skip // limit) + 1 if limit > 0 else 1
        }
    
    except Exception as e:
        logger.error(f"Failed to list credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential(
    credential_id: int,
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential with decrypted values for editing"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        cred = store.get_credential(credential_id)
        if not cred:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        cred_type = store.get_credential_type(cred.credential_type_id)
        
        try:
            decrypted = store.encryption_service.decrypt_dict(cred.encrypted_data)
            field_names = list(decrypted.keys())
        except:
            field_names = []
            decrypted = {}
        
        return CredentialResponse(
            id=cred.id,
            name=cred.name,
            credential_type_id=cred.credential_type_id,
            credential_type_name=cred_type.name,
            credential_type_display_name=cred_type.display_name,
            environment=cred.environment,
            description=cred.description,
            tags=cred.tags or [],
            is_active=cred.is_active,
            expires_at=cred.expires_at,
            last_tested=cred.last_tested,
            test_status=cred.test_status,
            test_message=cred.test_message,
            created_by=cred.created_by,
            created_at=cred.created_at,
            updated_at=cred.updated_at,
            has_credentials=True,
            field_names=field_names,
            credential_data=decrypted  # Add the decrypted data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    update_data: CredentialUpdate,
    request: Request,
    user_id: Optional[str] = Query(None),
    store: CredentialStore = Depends(get_credential_store)
):
    """Update an existing credential"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        ip_address = get_client_ip(request)
        
        updated_cred = store.update_credential(
            credential_id=credential_id,
            update_data=update_data,
            user_id=user_id,
            ip_address=ip_address
        )
        
        cred_type = store.get_credential_type(updated_cred.credential_type_id)
        
        try:
            decrypted = store.encryption_service.decrypt_dict(updated_cred.encrypted_data)
            field_names = list(decrypted.keys())
        except:
            field_names = []
        
        return CredentialResponse(
            id=updated_cred.id,
            name=updated_cred.name,
            credential_type_id=updated_cred.credential_type_id,
            credential_type_name=cred_type.name,
            credential_type_display_name=cred_type.display_name,
            environment=updated_cred.environment,
            description=updated_cred.description,
            tags=updated_cred.tags or [],
            is_active=updated_cred.is_active,
            expires_at=updated_cred.expires_at,
            last_tested=updated_cred.last_tested,
            test_status=updated_cred.test_status,
            test_message=updated_cred.test_message,
            created_by=updated_cred.created_by,
            created_at=updated_cred.created_at,
            updated_at=updated_cred.updated_at,
            has_credentials=True,
            field_names=field_names
        )
    
    except CredentialNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CredentialValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: int,
    request: Request,
    user_id: Optional[str] = Query(None),
    store: CredentialStore = Depends(get_credential_store),
    db: Session = Depends(get_db)
):
    """
    Securely delete a credential
    
    PRD-20: Auto-deactivates linked MCP servers when credential is deleted!
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        ip_address = get_client_ip(request)
        
        # PRD-20: DEACTIVATE linked MCP servers before deleting credential
        try:
            from services.mcp_auto_activation import MCPAutoActivationService
            
            activation_service = MCPAutoActivationService(db)
            deactivation_result = await activation_service.deactivate_mcp_servers_for_credential(
                credential_id=credential_id
            )
            
            if deactivation_result['deactivated_count'] > 0:
                logger.info(f"🔌 Auto-deactivated {deactivation_result['deactivated_count']} MCP servers")
            
        except Exception as e:
            # Don't fail deletion if deactivation fails
            logger.warning(f"⚠️  Auto-deactivation failed (continuing with deletion): {e}")
        
        # Delete the credential
        store.delete_credential(
            credential_id=credential_id,
            user_id=user_id,
            ip_address=ip_address
        )
        
        return {"message": "Credential deleted successfully", "credential_id": credential_id}
    
    except CredentialNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential Testing
# ============================================================================

@router.post("/{credential_id}/test", response_model=CredentialTestResponse)
async def test_credential(
    credential_id: int,
    user_id: Optional[str] = Query(None),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    Test a credential to verify it works.
    Tests database connections, API calls, etc. based on credential type.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        result = await store.test_credential(credential_id=credential_id, user_id=user_id)
        return result
    
    except CredentialNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to test credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Credential Resolution (Internal Use)
# ============================================================================

@router.post("/resolve", response_model=CredentialResolveResponse)
async def resolve_credential(
    resolve_request: CredentialResolveRequest,
    request: Request,
    store: CredentialStore = Depends(get_credential_store)
):
    """
    Resolve and decrypt a credential for service use.
    
    ⚠️ SECURITY WARNING: Returns decrypted credential values!
    This endpoint should only be used by internal services.
    All access is audited.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        ip_address = get_client_ip(request)
        
        # Get credential by ID or name
        if resolve_request.credential_id:
            credential = store.get_credential(resolve_request.credential_id)
            if not credential:
                raise HTTPException(status_code=404, detail="Credential not found")
        
        elif resolve_request.credential_name:
            credential = store.get_credential_by_name(
                resolve_request.credential_name,
                resolve_request.environment
            )
            if not credential:
                raise HTTPException(
                    status_code=404,
                    detail=f"Credential '{resolve_request.credential_name}' not found"
                )
        else:
            raise HTTPException(status_code=400, detail="Must provide credential_id or credential_name")
        
        # Decrypt credential data
        decrypted_data = store.get_decrypted_credential(
            credential_id=credential.id,
            user_id=resolve_request.service_name,
            ip_address=ip_address,
            service_name=resolve_request.service_name
        )
        
        cred_type = store.get_credential_type(credential.credential_type_id)
        
        return CredentialResolveResponse(
            credential_id=credential.id,
            credential_name=credential.name,
            credential_type=cred_type.name,
            data=decrypted_data,
            environment=credential.environment,
            resolved_at=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except CredentialValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EncryptionKeyError as e:
        raise HTTPException(status_code=500, detail=f"Decryption failed: {e}")
    except Exception as e:
        logger.error(f"Failed to resolve credential: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Audit Logs
# ============================================================================

@router.get("/audit/logs", response_model=List[CredentialAuditLogResponse])
async def get_audit_logs(
    credential_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential audit logs with filtering"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        logs = store.get_audit_logs(
            credential_id=credential_id,
            action=action,
            user_id=user_id,
            limit=limit
        )
        
        responses = []
        for log in logs:
            credential = store.get_credential(log.credential_id)
            
            responses.append(CredentialAuditLogResponse(
                id=log.id,
                credential_id=str(log.credential_id),
                tool_id=log.tool_id,
                action=log.action,
                user_id=log.user_id,
                ip_address=log.ip_address,
                success=log.success,
                error_message=log.error_message,
                details=log.details,
                created_at=log.timestamp
            ))
        
        return responses
    
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/health")
async def credentials_health():
    """Health check for credentials service"""
    from services.encryption_service import get_encryption_service
    
    try:
        encryption_service = get_encryption_service()
        key_info = encryption_service.get_key_info()
        
        return {
            "status": "healthy",
            "service": "credentials",
            "encryption": {
                "status": "active",
                "algorithm": key_info["algorithm"],
                "key_source": key_info["source"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/cache/clear")
async def clear_credential_cache(
    credential_name: Optional[str] = Query(None, description="Specific credential to clear")
):
    """Clear credential cache (useful after updates)"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        from services.credential_resolver import get_credential_resolver
        resolver = get_credential_resolver()
        resolver.clear_cache(credential_name)
        
        return {
            "message": f"Cache cleared for {credential_name if credential_name else 'all credentials'}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_credential_stats(
    db: Session = Depends(get_db)
):
    """Get credential system statistics"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        from sqlalchemy import func
        
        total_types = db.query(func.count(CredentialType.id)).scalar()
        active_types = db.query(func.count(CredentialType.id)).filter(
            CredentialType.is_active == True
        ).scalar()
        
        total_creds = db.query(func.count(Credential.id)).scalar()
        active_creds = db.query(func.count(Credential.id)).filter(
            Credential.is_active == True
        ).scalar()
        
        by_env = db.query(
            Credential.environment,
            func.count(Credential.id)
        ).group_by(Credential.environment).all()
        
        by_type = db.query(
            CredentialType.display_name,
            func.count(Credential.id)
        ).join(Credential).group_by(CredentialType.display_name).all()
        
        return {
            "credential_types": {
                "total": total_types,
                "active": active_types
            },
            "credentials": {
                "total": total_creds,
                "active": active_creds,
                "by_environment": {env: count for env, count in by_env},
                "by_type": {type_name: count for type_name, count in by_type}
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get credential stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

