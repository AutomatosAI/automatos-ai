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
from datetime import datetime
import logging
import uuid

from core.database.database import get_db
from core.models.credentials import (
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
from core.credentials.service import (
    CredentialStore,
    CredentialNotFoundError,
    CredentialValidationError
)
from core.credentials.encryption import EncryptionKeyError
from core.utils.logging_adapter import set_request_id
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission

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


def _check_credential_workspace(cred, ctx: RequestContext) -> None:
    """Verify credential belongs to the caller's workspace (BOLA protection)."""
    if hasattr(cred, 'workspace_id') and cred.workspace_id and str(cred.workspace_id) != str(ctx.workspace_id):
        raise HTTPException(status_code=404, detail="Credential not found")


# ============================================================================
# Credential Type Endpoints
# ============================================================================

@router.get("/types", response_model=List[CredentialTypeResponse])
async def list_credential_types(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Only active types"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    List all available credential types.
    Returns all 400+ credential type definitions for dynamic form generation.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        types = store.list_credential_types(category=category, active_only=active_only)
        return [CredentialTypeResponse.model_validate(t) for t in types]
    
    except Exception as e:
        logger.error(f"Failed to list credential types: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/types/categories", response_model=List[str])
async def list_credential_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get list of all credential categories"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        from core.credentials.types import get_all_categories
        return get_all_categories()
    
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/types/{type_id}", response_model=CredentialTypeResponse)
async def get_credential_type(
    type_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get a specific credential type with full schema definition"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        cred_type = store.get_credential_type(type_id)
        if not cred_type:
            raise HTTPException(status_code=404, detail="Credential type not found")
        
        return CredentialTypeResponse.model_validate(cred_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential type {type_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/types/by-name/{type_name}", response_model=CredentialTypeResponse)
async def get_credential_type_by_name(
    type_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential type by name (e.g., 'openai_api', 'postgres_credentials')"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        cred_type = store.get_credential_type_by_name(type_name)
        if not cred_type:
            raise HTTPException(status_code=404, detail=f"Credential type '{type_name}' not found")
        
        return CredentialTypeResponse.model_validate(cred_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential type '{type_name}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Credential CRUD Endpoints
# ============================================================================

@router.post("/", response_model=CredentialResponse, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def handle_request(
    credential: CredentialCreate,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store),
    db: Session = Depends(get_db)
):
    """
    Create a new credential with encryption.
    Credential values are immediately encrypted and never stored in plaintext.
    """
    set_request_id(str(uuid.uuid4()))

    try:
        logger.info(f"Creating credential: credential_type_id={credential.credential_type_id}")

        ip_address = get_client_ip(request)
        user_id = getattr(ctx.user, "id", None)

        created_cred = store.create_credential(
            credential_data=credential,
            user_id=user_id,
            ip_address=ip_address,
            workspace_id=ctx.workspace_id,
        )

        # Build response (without decrypted values)
        cred_type = store.get_credential_type(created_cred.credential_type_id)

        # Extract field names from encrypted data
        decrypted = store.encryption_service.decrypt_dict(created_cred.encrypted_data)
        field_names = list(decrypted.keys())

        # Dispatch integration bridge (Shopify→Composio etc.) — fail-soft.
        bridge_result = store.dispatch_integration_bridge(
            credential_id=created_cred.id,
            workspace_id=ctx.workspace_id,
            credential_type_name=cred_type.name,
            decrypted_data=decrypted,
        )

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
            field_names=field_names,
            connection_status=bridge_result.status if bridge_result else None,
            connection_id=bridge_result.connection_id if bridge_result else None,
            auth_config_id=bridge_result.auth_config_id if bridge_result else None,
            auth_scheme=bridge_result.auth_scheme if bridge_result else None,
            oauth_redirect_url=bridge_result.oauth_redirect_url if bridge_result else None,
            connection_error=bridge_result.error if bridge_result else None,
        )

        return response
    
    except CredentialValidationError as e:
        logger.error(f"Credential validation error during creation: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Credential validation failed")
    except EncryptionKeyError as e:
        logger.error(f"Encryption error during credential creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Encryption error")
    except Exception as e:
        logger.error(f"Failed to create credential: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create credential")


@router.get("/")
async def get_item(
    ctx: RequestContext = Depends(get_request_context_hybrid), 
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
        query = db.query(Credential).filter(Credential.workspace_id == ctx.workspace_id)
        
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
            except Exception:
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
        raise HTTPException(status_code=500, detail="Failed to list credentials")


@router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential(
    credential_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential with decrypted values for editing"""
    set_request_id(str(uuid.uuid4()))

    # Restrict decrypted credential access to admin or API key auth
    if ctx.auth_type not in ("api_key",) and getattr(ctx.user, "system_role", "user") not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        cred = store.get_credential(credential_id)
        if not cred:
            raise HTTPException(status_code=404, detail="Credential not found")

        _check_credential_workspace(cred, ctx)

        cred_type = store.get_credential_type(cred.credential_type_id)

        try:
            decrypted = store.encryption_service.decrypt_dict(cred.encrypted_data)
            field_names = list(decrypted.keys())
        except Exception:
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
            credential_data=decrypted
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credential {credential_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{credential_id}", response_model=CredentialResponse, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def update_credential(
    credential_id: int,
    update_data: CredentialUpdate,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Update an existing credential"""
    set_request_id(str(uuid.uuid4()))

    try:
        # Workspace isolation
        existing = store.get_credential(credential_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Credential not found")
        _check_credential_workspace(existing, ctx)

        ip_address = get_client_ip(request)
        user_id = getattr(ctx.user, "id", None)

        updated_cred = store.update_credential(
            credential_id=credential_id,
            update_data=update_data,
            user_id=user_id,
            ip_address=ip_address
        )
        
        cred_type = store.get_credential_type(updated_cred.credential_type_id)

        decrypted: Dict[str, Any] = {}
        try:
            decrypted = store.encryption_service.decrypt_dict(updated_cred.encrypted_data)
        except Exception:
            decrypted = {}
        field_names = list(decrypted.keys())

        # Re-dispatch the integration bridge so a rotated token / changed
        # OAuth credentials reconnect (only when credential_data changed).
        bridge_result = None
        if update_data.credential_data is not None and decrypted:
            bridge_result = store.dispatch_integration_bridge(
                credential_id=updated_cred.id,
                workspace_id=ctx.workspace_id,
                credential_type_name=cred_type.name,
                decrypted_data=decrypted,
            )

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
            field_names=field_names,
            connection_status=bridge_result.status if bridge_result else None,
            connection_id=bridge_result.connection_id if bridge_result else None,
            auth_config_id=bridge_result.auth_config_id if bridge_result else None,
            auth_scheme=bridge_result.auth_scheme if bridge_result else None,
            oauth_redirect_url=bridge_result.oauth_redirect_url if bridge_result else None,
            connection_error=bridge_result.error if bridge_result else None,
        )
    
    except CredentialNotFoundError as e:
        logger.error(f"Credential {credential_id} not found during update: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Credential not found")
    except CredentialValidationError as e:
        logger.error(f"Credential validation error during update of {credential_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Credential validation failed")
    except Exception as e:
        logger.error(f"Failed to update credential {credential_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{credential_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def delete_credential(
    credential_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store),
    db: Session = Depends(get_db)
):
    """
    Securely delete a credential
    """
    set_request_id(str(uuid.uuid4()))

    try:
        # Workspace isolation
        existing = store.get_credential(credential_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Credential not found")
        _check_credential_workspace(existing, ctx)

        ip_address = get_client_ip(request)
        user_id = getattr(ctx.user, "id", None)

        store.delete_credential(
            credential_id=credential_id,
            user_id=user_id,
            ip_address=ip_address
        )
        
        return {"message": "Credential deleted successfully", "credential_id": credential_id}
    
    except CredentialNotFoundError as e:
        logger.error(f"Credential {credential_id} not found during deletion: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Credential not found")
    except Exception as e:
        logger.error(f"Failed to delete credential {credential_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Credential Testing
# ============================================================================

@router.post("/{credential_id}/test", response_model=CredentialTestResponse, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def test_credential(
    credential_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    Test a credential to verify it works.
    Tests database connections, API calls, etc. based on credential type.
    """
    set_request_id(str(uuid.uuid4()))

    try:
        # Workspace isolation
        existing = store.get_credential(credential_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Credential not found")
        _check_credential_workspace(existing, ctx)

        user_id = getattr(ctx.user, "id", None)
        result = await store.test_credential(credential_id=credential_id, user_id=user_id)
        return result
    
    except CredentialNotFoundError as e:
        logger.error(f"Credential {credential_id} not found during test: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail="Credential not found")
    except Exception as e:
        logger.error(f"Failed to test credential {credential_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Credential Resolution (Internal Use)
# ============================================================================

@router.post("/resolve", response_model=CredentialResolveResponse)
async def resolve_credential(
    resolve_request: CredentialResolveRequest,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """
    Resolve and decrypt a credential for service use.

    ⚠️ SECURITY WARNING: Returns decrypted credential values!
    Restricted to admin users and API key authentication only.
    All access is audited.
    """
    set_request_id(str(uuid.uuid4()))

    try:
        # Restrict to admin or API key auth
        if ctx.auth_type not in ("api_key",) and getattr(ctx.user, "system_role", "user") not in ("admin", "super_admin"):
            raise HTTPException(status_code=403, detail="Admin access required to resolve credentials")

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

        # SECURITY: verify resolved credential belongs to the caller's workspace
        # (OWASP A01:2021 Broken Access Control / BOLA protection)
        _check_credential_workspace(credential, ctx)
        
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
        logger.error(f"Credential validation error during resolution: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Credential validation failed")
    except EncryptionKeyError as e:
        logger.error(f"Decryption failed during credential resolution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Decryption failed")
    except Exception as e:
        logger.error(f"Failed to resolve credential: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resolve credential")


# ============================================================================
# Audit Logs
# ============================================================================

@router.get("/audit/logs", response_model=List[CredentialAuditLogResponse])
async def get_audit_logs(
    credential_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    store: CredentialStore = Depends(get_credential_store)
):
    """Get credential audit logs with filtering"""
    set_request_id(str(uuid.uuid4()))

    try:
        user_id = ctx.user.id
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
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/health")
async def credentials_health(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Health check for credentials service"""
    from core.credentials.encryption import get_encryption_service

    try:
        encryption_service = get_encryption_service()
        key_info = encryption_service.get_key_info()

        return {
            "status": "healthy",
            "service": "credentials",
            "encryption": {
                "status": "active",
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Credentials health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/cache/clear", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def clear_credential_cache(
    credential_name: Optional[str] = Query(None, description="Specific credential to clear"),
    ctx: RequestContext = Depends(get_request_context_hybrid)
):
    """Clear credential cache (useful after updates)"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        from core.credentials.resolver import get_credential_resolver
        resolver = get_credential_resolver()
        resolver.clear_cache(credential_name)
        
        return {
            "message": f"Cache cleared for {credential_name if credential_name else 'all credentials'}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats")
async def get_credential_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
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
        
        total_creds = db.query(func.count(Credential.id)).filter(
            Credential.workspace_id == ctx.workspace_id
        ).scalar()
        active_creds = db.query(func.count(Credential.id)).filter(
            Credential.is_active == True,
            Credential.workspace_id == ctx.workspace_id
        ).scalar()

        by_env = db.query(
            Credential.environment,
            func.count(Credential.id)
        ).filter(
            Credential.workspace_id == ctx.workspace_id
        ).group_by(Credential.environment).all()

        by_type = db.query(
            CredentialType.display_name,
            func.count(Credential.id)
        ).join(Credential).filter(
            Credential.workspace_id == ctx.workspace_id
        ).group_by(CredentialType.display_name).all()
        
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
        raise HTTPException(status_code=500, detail="Internal server error")

