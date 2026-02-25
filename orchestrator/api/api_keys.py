"""
SDK API Key Management Endpoints (PRD-37)
==========================================

CRUD endpoints for workspace SDK API keys (public / server).
Keys are shown in full exactly once on creation, then only as masked prefixes.
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.services.api_key_service import ApiKeyService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])

VALID_PERMISSIONS = [
    "chat",
    "documents:read",
    "documents:write",
    "data:query",
    "data:execute",
    "agents:read",
    "agents:execute",
    "workflows:read",
    "workflows:execute",
]


# ── Pydantic schemas ─────────────────────────────────────────────────


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable key name")
    key_type: str = Field("server", description="'public' or 'server'")
    permissions: List[str] = Field(default_factory=list, description="Granted permission scopes")
    allowed_domains: Optional[List[str]] = Field(None, description="Origins allowed (required for public keys)")
    allowed_ips: Optional[List[str]] = Field(None, description="IP allowlist")
    rate_limit_requests: Optional[int] = Field(None, ge=1, description="Max requests per minute")
    rate_limit_tokens: Optional[int] = Field(None, ge=1, description="Max tokens per minute")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp (UTC)")

    @field_validator("key_type")
    @classmethod
    def validate_key_type(cls, v: str) -> str:
        if v not in ("public", "server"):
            raise ValueError("key_type must be 'public' or 'server'")
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v: List[str]) -> List[str]:
        invalid = [p for p in v if p not in VALID_PERMISSIONS]
        if invalid:
            raise ValueError(f"Invalid permissions: {invalid}. Valid: {VALID_PERMISSIONS}")
        return v


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    key: str = Field(..., description="Full API key — shown only on creation")
    key_type: str
    permissions: List[str]
    allowed_domains: Optional[List[str]] = None
    allowed_ips: Optional[List[str]] = None
    rate_limit_requests: Optional[int] = None
    rate_limit_tokens: Optional[int] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ApiKeyListItem(BaseModel):
    id: str
    name: str
    key_prefix: str = Field(..., description="Masked key prefix")
    key_type: str
    permissions: List[str]
    allowed_domains: Optional[List[str]] = None
    allowed_ips: Optional[List[str]] = None
    rate_limit_requests: Optional[int] = None
    rate_limit_tokens: Optional[int] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: Optional[bool] = None


class RevokeResponse(BaseModel):
    detail: str


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key(
    body: ApiKeyCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new SDK API key for the workspace. The full key is returned only once."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    # Public keys must declare allowed origins
    if body.key_type == "public":
        if not body.allowed_domains:
            raise HTTPException(
                422,
                "Public keys require a non-empty allowed_domains list",
            )

    result = ApiKeyService.create_api_key(
        db=db,
        workspace_id=ctx.workspace_id,
        name=body.name,
        key_type=body.key_type,
        permissions=body.permissions,
        allowed_domains=body.allowed_domains,
        allowed_ips=body.allowed_ips,
        rate_limit_requests=body.rate_limit_requests,
        rate_limit_tokens=body.rate_limit_tokens,
        expires_at=body.expires_at,
    )

    logger.info(
        "SDK API key created name=%s type=%s workspace=%s",
        body.name,
        body.key_type,
        ctx.workspace_id,
    )
    return ApiKeyCreateResponse(**result)


@router.get("", response_model=List[ApiKeyListItem])
async def list_api_keys(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all SDK API keys for the workspace (keys are masked)."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    keys = ApiKeyService.list_api_keys(db=db, workspace_id=ctx.workspace_id)
    return [ApiKeyListItem(**k) for k in keys]


@router.delete("/{key_id}", response_model=RevokeResponse)
async def revoke_api_key(
    key_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Revoke (deactivate) an SDK API key."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    revoked = ApiKeyService.revoke_api_key(
        db=db,
        key_id=key_id,
        workspace_id=ctx.workspace_id,
    )
    if not revoked:
        raise HTTPException(404, "API key not found")

    logger.info("SDK API key revoked key_id=%s workspace=%s", key_id, ctx.workspace_id)
    return RevokeResponse(detail="API key revoked")
