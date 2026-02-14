"""
BYOK API Keys Management (PRD-54)
==================================

CRUD endpoints for user-provided API keys (Bring Your Own Key).
Keys are encrypted at rest using the platform's EncryptionService.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import UserApiKey
from core.credentials.encryption import get_encryption_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/keys", tags=["API Keys"])

SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "openrouter", "azure", "grok"]


# ── Pydantic schemas ─────────────────────────────────────────────────

class ApiKeyCreate(BaseModel):
    provider: str = Field(..., description="LLM provider name")
    api_key: str = Field(..., min_length=8, description="The raw API key")
    display_name: Optional[str] = Field(None, description="Friendly label")


class ApiKeyOut(BaseModel):
    id: int
    provider: str
    display_name: Optional[str]
    masked_key: str  # e.g. sk-...abc1
    is_active: bool
    last_used_at: Optional[datetime]
    usage_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyTestResult(BaseModel):
    valid: bool
    message: str
    provider: str


# ── Helpers ───────────────────────────────────────────────────────────

def _mask_key(raw: str) -> str:
    """Return first 5 and last 4 chars with dots in between."""
    if len(raw) <= 12:
        return raw[:3] + "..." + raw[-2:]
    return raw[:5] + "..." + raw[-4:]


def _row_to_out(row: UserApiKey, encryption) -> ApiKeyOut:
    try:
        raw = encryption.decrypt(row.encrypted_key)
    except Exception:
        raw = "********"
    return ApiKeyOut(
        id=row.id,
        provider=row.provider,
        display_name=row.display_name,
        masked_key=_mask_key(raw),
        is_active=row.is_active,
        last_used_at=row.last_used_at,
        usage_count=row.usage_count or 0,
        created_at=row.created_at,
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("", response_model=ApiKeyOut, status_code=201)
async def add_api_key(
    body: ApiKeyCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Add a new BYOK API key for the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    if body.provider.lower() not in SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unsupported provider. Supported: {SUPPORTED_PROVIDERS}")

    encryption = get_encryption_service()
    encrypted = encryption.encrypt(body.api_key)

    row = UserApiKey(
        workspace_id=ctx.workspace_id,
        provider=body.provider.lower(),
        encrypted_key=encrypted,
        display_name=body.display_name or f"My {body.provider.title()} Key",
        is_active=True,
        usage_count=0,
    )
    db.add(row)

    # Auto-enable BYOK for this provider on the workspace
    from core.models.workspaces import Workspace
    from sqlalchemy.orm.attributes import flag_modified

    workspace = db.query(Workspace).get(ctx.workspace_id)
    if workspace:
        settings = dict(workspace.settings or {})
        overrides = dict(settings.get("byok_overrides", {}))
        overrides[body.provider.lower()] = True
        settings["byok_overrides"] = overrides
        workspace.settings = settings
        flag_modified(workspace, "settings")

    db.commit()
    db.refresh(row)

    logger.info(f"API key added for provider={body.provider} workspace={ctx.workspace_id} (BYOK auto-enabled)")
    return _row_to_out(row, encryption)


@router.get("", response_model=List[ApiKeyOut])
async def list_api_keys(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all API keys for the current workspace (masked)."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    rows = (
        db.query(UserApiKey)
        .filter(UserApiKey.workspace_id == ctx.workspace_id)
        .order_by(UserApiKey.created_at.desc())
        .all()
    )
    encryption = get_encryption_service()
    return [_row_to_out(r, encryption) for r in rows]


@router.delete("/{key_id}", status_code=204)
async def delete_api_key(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a BYOK API key."""
    row = (
        db.query(UserApiKey)
        .filter(UserApiKey.id == key_id, UserApiKey.workspace_id == ctx.workspace_id)
        .first()
    )
    if not row:
        raise HTTPException(404, "API key not found")

    db.delete(row)
    db.commit()
    logger.info(f"API key {key_id} deleted for workspace={ctx.workspace_id}")


@router.post("/{key_id}/test", response_model=ApiKeyTestResult)
async def test_api_key(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Test if a BYOK API key is valid by making a minimal API call."""
    row = (
        db.query(UserApiKey)
        .filter(UserApiKey.id == key_id, UserApiKey.workspace_id == ctx.workspace_id)
        .first()
    )
    if not row:
        raise HTTPException(404, "API key not found")

    encryption = get_encryption_service()
    raw_key = encryption.decrypt(row.encrypted_key)

    try:
        if row.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=raw_key)
            client.models.list()
        elif row.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=raw_key)
            client.models.list()
        elif row.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=raw_key)
            genai.list_models()
        elif row.provider == "openrouter":
            from openai import OpenAI
            client = OpenAI(
                api_key=raw_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://automatos.app", "X-Title": "Automatos AI"},
            )
            client.models.list()
        else:
            return ApiKeyTestResult(valid=True, message="Key saved (validation not available for this provider)", provider=row.provider)

        # Mark last used
        row.last_used_at = datetime.utcnow()
        db.commit()

        return ApiKeyTestResult(valid=True, message="API key is valid", provider=row.provider)

    except Exception as e:
        return ApiKeyTestResult(valid=False, message=f"Invalid key: {str(e)[:200]}", provider=row.provider)


@router.get("/platform-status")
async def get_platform_key_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Return which providers have platform-level API keys configured.

    Checks the credential store for each supported provider.
    Returns boolean only — never exposes key values.
    """
    from core.credentials.resolver import get_credential_resolver

    resolver = get_credential_resolver()
    result = {}

    for provider in SUPPORTED_PROVIDERS:
        configured = False
        cred_names = [
            f"development_{provider}_api",
            f"development_{provider}",
            f"{provider}_api",
            provider,
        ]
        for cred_name in cred_names:
            try:
                key = resolver.get_credential_field(cred_name, "api_key")
                if not key:
                    key = resolver.get_credential_field(cred_name, "api_token")
                if key:
                    configured = True
                    break
            except Exception:
                continue
        result[provider] = {"configured": configured}

    return {"platform_keys": result}
