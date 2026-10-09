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
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import UserApiKey
from core.credentials.encryption import get_encryption_service
from core.llm import providers as provider_registry
from core.llm.byok_endpoint import EndpointRefused, clean_endpoint, endpoint_required
from core.llm.anthropic_workspace import (
    WORKSPACE_ID_MAX_LENGTH, WorkspaceIdRefused, clean_workspace_id, clear_cache, key_fingerprint, workspace_headers,
)
from config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/keys", tags=["API Keys"])

# PRD-236: the registry is the one list of providers (core/llm/providers.py).
SUPPORTED_PROVIDERS = provider_registry.byok_slugs()
# A customer's endpoint must answer the key check quickly or the save fails.
AZURE_CHECK_TIMEOUT_SECONDS = 15.0
KEY_VALID = "API key is valid"


# ── Pydantic schemas ─────────────────────────────────────────────────

class ApiKeyCreate(BaseModel):
    provider: str = Field(..., description="LLM provider name")
    api_key: str = Field(..., min_length=8, description="The raw API key")
    display_name: Optional[str] = Field(None, description="Friendly label")
    base_url: Optional[str] = Field(
        None, max_length=2048, description="The key's own endpoint (Azure: https://<resource>.openai.azure.com)"
    )
    workspace_id: Optional[str] = Field(
        None, max_length=WORKSPACE_ID_MAX_LENGTH,
        description="Anthropic only: the workspace an organization-level key bills (wrkspc_…)",
    )


class ApiKeyValidation(BaseModel):
    """Live provider-test result carried on a key save (PRD-222 US-006).

    The badge must never lie: a key is only trusted (``is_active`` / BYOK-enabled)
    when ``valid`` is True. The save response embeds this so the power-up card
    (US-013) renders the outcome in-flow, provider error text and all.
    """
    valid: bool
    message: str
    tested_at: Optional[datetime] = None


class ApiKeyOut(BaseModel):
    id: int
    provider: str
    display_name: Optional[str]
    masked_key: str  # e.g. sk-...abc1
    is_active: bool
    last_used_at: Optional[datetime]
    usage_count: int
    created_at: datetime
    # Populated only on the save response (US-006); None when listing keys.
    validation: Optional[ApiKeyValidation] = None

    class Config:
        from_attributes = True


class ApiKeyTestResult(BaseModel):
    valid: bool
    message: str
    provider: str


class PlatformKeyCreate(BaseModel):
    provider: str = Field(..., description="LLM provider name")
    api_key: str = Field(..., min_length=8, description="The raw API key")


# ── Helpers ───────────────────────────────────────────────────────────

def _mask_key(raw: str) -> str:
    """Return first 5 and last 4 chars with dots in between."""
    if len(raw) <= 12:
        return raw[:3] + "..." + raw[-2:]
    return raw[:5] + "..." + raw[-4:]


def _row_to_out(
    row: UserApiKey, encryption, validation: Optional[ApiKeyValidation] = None
) -> ApiKeyOut:
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
        validation=validation,
    )


def _openai_compatible_models_list(provider: str) -> bool:
    """True for registry providers validated by a models-list call on the OpenAI SDK."""
    spec = provider_registry.get_spec(provider)
    return bool(
        spec
        and spec.adapter == provider_registry.ADAPTER_OPENAI_COMPATIBLE
        and spec.validation == provider_registry.VALIDATION_MODELS_LIST
    )


def _key_endpoint(provider: str, raw: Optional[str]) -> Optional[str]:
    """The endpoint to save with a key (#873), or a 400 the dialog can show."""
    try:
        endpoint = clean_endpoint(provider, raw)
    except EndpointRefused as exc:
        raise HTTPException(400, str(exc)) from exc
    if endpoint is None and endpoint_required(provider):
        raise HTTPException(400, "Enter the endpoint for this key, e.g. https://<resource>.openai.azure.com")
    return endpoint


def _key_workspace(provider: str, raw: Optional[str]) -> Optional[str]:
    """The Anthropic workspace to save with a key, or a 400 the dialog can show."""
    try:
        return clean_workspace_id(provider, raw)
    except WorkspaceIdRefused as exc:
        raise HTTPException(400, str(exc)) from exc


def _check_openai(_provider: str, raw_key: str, _base_url: Optional[str], _workspace_id: Optional[str] = None) -> str:
    from openai import OpenAI
    OpenAI(api_key=raw_key).models.list()
    return KEY_VALID


def _check_anthropic(_provider: str, raw_key: str, _base_url: Optional[str], workspace_id: Optional[str] = None) -> str:
    """A models-list call, naming the workspace for an organization-level key (9 Oct 2026)."""
    import anthropic
    anthropic.Anthropic(api_key=raw_key, default_headers=workspace_headers(workspace_id)).models.list()
    return KEY_VALID


def _check_google(_provider: str, raw_key: str, _base_url: Optional[str], _workspace_id: Optional[str] = None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=raw_key)
    genai.list_models()
    return KEY_VALID


def _check_openai_compatible(
    provider: str, raw_key: str, _base_url: Optional[str], _workspace_id: Optional[str] = None,
) -> str:
    """OpenRouter, NVIDIA, DeepSeek: a models-list call against the provider's own base URL (PRD-236 S0.3)."""
    from openai import OpenAI
    spec = provider_registry.get_spec(provider)
    kwargs = {"api_key": raw_key, "base_url": provider_registry.base_url_for(spec.slug)}
    headers = provider_registry.headers_for(spec.slug)
    if headers:
        kwargs["default_headers"] = headers
    OpenAI(**kwargs).models.list()
    return KEY_VALID


def _check_azure(_provider: str, raw_key: str, base_url: Optional[str], _workspace_id: Optional[str] = None) -> str:
    """A models-list call on the key's own v1 endpoint, pinned on saas (#873).

    An endpoint that answers 404 to the list has accepted the key and simply has no
    list to give, so the key is saved and the message says it wasn't checked.
    """
    endpoint = base_url or config.AZURE_OPENAI_ENDPOINT
    if not endpoint:
        return "Key saved (no endpoint to check it against)"
    from openai import OpenAI
    from core.llm.byok_endpoint import endpoint_http_client
    from core.llm.clients.azure_v1 import v1_base_url

    kwargs = {"api_key": raw_key, "base_url": v1_base_url(endpoint), "timeout": AZURE_CHECK_TIMEOUT_SECONDS,
              "max_retries": 0}
    http_client = endpoint_http_client(AZURE_CHECK_TIMEOUT_SECONDS) if base_url else None
    if http_client:
        kwargs["http_client"] = http_client
    try:
        OpenAI(**kwargs).models.list()
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
        return "Key saved (this endpoint has no models list, so the key was not checked)"
    return KEY_VALID


_LIVE_CHECKS = {"openai": _check_openai, "anthropic": _check_anthropic, "google": _check_google, "azure": _check_azure}


def _live_check_for(provider: str):
    """The provider's live key check, or None when there is none to run."""
    if provider in _LIVE_CHECKS:
        return _LIVE_CHECKS[provider]
    return _check_openai_compatible if _openai_compatible_models_list(provider) else None


async def _validate_provider_key(
    provider: str, raw_key: str, base_url: Optional[str] = None, workspace_id: Optional[str] = None,
) -> ApiKeyValidation:
    """Make a real, minimal provider call to prove a BYOK key works (PRD-222 US-006).

    Shared by ``add_api_key`` (validate-on-save — the fix for the 2026-07-29
    dead-key incident where a provider-deleted key showed a healthy badge) and the
    manual ``test_api_key`` endpoint, so both speak one truth. A provider we have
    no live check for returns ``valid=True`` with an honest "not available"
    message — we never CLAIM a validation we did not run. Never raises: a failed
    call becomes ``valid=False`` carrying the provider's own error text.
    ``base_url`` is the key's own endpoint, for a provider that takes one (#873);
    ``workspace_id`` the Anthropic workspace an organization-level key bills.
    """
    provider = (provider or "").lower()
    tested_at = datetime.utcnow()
    check = _live_check_for(provider)
    if check is None:
        return ApiKeyValidation(
            valid=True,
            message="Key saved (live validation not available for this provider)",
            tested_at=tested_at,
        )
    try:
        message = check(provider, raw_key, base_url, workspace_id)
    except Exception as e:
        return ApiKeyValidation(valid=False, message=f"Invalid key: {str(e)[:200]}", tested_at=tested_at)
    return ApiKeyValidation(valid=True, message=message, tested_at=tested_at)


def _enable_proven_key(db: Session, workspace_id: Any, provider: str) -> None:
    """A key that passed its check switches BYOK on for its provider and converts the trial.

    Only a proven key auto-enables BYOK for this provider, and only a proven key
    converts the trial (US-006 hook into the US-005 ledger; a no-op for a converted
    or never-granted workspace).
    """
    from core.models.workspaces import Workspace
    from sqlalchemy.orm.attributes import flag_modified

    workspace = db.query(Workspace).get(workspace_id)
    if workspace:
        settings = dict(workspace.settings or {})
        overrides = dict(settings.get("byok_overrides", {}))
        overrides[provider] = True
        settings["byok_overrides"] = overrides
        workspace.settings = settings
        flag_modified(workspace, "settings")

    from services.trial_ledger import mark_trial_converted

    if mark_trial_converted(db, workspace_id):
        logger.info(f"Trial converted on validated key save workspace={workspace_id}")


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("", response_model=ApiKeyOut, status_code=201, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def add_api_key(
    body: ApiKeyCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Add a new BYOK API key for the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    provider = body.provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unsupported provider. Supported: {SUPPORTED_PROVIDERS}")

    endpoint = _key_endpoint(provider, body.base_url)
    workspace_id = _key_workspace(provider, body.workspace_id)
    encryption = get_encryption_service()
    encrypted = encryption.encrypt(body.api_key)

    # PRD-222 US-006 — validate BEFORE the key is trusted (the badge must never
    # lie). user_api_keys has no test_status/tested_at column and this branch
    # spends the ONE migration elsewhere, so the resolver-visible truth is
    # persisted where _resolve_api_key filters (is_active) and where the frontend
    # BYOK badge reads (is_active); tested_at rides on last_used_at. A dead key is
    # stored is_active=False → it never resolves and never wears a "BYOK" badge.
    validation = await _validate_provider_key(provider, body.api_key, endpoint, workspace_id=workspace_id)

    row = UserApiKey(
        workspace_id=ctx.workspace_id,
        provider=provider,
        encrypted_key=encrypted,
        base_url=endpoint,
        provider_workspace_id=workspace_id,
        key_fingerprint=key_fingerprint(body.api_key) if workspace_id else None,
        display_name=body.display_name or f"My {body.provider.title()} Key",
        is_active=validation.valid,
        usage_count=0,
        last_used_at=validation.tested_at if validation.valid else None,
    )
    db.add(row)

    if validation.valid:
        _enable_proven_key(db, ctx.workspace_id, provider)

    db.commit()
    db.refresh(row)
    clear_cache()

    # Nothing from the request body is logged (CodeQL py/clear-text-logging-sensitive-data: the body
    # carries the key): the provider is the registry's own slug, the outcome the branch taken.
    spec = provider_registry.get_spec(provider)
    slug = spec.slug if spec else "unknown"
    if validation.valid:
        logger.info("API key saved for provider=%s workspace=%s (valid, BYOK enabled)", slug, ctx.workspace_id)
    else:
        logger.info("API key saved for provider=%s workspace=%s (failed validation, BYOK NOT enabled)",
                    slug, ctx.workspace_id)
    return _row_to_out(row, encryption, validation=validation)


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


@router.delete("/{key_id}", status_code=204, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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
    clear_cache()
    logger.info(f"API key {key_id} deleted for workspace={ctx.workspace_id}")


@router.post("/{key_id}/test", response_model=ApiKeyTestResult, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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

    # Reuse the single validate-on-save path (PRD-222 US-006) so the manual test
    # and the save-time check can never diverge.
    result = await _validate_provider_key(
        row.provider, raw_key, getattr(row, "base_url", None),
        workspace_id=getattr(row, "provider_workspace_id", None),
    )
    if result.valid:
        row.last_used_at = result.tested_at or datetime.utcnow()
        db.commit()
    return ApiKeyTestResult(valid=result.valid, message=result.message, provider=row.provider)


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

    for provider in provider_registry.platform_key_slugs():
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


@router.get("/providers")
async def list_providers(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """The provider registry as the UI renders it (PRD-236 S0.6).

    Labels, key placeholders, docs links, the NVIDIA trial and rate-limit
    notes, and per-edition flags (``platform_key`` is False for byok_only
    providers in saas). Never a key, never a config attribute name.
    """
    return provider_registry.public_registry()


@router.put("/platform", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def set_platform_key(
    body: PlatformKeyCreate,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Create or update a platform-level API key for a provider.

    Stores an encrypted credential named ``{provider}_api`` in the
    credentials table so the CredentialResolver picks it up for all
    workspaces.
    """
    from core.models.credentials import Credential, CredentialType
    from sqlalchemy import and_

    provider = body.provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unsupported provider. Supported: {SUPPORTED_PROVIDERS}")
    if not provider_registry.platform_key_allowed(provider):
        # PRD-236 §Terms: NVIDIA's trial endpoint is BYO key only in saas.
        raise HTTPException(
            400,
            f"'{provider}' is a bring-your-own-key provider in this edition — "
            "add it under the workspace's own API keys instead of a platform key.",
        )

    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    cred_name = f"{provider}_api"

    try:
        encryption = get_encryption_service()
        encrypted_data = encryption.encrypt_dict({"api_key": body.api_key})

        existing = (
            db.query(Credential)
            .filter(and_(
                Credential.name == cred_name,
                Credential.is_active == True,
                Credential.workspace_id == ctx.workspace_id,
            ))
            .first()
        )

        if existing:
            existing.encrypted_data = encrypted_data
            existing.test_status = "not_tested"
            existing.updated_at = datetime.utcnow()
            db.commit()
            logger.info(f"Platform key updated for provider={provider}")
            try:
                from core.credentials.resolver import get_credential_resolver
                get_credential_resolver().clear_cache(cred_name)
            except Exception:
                pass
            return {"status": "updated", "provider": provider}

        cred_type = (
            db.query(CredentialType)
            .filter(CredentialType.name == cred_name)
            .first()
        )
        if not cred_type:
            cred_type = (
                db.query(CredentialType)
                .filter(CredentialType.name == "generic_api")
                .first()
            )
        if not cred_type:
            raise HTTPException(500, "No suitable credential type found — run seed data")

        new_cred = Credential(
            name=cred_name,
            credential_type_id=cred_type.id,
            workspace_id=ctx.workspace_id,
            encrypted_data=encrypted_data,
            environment="development",
            description=f"Platform API key for {provider}",
            is_active=True,
            test_status="not_tested",
            created_by=str(ctx.user.id) if ctx.user and ctx.user.id else None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(new_cred)
        db.commit()
        logger.info(f"Platform key created for provider={provider}")

        return {"status": "created", "provider": provider}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Platform key save failed for provider={provider}")
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)[:500]}",
        )


@router.delete("/platform/{provider}", status_code=200, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def remove_platform_key(
    provider: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove (deactivate) a platform-level API key for a provider."""
    from core.models.credentials import Credential
    from sqlalchemy import and_

    provider = provider.lower()
    cred_name = f"{provider}_api"

    # SECURITY: scope to caller's workspace to prevent cross-tenant deletion (OWASP A01:2021)
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    cred = (
        db.query(Credential)
        .filter(and_(
            Credential.name == cred_name,
            Credential.is_active == True,
            Credential.workspace_id == ctx.workspace_id,
        ))
        .first()
    )
    if not cred:
        raise HTTPException(404, f"No platform key found for {provider}")

    cred.is_active = False
    cred.updated_at = datetime.utcnow()
    db.commit()

    # Clear resolver cache
    try:
        from core.credentials.resolver import get_credential_resolver
        get_credential_resolver().clear_cache(cred_name)
    except Exception:
        pass

    logger.info(f"Platform key deactivated for provider={provider}")
    return {"status": "removed", "provider": provider}
