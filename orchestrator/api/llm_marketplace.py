"""
LLM Marketplace API (PRD-54)
=============================

Browse, compare, and install LLM models to workspaces.
Extends the existing marketplace with LLM-specific endpoints.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import LLMModel, WorkspaceModel
from core.models.openrouter_cache import OpenRouterModelCache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/marketplace/llm", tags=["LLM Marketplace"])


# ── Pydantic schemas ─────────────────────────────────────────────────

class LLMModelOut(BaseModel):
    id: int
    provider: str
    model_id: str
    display_name: str
    model_family: Optional[str]
    description: Optional[str]
    context_window: int
    max_output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    recommended_for: List[str] = Field(default_factory=list)
    supports_functions: bool
    supports_vision: bool
    supports_streaming: bool
    status: str
    tier: Optional[str]
    category: Optional[str]
    tags: List[str] = Field(default_factory=list)
    is_featured: bool = False
    is_default: bool = False
    requires_plan: Optional[str]
    install_count: int = 0
    is_installed: bool = False  # populated per-request based on workspace

    class Config:
        from_attributes = True


class CompareOut(BaseModel):
    models: List[LLMModelOut]


class InstallResult(BaseModel):
    success: bool
    message: str
    model_id: str


# ── Helpers ───────────────────────────────────────────────────────────

def _model_to_out(m: LLMModel, installed_model_ids: set = None) -> LLMModelOut:
    return LLMModelOut(
        id=m.id,
        provider=m.provider,
        model_id=m.model_id,
        display_name=m.display_name,
        model_family=m.model_family,
        description=m.description,
        context_window=m.context_window,
        max_output_tokens=m.max_output_tokens,
        input_cost_per_1k=float(m.input_cost_per_1k_tokens or 0),
        output_cost_per_1k=float(m.output_cost_per_1k_tokens or 0),
        capabilities=m.capabilities or {},
        recommended_for=m.recommended_for or [],
        supports_functions=m.supports_functions or False,
        supports_vision=m.supports_vision or False,
        supports_streaming=m.supports_streaming if m.supports_streaming is not None else True,
        status=m.status or "active",
        tier=m.tier,
        category=m.category,
        tags=m.tags or [],
        is_featured=m.is_featured or False,
        is_default=m.is_default or False,
        requires_plan=m.requires_plan,
        install_count=m.install_count or 0,
        is_installed=m.id in (installed_model_ids or set()),
    )


def _get_or_create_from_cache(db: Session, model_id: str) -> Optional[LLMModel]:
    """
    Find model in llm_models, or auto-create from openrouter_models_cache.
    This bridges the two-table architecture: marketplace browses from OpenRouter
    cache, but install/workspace tracking uses the llm_models table.
    """
    m = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
    if m:
        return m

    # Try OpenRouter cache
    cached = db.query(OpenRouterModelCache).filter(
        OpenRouterModelCache.model_id == model_id
    ).first()
    if not cached:
        return None

    # Auto-create LLMModel from cache data
    m = LLMModel(
        provider=cached.provider,
        model_id=cached.model_id,
        display_name=cached.display_name,
        description=cached.description,
        model_family=cached.provider,
        context_window=cached.context_length or 0,
        max_output_tokens=cached.max_completion_tokens or 0,
        input_cost_per_1k_tokens=(cached.prompt_cost or 0) * 1000,
        output_cost_per_1k_tokens=(cached.completion_cost or 0) * 1000,
        supports_functions=cached.supports_tools or False,
        supports_vision=cached.supports_vision or False,
        supports_streaming=cached.supports_streaming if cached.supports_streaming is not None else True,
        status="active",
        tier="aggregator",
        category=cached.category,
        tags=cached.tags or [],
        capabilities={},
        recommended_for=[],
        external_id=cached.model_id,
    )
    db.add(m)
    db.flush()
    logger.info(f"Auto-created LLMModel from OpenRouter cache: {model_id}")
    return m


def _get_available_providers(db: Session, workspace_id) -> set:
    """
    Return the set of provider names the workspace can actually use.

    Source of truth is the DB — NOT env vars (those are legacy).
    Checks: 1) BYOK keys (UserApiKey), 2) Credential store entries.
    """
    available: set = set()

    ALL_PROVIDERS = [
        "openai", "anthropic", "google", "openrouter", "deepseek",
        "azure", "bedrock", "grok", "x-ai", "cohere", "huggingface",
        "meta-llama", "qwen",
    ]

    # 1. BYOK keys stored in DB for this workspace
    if workspace_id:
        try:
            from core.models.core import UserApiKey
            byok_providers = (
                db.query(UserApiKey.provider)
                .filter(
                    UserApiKey.workspace_id == workspace_id,
                    UserApiKey.is_active == True,
                )
                .distinct()
                .all()
            )
            available.update(p[0] for p in byok_providers)
        except Exception:
            pass  # table may not exist yet

    # 2. Credential store entries (platform keys added via Settings)
    try:
        from core.credentials.resolver import get_credential_resolver
        resolver = get_credential_resolver()
        for provider in ALL_PROVIDERS:
            if provider in available:
                continue
            for variation in [f"{provider}_api", provider]:
                try:
                    key = resolver.get_credential_field(variation, "api_key", silent=True)
                    if key:
                        available.add(provider)
                        break
                except Exception:
                    continue
    except Exception:
        pass

    return available


def _get_installed_ids(db: Session, workspace_id) -> set:
    if not workspace_id:
        return set()
    rows = (
        db.query(WorkspaceModel.model_id)
        .filter(WorkspaceModel.workspace_id == workspace_id, WorkspaceModel.is_active == True)
        .all()
    )
    return {r[0] for r in rows}


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/models", response_model=List[LLMModelOut])
async def browse_models(
    provider: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    min_context: Optional[int] = Query(None),
    max_cost: Optional[float] = Query(None, description="Max input cost per 1K tokens"),
    capability: Optional[str] = Query(None, description="Required capability key"),
    search: Optional[str] = Query(None),
    sort_by: Optional[str] = Query("popularity", description="cost|context|popularity|name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Browse LLM models with rich filtering."""
    query = db.query(LLMModel).filter(LLMModel.status == "active")

    if provider:
        query = query.filter(LLMModel.provider == provider)
    if category:
        query = query.filter(LLMModel.category == category)
    if tier:
        query = query.filter(LLMModel.tier == tier)
    if min_context:
        query = query.filter(LLMModel.context_window >= min_context)
    if max_cost is not None:
        query = query.filter(LLMModel.input_cost_per_1k_tokens <= max_cost)
    if capability:
        # Filter models whose JSON capabilities dict contains the requested key
        query = query.filter(LLMModel.capabilities[capability].isnot(None))
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            or_(
                LLMModel.display_name.ilike(pattern),
                LLMModel.model_id.ilike(pattern),
                LLMModel.description.ilike(pattern),
            )
        )

    # Sorting
    if sort_by == "cost":
        query = query.order_by(LLMModel.input_cost_per_1k_tokens.asc())
    elif sort_by == "context":
        query = query.order_by(LLMModel.context_window.desc())
    elif sort_by == "name":
        query = query.order_by(LLMModel.display_name.asc())
    else:  # popularity
        query = query.order_by(LLMModel.install_count.desc(), LLMModel.is_featured.desc())

    models = query.offset(offset).limit(limit).all()
    installed = _get_installed_ids(db, ctx.workspace_id)
    return [_model_to_out(m, installed) for m in models]


@router.get("/installed", response_model=List[LLMModelOut])
async def get_installed_models(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get usable models for the current workspace.

    Two-gate filter:
      1. Provider gate — does the provider have an API key? (env, BYOK, cred store)
      2. Model gate   — has the user installed this model to the workspace?

    Result: only models the user explicitly chose AND can actually call.
    """
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    installed_ids = _get_installed_ids(db, ctx.workspace_id)
    available_providers = _get_available_providers(db, ctx.workspace_id)

    if not installed_ids:
        # Workspace has no models installed yet — return empty so the UI
        # can prompt the user to visit the marketplace.
        return []

    # Fetch only workspace-installed models
    models = (
        db.query(LLMModel)
        .filter(
            LLMModel.status == "active",
            LLMModel.id.in_(installed_ids),
        )
        .order_by(LLMModel.provider, LLMModel.display_name)
        .all()
    )

    # Gate 1: filter to providers with an API key.
    # Use the model's actual provider (e.g. "google", "anthropic") even for
    # aggregator-routed models, so users only see providers they have keys for.
    # Special case: models whose provider IS "openrouter" (e.g. Auto Router)
    # are shown when the openrouter key exists.
    usable = []
    for m in models:
        provider = (m.provider or "").lower()
        if provider in available_providers:
            usable.append(m)

    return [_model_to_out(m, installed_ids) for m in usable]


@router.get("/installed-ids")
async def get_installed_model_ids(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return just the model_id strings installed in this workspace (lightweight)."""
    if not ctx.workspace_id:
        return {"model_ids": []}
    installed_llm_ids = _get_installed_ids(db, ctx.workspace_id)
    if not installed_llm_ids:
        return {"model_ids": []}
    rows = (
        db.query(LLMModel.model_id)
        .filter(LLMModel.id.in_(installed_llm_ids))
        .all()
    )
    return {"model_ids": [r[0] for r in rows]}


@router.get("/models/{model_id:path}", response_model=LLMModelOut)
async def get_model_detail(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full detail for a single model."""
    m = _get_or_create_from_cache(db, model_id)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}")
    installed = _get_installed_ids(db, ctx.workspace_id)
    return _model_to_out(m, installed)


@router.post("/models/{model_id:path}/install", response_model=InstallResult)
async def install_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Install a model to the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    m = _get_or_create_from_cache(db, model_id)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}")

    existing = (
        db.query(WorkspaceModel)
        .filter(WorkspaceModel.workspace_id == ctx.workspace_id, WorkspaceModel.model_id == m.id)
        .first()
    )
    if existing:
        if not existing.is_active:
            existing.is_active = True
            db.commit()
            return InstallResult(success=True, message="Model re-activated", model_id=model_id)
        return InstallResult(success=True, message="Model already installed", model_id=model_id)

    wm = WorkspaceModel(
        workspace_id=ctx.workspace_id,
        model_id=m.id,
        source="marketplace",
    )
    db.add(wm)
    m.install_count = (m.install_count or 0) + 1
    db.commit()

    logger.info(f"Model {model_id} installed to workspace {ctx.workspace_id}")
    return InstallResult(success=True, message="Model installed", model_id=model_id)


@router.post("/models/{model_id:path}/uninstall", response_model=InstallResult)
async def uninstall_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a model from the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    m = _get_or_create_from_cache(db, model_id)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}")

    wm = (
        db.query(WorkspaceModel)
        .filter(WorkspaceModel.workspace_id == ctx.workspace_id, WorkspaceModel.model_id == m.id)
        .first()
    )
    if not wm:
        return InstallResult(success=True, message="Model was not installed", model_id=model_id)

    wm.is_active = False
    db.commit()
    return InstallResult(success=True, message="Model uninstalled", model_id=model_id)


@router.get("/compare", response_model=CompareOut)
async def compare_models(
    ids: str = Query(..., description="Comma-separated model_ids"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Side-by-side comparison of 2-4 models."""
    model_ids = [mid.strip() for mid in ids.split(",") if mid.strip()]
    if len(model_ids) < 2 or len(model_ids) > 4:
        raise HTTPException(400, "Provide 2-4 model IDs for comparison")

    models = db.query(LLMModel).filter(LLMModel.model_id.in_(model_ids)).all()
    if not models:
        raise HTTPException(404, "No models found")

    installed = _get_installed_ids(db, ctx.workspace_id)
    return CompareOut(models=[_model_to_out(m, installed) for m in models])


@router.get("/categories", response_model=List[Dict[str, Any]])
async def list_categories(db: Session = Depends(get_db)):
    """List available model categories with counts."""
    rows = (
        db.query(LLMModel.category, func.count(LLMModel.id))
        .filter(LLMModel.status == "active", LLMModel.category.isnot(None))
        .group_by(LLMModel.category)
        .all()
    )
    return [{"category": cat, "count": count} for cat, count in rows]


@router.get("/providers", response_model=List[Dict[str, Any]])
async def list_providers(db: Session = Depends(get_db)):
    """List available providers with counts."""
    rows = (
        db.query(LLMModel.provider, func.count(LLMModel.id))
        .filter(LLMModel.status == "active")
        .group_by(LLMModel.provider)
        .all()
    )
    return [{"provider": prov, "count": count} for prov, count in rows]
