"""
LLM Marketplace API (PRD-54, PRD-236 W1)
========================================

Browse, compare, and install LLM ROUTES to workspaces. A route is one row of
``llm_models`` keyed by ``(serving_provider, model_id)``: the same vendor id
("moonshotai/kimi-k3") is a different route on OpenRouter (paid) and on NVIDIA
(free, trial terms), each with its own price. Installing a route creates the
``workspace_models`` row for THAT row — the install is the tag the factory
routes to.

Catalogue rows are written by ``core.services.provider_catalog_sync``
(OpenRouter via its cache, NVIDIA from its public list); direct providers keep
their seeded rows.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import LLMModel, WorkspaceModel
from core.models.openrouter_cache import OpenRouterModelCache
from core.llm import providers as registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/marketplace/llm", tags=["LLM Marketplace"])

# Pricing tier thresholds, $ per 1K input tokens — the same bands the
# OpenRouter cache uses per token (free / ≤$0.50 per M / ≤$3 per M / above).
_PRICE_TIER_BUDGET_PER_1K = 0.0005
_PRICE_TIER_MID_PER_1K = 0.003


# ── Pydantic schemas ─────────────────────────────────────────────────

class LLMModelOut(BaseModel):
    id: int
    provider: str                      # the VENDOR ('openai', 'moonshotai', …)
    vendor: str                        # alias of provider, explicit for the UI
    serving_provider: str              # the ROUTE ('openrouter', 'nvidia', 'openai', …)
    serving_provider_label: str
    route_label: str                   # "Kimi K3 · NVIDIA"
    model_id: str
    display_name: str
    model_family: Optional[str]
    description: Optional[str]
    context_window: int
    max_output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    is_free: bool = False
    price_tier: str = "premium"        # free | budget | mid | premium (from price)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    recommended_for: List[str] = Field(default_factory=list)
    supports_functions: bool
    supports_vision: bool
    supports_streaming: bool
    status: str
    sourcing: Optional[str]            # direct | aggregator | hosted_open (PRD-223 Q1)
    category: Optional[str]
    tags: List[str] = Field(default_factory=list)
    is_featured: bool = False
    is_default: bool = False
    requires_plan: Optional[str]
    install_count: int = 0
    is_installed: bool = False         # THIS route, in this workspace
    key_available: bool = True         # the workspace can call this route today
    terms_note: Optional[str] = None
    rate_limit_note: Optional[str] = None

    class Config:
        from_attributes = True


class CatalogOut(BaseModel):
    models: List[LLMModelOut]
    total: int
    providers: Dict[str, int]          # serving provider → count (unfiltered)
    provider_labels: Dict[str, str]
    vendors: Dict[str, int]            # vendor → count (within the provider filter)
    last_synced: Dict[str, Optional[str]]
    syncable: List[str]


class CompareOut(BaseModel):
    models: List[LLMModelOut]


class InstallResult(BaseModel):
    success: bool
    message: str
    model_id: str
    provider: str


# ── Helpers ───────────────────────────────────────────────────────────

def price_tier_for(input_cost_per_1k: float) -> str:
    cost = float(input_cost_per_1k or 0)
    if cost <= 0:
        return "free"
    if cost <= _PRICE_TIER_BUDGET_PER_1K:
        return "budget"
    if cost <= _PRICE_TIER_MID_PER_1K:
        return "mid"
    return "premium"


def route_key(serving_provider: str, model_id: str) -> str:
    return f"{serving_provider}:{model_id}"


def _model_to_out(
    m: LLMModel,
    installed_row_ids: set = None,
    available_providers: set = None,
) -> LLMModelOut:
    spec = registry.get_spec(m.serving_provider)
    label = spec.label if spec else (m.serving_provider or "").title()
    in_cost = float(m.input_cost_per_1k_tokens or 0)
    out_cost = float(m.output_cost_per_1k_tokens or 0)
    is_free = in_cost <= 0 and out_cost <= 0 and bool(spec and spec.free) or (in_cost <= 0 and out_cost <= 0)
    return LLMModelOut(
        id=m.id,
        provider=m.provider,
        vendor=m.provider,
        serving_provider=m.serving_provider,
        serving_provider_label=label,
        route_label=f"{m.display_name} · {label}",
        model_id=m.model_id,
        display_name=m.display_name,
        model_family=m.model_family,
        description=m.description,
        context_window=m.context_window or 0,
        max_output_tokens=m.max_output_tokens or 0,
        input_cost_per_1k=in_cost,
        output_cost_per_1k=out_cost,
        is_free=is_free,
        price_tier=price_tier_for(in_cost),
        capabilities=m.capabilities or {},
        recommended_for=m.recommended_for or [],
        supports_functions=m.supports_functions or False,
        supports_vision=m.supports_vision or False,
        supports_streaming=m.supports_streaming if m.supports_streaming is not None else True,
        status=m.status or "active",
        sourcing=m.sourcing,
        category=m.category,
        tags=m.tags or [],
        is_featured=m.is_featured or False,
        is_default=m.is_default or False,
        requires_plan=m.requires_plan,
        install_count=m.install_count or 0,
        is_installed=m.id in (installed_row_ids or set()),
        key_available=(available_providers is None) or (m.serving_provider in available_providers),
        terms_note=spec.terms_note if spec else None,
        rate_limit_note=spec.rate_limit_note if spec else None,
    )


def _find_route(db: Session, model_id: str, provider: Optional[str] = None) -> Optional[LLMModel]:
    """The catalogue row for a model id — the caller's route, else OpenRouter's, else any."""
    q = db.query(LLMModel).filter(LLMModel.model_id == model_id)
    route = registry.normalize_slug(provider) if provider else None
    if route:
        return q.filter(LLMModel.serving_provider == route).first()
    return q.filter(LLMModel.serving_provider == "openrouter").first() or q.first()


def _get_or_create_from_cache(db: Session, model_id: str, provider: Optional[str] = None) -> Optional[LLMModel]:
    """Find the route row; for OpenRouter, auto-create it from the cache.

    Kept for the write paths that validate a model id (orchestrator settings,
    agent config, the chat tools). The catalogue sync normally populates
    OpenRouter rows ahead of time; this covers a cache row that was never
    projected. Other providers are only served from synced rows.
    """
    m = _find_route(db, model_id, provider)
    if m:
        return m
    route = registry.normalize_slug(provider) if provider else None
    if route not in (None, "openrouter"):
        return None

    cached = db.query(OpenRouterModelCache).filter(
        OpenRouterModelCache.model_id == model_id
    ).first()
    if not cached:
        return None

    m = LLMModel(
        provider=cached.provider,
        serving_provider="openrouter",
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
        sourcing="aggregator",
        category=cached.category,
        tags=cached.tags or [],
        capabilities={},
        recommended_for=[],
        external_id=cached.model_id,
    )
    db.add(m)
    db.flush()
    logger.info(f"Auto-created OpenRouter route from cache: {model_id}")
    return m


def _get_available_providers(db: Session, workspace_id) -> set:
    """Serving providers the workspace can call today (a key resolves for them).

    Checks: 1) BYOK keys (UserApiKey), 2) the credential store (platform keys),
    3) the operator's env keys where a platform key is allowed in this edition
    (the local edition's .env). Never the key values — presence only.
    """
    available: set = set()

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

    try:
        from core.credentials.resolver import get_credential_resolver
        resolver = get_credential_resolver()
        for provider in registry.platform_key_slugs():
            if provider in available:
                continue
            for variation in [f"{provider}_api", provider]:
                try:
                    key = resolver.get_credential_field(variation, "api_key")
                    if key:
                        available.add(provider)
                        break
                except Exception:
                    continue
    except Exception:
        pass

    for provider in registry.platform_key_slugs():
        if provider not in available and registry.env_api_key(provider):
            available.add(provider)

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


def _apply_filters(query, *, provider, vendor, category, price_tier, min_context,
                   max_cost, capability, supports_tools, supports_vision, search):
    if provider:
        query = query.filter(LLMModel.serving_provider == registry.normalize_slug(provider))
    if vendor:
        query = query.filter(LLMModel.provider == vendor)
    if category:
        query = query.filter(LLMModel.category == category)
    if price_tier == "free":
        query = query.filter(LLMModel.input_cost_per_1k_tokens <= 0)
    elif price_tier == "budget":
        query = query.filter(LLMModel.input_cost_per_1k_tokens > 0,
                             LLMModel.input_cost_per_1k_tokens <= _PRICE_TIER_BUDGET_PER_1K)
    elif price_tier == "mid":
        query = query.filter(LLMModel.input_cost_per_1k_tokens > _PRICE_TIER_BUDGET_PER_1K,
                             LLMModel.input_cost_per_1k_tokens <= _PRICE_TIER_MID_PER_1K)
    elif price_tier == "premium":
        query = query.filter(LLMModel.input_cost_per_1k_tokens > _PRICE_TIER_MID_PER_1K)
    if min_context:
        query = query.filter(LLMModel.context_window >= min_context)
    if max_cost is not None:
        query = query.filter(LLMModel.input_cost_per_1k_tokens <= max_cost)
    if capability:
        query = query.filter(LLMModel.capabilities[capability].isnot(None))
    if supports_tools:
        query = query.filter(LLMModel.supports_functions == True)
    if supports_vision:
        query = query.filter(LLMModel.supports_vision == True)
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            or_(
                LLMModel.display_name.ilike(pattern),
                LLMModel.model_id.ilike(pattern),
                LLMModel.description.ilike(pattern),
            )
        )
    return query


def _apply_sort(query, sort_by: Optional[str]):
    if sort_by == "cost":
        return query.order_by(LLMModel.input_cost_per_1k_tokens.asc(), LLMModel.display_name.asc())
    if sort_by == "context":
        return query.order_by(LLMModel.context_window.desc())
    if sort_by == "name":
        return query.order_by(LLMModel.display_name.asc())
    if sort_by == "newest":
        return query.order_by(LLMModel.created_at.desc().nullslast())
    return query.order_by(LLMModel.install_count.desc(), LLMModel.is_featured.desc(), LLMModel.display_name.asc())


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/catalog", response_model=CatalogOut)
async def browse_catalog(
    provider: Optional[str] = Query(None, description="Serving provider slug (openrouter, nvidia, openai…)"),
    vendor: Optional[str] = Query(None, description="Vendor (moonshotai, deepseek, openai…)"),
    category: Optional[str] = Query(None),
    price_tier: Optional[str] = Query(None, description="free | budget | mid | premium"),
    min_context: Optional[int] = Query(None),
    max_cost: Optional[float] = Query(None, description="Max input cost per 1K tokens"),
    capability: Optional[str] = Query(None),
    supports_tools: Optional[bool] = Query(None),
    supports_vision: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: Optional[str] = Query("popularity", description="cost|context|popularity|name|newest"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """The per-route catalogue: one card per (serving provider, model)."""
    from core.services.provider_catalog_sync import ProviderCatalogSync, SYNCABLE_PROVIDERS

    base = db.query(LLMModel).filter(LLMModel.status == "active")
    query = _apply_filters(
        base, provider=provider, vendor=vendor, category=category, price_tier=price_tier,
        min_context=min_context, max_cost=max_cost, capability=capability,
        supports_tools=supports_tools, supports_vision=supports_vision, search=search,
    )
    total = query.count()
    models = _apply_sort(query, sort_by).offset(offset).limit(limit).all()

    provider_counts = dict(
        db.query(LLMModel.serving_provider, func.count(LLMModel.id))
        .filter(LLMModel.status == "active")
        .group_by(LLMModel.serving_provider)
        .all()
    )
    vendor_query = db.query(LLMModel.provider, func.count(LLMModel.id)).filter(LLMModel.status == "active")
    if provider:
        vendor_query = vendor_query.filter(LLMModel.serving_provider == registry.normalize_slug(provider))
    vendor_counts = dict(vendor_query.group_by(LLMModel.provider).all())

    installed = _get_installed_ids(db, ctx.workspace_id)
    available = _get_available_providers(db, ctx.workspace_id)
    labels = {
        slug: (registry.get_spec(slug).label if registry.get_spec(slug) else slug.title())
        for slug in provider_counts
    }
    return CatalogOut(
        models=[_model_to_out(m, installed, available) for m in models],
        total=total,
        providers=provider_counts,
        provider_labels=labels,
        vendors=vendor_counts,
        last_synced=ProviderCatalogSync(db).last_synced(),
        syncable=list(SYNCABLE_PROVIDERS),
    )


@router.get("/models", response_model=List[LLMModelOut])
async def browse_models(
    provider: Optional[str] = Query(None, description="Serving provider slug"),
    vendor: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    price_tier: Optional[str] = Query(None),
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
    """Browse routes as a flat list (the catalog endpoint carries the facets)."""
    query = _apply_filters(
        db.query(LLMModel).filter(LLMModel.status == "active"),
        provider=provider, vendor=vendor, category=category, price_tier=price_tier,
        min_context=min_context, max_cost=max_cost, capability=capability,
        supports_tools=None, supports_vision=None, search=search,
    )
    models = _apply_sort(query, sort_by).offset(offset).limit(limit).all()
    installed = _get_installed_ids(db, ctx.workspace_id)
    available = _get_available_providers(db, ctx.workspace_id)
    return [_model_to_out(m, installed, available) for m in models]


@router.get("/installed", response_model=List[LLMModelOut])
async def get_installed_models(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Every route the workspace has installed — what the pickers list.

    Provider availability is reported (``key_available``), not enforced: the
    factory's key resolution is the gate at execution time.
    """
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    installed_ids = _get_installed_ids(db, ctx.workspace_id)
    if not installed_ids:
        return []

    models = (
        db.query(LLMModel)
        .filter(LLMModel.status == "active", LLMModel.id.in_(installed_ids))
        .order_by(LLMModel.serving_provider, LLMModel.display_name)
        .all()
    )
    available = _get_available_providers(db, ctx.workspace_id)
    return [_model_to_out(m, installed_ids, available) for m in models]


@router.get("/installed-ids")
async def get_installed_ids(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Installed routes, lightweight: ``routes`` are "provider:model_id" keys."""
    if not ctx.workspace_id:
        return {"model_ids": [], "routes": []}
    installed_llm_ids = _get_installed_ids(db, ctx.workspace_id)
    if not installed_llm_ids:
        return {"model_ids": [], "routes": []}
    rows = (
        db.query(LLMModel.serving_provider, LLMModel.model_id)
        .filter(LLMModel.id.in_(installed_llm_ids))
        .all()
    )
    return {
        "model_ids": sorted({r[1] for r in rows}),
        "routes": sorted(route_key(r[0], r[1]) for r in rows),
    }


@router.get("/models/{model_id:path}", response_model=LLMModelOut)
async def get_model_detail(
    model_id: str,
    provider: Optional[str] = Query(None, description="Serving provider slug"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Full detail for one route."""
    m = _get_or_create_from_cache(db, model_id, provider)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}")
    installed = _get_installed_ids(db, ctx.workspace_id)
    return _model_to_out(m, installed, _get_available_providers(db, ctx.workspace_id))


@router.post("/models/{model_id:path}/install", response_model=InstallResult, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def install_model(
    model_id: str,
    provider: Optional[str] = Query(None, description="Serving provider slug — the route to install"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Install a ROUTE to the current workspace (the install is the tag)."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    m = _get_or_create_from_cache(db, model_id, provider)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}" + (f" on {provider}" if provider else ""))

    existing = (
        db.query(WorkspaceModel)
        .filter(WorkspaceModel.workspace_id == ctx.workspace_id, WorkspaceModel.model_id == m.id)
        .first()
    )
    if existing:
        if not existing.is_active:
            existing.is_active = True
            db.commit()
            return InstallResult(success=True, message="Model re-activated", model_id=model_id, provider=m.serving_provider)
        return InstallResult(success=True, message="Model already installed", model_id=model_id, provider=m.serving_provider)

    wm = WorkspaceModel(
        workspace_id=ctx.workspace_id,
        model_id=m.id,
        source="marketplace",
    )
    db.add(wm)
    m.install_count = (m.install_count or 0) + 1
    db.commit()

    logger.info(f"Route {m.serving_provider}:{model_id} installed to workspace {ctx.workspace_id}")
    return InstallResult(success=True, message="Model installed", model_id=model_id, provider=m.serving_provider)


@router.post("/models/{model_id:path}/uninstall", response_model=InstallResult, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def uninstall_model(
    model_id: str,
    provider: Optional[str] = Query(None, description="Serving provider slug — the route to remove"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a route from the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    m = _find_route(db, model_id, provider)
    if not m:
        raise HTTPException(404, f"Model not found: {model_id}")

    wm = (
        db.query(WorkspaceModel)
        .filter(WorkspaceModel.workspace_id == ctx.workspace_id, WorkspaceModel.model_id == m.id)
        .first()
    )
    if not wm:
        return InstallResult(success=True, message="Model was not installed", model_id=model_id, provider=m.serving_provider)

    wm.is_active = False
    db.commit()
    return InstallResult(success=True, message="Model uninstalled", model_id=model_id, provider=m.serving_provider)


@router.get("/compare", response_model=CompareOut)
async def compare_models(
    ids: str = Query(..., description="Comma-separated model_ids or provider:model_id route keys"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Side-by-side comparison of 2-4 routes. A bare model id compares every route for it."""
    keys = [k.strip() for k in ids.split(",") if k.strip()]
    if len(keys) < 2 or len(keys) > 4:
        raise HTTPException(400, "Provide 2-4 model IDs for comparison")

    models: List[LLMModel] = []
    for key in keys:
        if ":" in key and "/" not in key.split(":", 1)[0]:
            prov, mid = key.split(":", 1)
            m = _find_route(db, mid, prov)
            if m:
                models.append(m)
        else:
            models.extend(db.query(LLMModel).filter(LLMModel.model_id == key).all())
    if not models:
        raise HTTPException(404, "No models found")

    installed = _get_installed_ids(db, ctx.workspace_id)
    available = _get_available_providers(db, ctx.workspace_id)
    return CompareOut(models=[_model_to_out(m, installed, available) for m in models])


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
    """Serving providers with route counts."""
    rows = (
        db.query(LLMModel.serving_provider, func.count(LLMModel.id))
        .filter(LLMModel.status == "active")
        .group_by(LLMModel.serving_provider)
        .all()
    )
    out = []
    for slug, count in rows:
        spec = registry.get_spec(slug)
        out.append({"provider": slug, "label": spec.label if spec else slug, "count": count})
    return out


@router.get("/sync/status")
async def sync_status(db: Session = Depends(get_db)):
    """When each syncable provider's catalogue was last refreshed."""
    from core.services.provider_catalog_sync import ProviderCatalogSync, SYNCABLE_PROVIDERS
    return {"last_synced": ProviderCatalogSync(db).last_synced(), "syncable": list(SYNCABLE_PROVIDERS)}


@router.post("/sync/{provider}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def sync_provider(
    provider: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Refresh one provider's catalogue (admin). OpenRouter: cache sync + projection;
    NVIDIA: the public model list, metadata borrowed from OpenRouter's rows."""
    from core.services.provider_catalog_sync import ProviderCatalogSync, SYNCABLE_PROVIDERS

    slug = registry.normalize_slug(provider)
    if slug not in SYNCABLE_PROVIDERS:
        raise HTTPException(400, f"'{provider}' has no catalogue sync. Syncable: {', '.join(SYNCABLE_PROVIDERS)}")
    logger.info(f"Catalogue sync for {slug} requested by workspace {ctx.workspace_id}")
    try:
        result = ProviderCatalogSync(db).sync(slug)
    except Exception as exc:
        raise HTTPException(502, f"{slug} catalogue sync failed: {str(exc)[:300]}")
    logger.info(f"Catalogue sync for {slug} completed: {result}")
    return result
