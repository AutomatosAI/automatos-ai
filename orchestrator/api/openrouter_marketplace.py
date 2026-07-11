"""
OpenRouter Model Marketplace API

Serves the cached OpenRouter model catalog with rich filters.
Sync is completely independent from the Composio tools sync.

Endpoints:
    GET  /api/openrouter/models       - Browse cached models with filters
    GET  /api/openrouter/providers     - List providers with counts
    GET  /api/openrouter/stats         - Cache statistics
    POST /api/openrouter/sync          - Trigger sync from OpenRouter API
    GET  /api/openrouter/sync/history  - Sync job history
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob
from core.services.openrouter_sync_service import OpenRouterSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/openrouter", tags=["OpenRouter Marketplace"])


# ── Pydantic schemas ──────────────────────────────────────────────────

class OpenRouterModelOut(BaseModel):
    id: int
    model_id: str
    slug: Optional[str]
    display_name: str
    description: Optional[str]
    provider: str

    # Pricing (per token)
    prompt_cost: float = 0
    completion_cost: float = 0
    image_cost: float = 0
    request_cost: float = 0
    cache_read_cost: float = 0
    cache_write_cost: float = 0

    # Sizing
    context_length: int = 0
    max_completion_tokens: int = 0

    # Architecture
    modality: Optional[str]
    input_modalities: List[str] = Field(default_factory=list)
    output_modalities: List[str] = Field(default_factory=list)
    tokenizer: Optional[str]

    # Capabilities
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_reasoning: bool = False

    # Categorization
    category: Optional[str]
    tier: Optional[str]
    tags: List[str] = Field(default_factory=list)
    status: str = "active"

    # Provider info
    is_moderated: bool = False

    last_synced_at: Optional[str]

    class Config:
        from_attributes = True


class OpenRouterMarketplaceOut(BaseModel):
    models: List[OpenRouterModelOut]
    total: int
    providers: Dict[str, int]
    last_synced: Optional[str]


class OpenRouterStatsOut(BaseModel):
    total_models: int
    providers: Dict[str, int]
    categories: Dict[str, int]
    tiers: Dict[str, int]
    last_synced: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────

def _model_to_out(m: OpenRouterModelCache) -> OpenRouterModelOut:
    return OpenRouterModelOut(
        id=m.id,
        model_id=m.model_id,
        slug=m.slug,
        display_name=m.display_name,
        description=m.description,
        provider=m.provider,
        prompt_cost=m.prompt_cost or 0,
        completion_cost=m.completion_cost or 0,
        image_cost=m.image_cost or 0,
        request_cost=m.request_cost or 0,
        cache_read_cost=m.cache_read_cost or 0,
        cache_write_cost=m.cache_write_cost or 0,
        context_length=m.context_length or 0,
        max_completion_tokens=m.max_completion_tokens or 0,
        modality=m.modality,
        input_modalities=m.input_modalities or [],
        output_modalities=m.output_modalities or [],
        tokenizer=m.tokenizer,
        supports_tools=m.supports_tools or False,
        supports_vision=m.supports_vision or False,
        supports_streaming=m.supports_streaming if m.supports_streaming is not None else True,
        supports_json_mode=m.supports_json_mode or False,
        supports_reasoning=m.supports_reasoning or False,
        category=m.category,
        tier=m.tier,
        tags=m.tags or [],
        status=m.status or "active",
        is_moderated=m.is_moderated or False,
        last_synced_at=m.last_synced_at.isoformat() if m.last_synced_at else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/models", response_model=OpenRouterMarketplaceOut)
async def browse_models(
    provider: Optional[str] = Query(None, description="Filter by provider slug"),
    category: Optional[str] = Query(None, description="fast, balanced, premium, coding, reasoning, vision, free"),
    tier: Optional[str] = Query(None, description="free, budget, mid, premium"),
    modality: Optional[str] = Query(None, description="Filter by modality substring"),
    supports_tools: Optional[bool] = Query(None),
    supports_vision: Optional[bool] = Query(None),
    supports_reasoning: Optional[bool] = Query(None),
    min_context: Optional[int] = Query(None, description="Minimum context window"),
    max_prompt_cost: Optional[float] = Query(None, description="Max prompt cost per token"),
    search: Optional[str] = Query(None, description="Search name, model_id, or description"),
    sort_by: Optional[str] = Query("popularity", description="cost|context|name|newest"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Browse OpenRouter models from local cache with rich filters."""

    q = db.query(OpenRouterModelCache).filter(OpenRouterModelCache.status == "active")

    # Filters
    if provider:
        q = q.filter(OpenRouterModelCache.provider == provider)
    if category:
        q = q.filter(OpenRouterModelCache.category == category)
    if tier:
        q = q.filter(OpenRouterModelCache.tier == tier)
    if modality:
        q = q.filter(OpenRouterModelCache.modality.ilike(f"%{modality}%"))
    if supports_tools is not None:
        q = q.filter(OpenRouterModelCache.supports_tools == supports_tools)
    if supports_vision is not None:
        q = q.filter(OpenRouterModelCache.supports_vision == supports_vision)
    if supports_reasoning is not None:
        q = q.filter(OpenRouterModelCache.supports_reasoning == supports_reasoning)
    if min_context:
        q = q.filter(OpenRouterModelCache.context_length >= min_context)
    if max_prompt_cost is not None:
        q = q.filter(OpenRouterModelCache.prompt_cost <= max_prompt_cost)
    if search:
        pattern = f"%{search}%"
        q = q.filter(or_(
            OpenRouterModelCache.display_name.ilike(pattern),
            OpenRouterModelCache.model_id.ilike(pattern),
            OpenRouterModelCache.description.ilike(pattern),
        ))

    total = q.count()

    # Sort
    if sort_by == "cost":
        q = q.order_by(OpenRouterModelCache.prompt_cost.asc())
    elif sort_by == "context":
        q = q.order_by(OpenRouterModelCache.context_length.desc())
    elif sort_by == "name":
        q = q.order_by(OpenRouterModelCache.display_name.asc())
    elif sort_by == "newest":
        q = q.order_by(OpenRouterModelCache.created_timestamp.desc().nullslast())
    else:  # popularity / default
        q = q.order_by(OpenRouterModelCache.context_length.desc())

    models = q.offset(offset).limit(limit).all()

    # Provider counts for filter chips
    provider_counts = dict(
        db.query(OpenRouterModelCache.provider, func.count(OpenRouterModelCache.id))
        .filter(OpenRouterModelCache.status == "active")
        .group_by(OpenRouterModelCache.provider)
        .all()
    )

    # Last sync
    last_job = (
        db.query(OpenRouterSyncJob)
        .filter_by(status="completed")
        .order_by(OpenRouterSyncJob.completed_at.desc())
        .first()
    )
    last_synced = last_job.completed_at.isoformat() if last_job else None

    return OpenRouterMarketplaceOut(
        models=[_model_to_out(m) for m in models],
        total=total,
        providers=provider_counts,
        last_synced=last_synced,
    )


@router.get("/providers")
async def list_providers(
    db: Session = Depends(get_db),
):
    """List all providers with model counts."""
    rows = (
        db.query(OpenRouterModelCache.provider, func.count(OpenRouterModelCache.id))
        .filter(OpenRouterModelCache.status == "active")
        .group_by(OpenRouterModelCache.provider)
        .order_by(func.count(OpenRouterModelCache.id).desc())
        .all()
    )
    return [{"provider": p, "count": c} for p, c in rows]


@router.get("/stats", response_model=OpenRouterStatsOut)
async def stats(
    db: Session = Depends(get_db),
):
    """Get cache statistics."""
    total = db.query(func.count(OpenRouterModelCache.id)).filter(
        OpenRouterModelCache.status == "active"
    ).scalar() or 0

    provider_rows = (
        db.query(OpenRouterModelCache.provider, func.count(OpenRouterModelCache.id))
        .filter(OpenRouterModelCache.status == "active")
        .group_by(OpenRouterModelCache.provider)
        .all()
    )

    category_rows = (
        db.query(OpenRouterModelCache.category, func.count(OpenRouterModelCache.id))
        .filter(OpenRouterModelCache.status == "active", OpenRouterModelCache.category.isnot(None))
        .group_by(OpenRouterModelCache.category)
        .all()
    )

    tier_rows = (
        db.query(OpenRouterModelCache.tier, func.count(OpenRouterModelCache.id))
        .filter(OpenRouterModelCache.status == "active", OpenRouterModelCache.tier.isnot(None))
        .group_by(OpenRouterModelCache.tier)
        .all()
    )

    last_job = (
        db.query(OpenRouterSyncJob)
        .filter_by(status="completed")
        .order_by(OpenRouterSyncJob.completed_at.desc())
        .first()
    )

    return OpenRouterStatsOut(
        total_models=total,
        providers={p: c for p, c in provider_rows},
        categories={c: n for c, n in category_rows},
        tiers={t: n for t, n in tier_rows},
        last_synced=last_job.completed_at.isoformat() if last_job else None,
    )


@router.post("/sync", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def sync(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Trigger a full sync of OpenRouter models to local cache.

    This is an admin action (separate from Composio tools sync).
    """
    logger.info(f"OpenRouter sync requested by workspace {ctx.workspace_id}")
    service = OpenRouterSyncService(db)
    result = service.run_full_sync()
    logger.info(f"OpenRouter sync completed: {result}")
    return result


@router.get("/sync/history")
async def sync_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get sync job history."""
    service = OpenRouterSyncService(db)
    return {"jobs": service.get_sync_history(limit), "count": limit}
