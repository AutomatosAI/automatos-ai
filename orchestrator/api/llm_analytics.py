"""
LLM Analytics & Usage Tracking API (PRD-54)
=============================================

Usage tracking, cost analytics, optimization recommendations,
and OpenRouter integration (credits, key info, activity sync).
"""

import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import LLMUsage, LLMModel, UserApiKey
from core.credentials.encryption import get_encryption_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/llm", tags=["LLM Analytics"])


# ── Pydantic schemas ─────────────────────────────────────────────────

class UsageGroup(BaseModel):
    key: str
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


class CostBreakdown(BaseModel):
    key: str
    input_cost: float
    output_cost: float
    total_cost: float
    request_count: int


class UsageSummary(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    avg_latency_ms: Optional[float]
    error_rate: float
    top_models: List[Dict[str, Any]]
    cost_trend: List[Dict[str, Any]]


class Recommendation(BaseModel):
    type: str  # cost_optimization, model_switch, quota_warning
    title: str
    description: str
    potential_savings: Optional[float] = None
    affected_agent_ids: List[int] = Field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────

PERIOD_MAP = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}


def _period_start(period: str) -> datetime:
    delta = PERIOD_MAP.get(period, timedelta(days=7))
    return datetime.utcnow() - delta


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/usage", response_model=List[UsageGroup])
async def get_usage(
    period: str = Query("7d", description="1h|24h|7d|30d"),
    group_by: str = Query("model", description="model|provider|agent|tier"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Token usage grouped by dimension."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)

    group_col_map = {
        "model": LLMUsage.model_id,
        "provider": LLMUsage.provider,
        "agent": LLMUsage.agent_id,
        "tier": LLMUsage.tier,
    }
    group_col = group_col_map.get(group_by, LLMUsage.model_id)

    rows = (
        db.query(
            group_col.label("key"),
            func.count(LLMUsage.id).label("request_count"),
            func.sum(LLMUsage.input_tokens).label("input_tokens"),
            func.sum(LLMUsage.output_tokens).label("output_tokens"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
        )
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(group_col)
        .order_by(desc("total_cost"))
        .all()
    )

    return [
        UsageGroup(
            key=str(r.key or "unknown"),
            request_count=r.request_count,
            input_tokens=int(r.input_tokens or 0),
            output_tokens=int(r.output_tokens or 0),
            total_tokens=int(r.total_tokens or 0),
            total_cost=float(r.total_cost or 0),
        )
        for r in rows
    ]


@router.get("/costs", response_model=List[CostBreakdown])
async def get_costs(
    period: str = Query("7d"),
    breakdown: str = Query("model", description="model|provider|agent|daily"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Cost breakdown by dimension."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)

    if breakdown == "daily":
        group_col = func.date(LLMUsage.created_at)
    else:
        col_map = {
            "model": LLMUsage.model_id,
            "provider": LLMUsage.provider,
            "agent": LLMUsage.agent_id,
        }
        group_col = col_map.get(breakdown, LLMUsage.model_id)

    rows = (
        db.query(
            group_col.label("key"),
            func.sum(LLMUsage.input_cost).label("input_cost"),
            func.sum(LLMUsage.output_cost).label("output_cost"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.count(LLMUsage.id).label("request_count"),
        )
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(group_col)
        .order_by(desc("total_cost"))
        .all()
    )

    return [
        CostBreakdown(
            key=str(r.key or "unknown"),
            input_cost=float(r.input_cost or 0),
            output_cost=float(r.output_cost or 0),
            total_cost=float(r.total_cost or 0),
            request_count=r.request_count,
        )
        for r in rows
    ]


@router.get("/summary", response_model=UsageSummary)
async def get_summary(
    period: str = Query("7d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Dashboard summary: totals, top models, cost trend."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)
    base = db.query(LLMUsage).filter(
        LLMUsage.workspace_id == ctx.workspace_id,
        LLMUsage.created_at >= since,
    )

    # Aggregates
    agg = base.with_entities(
        func.count(LLMUsage.id).label("cnt"),
        func.sum(LLMUsage.total_tokens).label("tokens"),
        func.sum(LLMUsage.total_cost).label("cost"),
        func.avg(LLMUsage.latency_ms).label("latency"),
    ).first()

    total_requests = agg.cnt or 0
    error_count = base.filter(LLMUsage.status == "error").count()
    error_rate = error_count / total_requests if total_requests > 0 else 0.0

    # Top models
    top = (
        base.with_entities(
            LLMUsage.model_id,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.count(LLMUsage.id).label("cnt"),
        )
        .group_by(LLMUsage.model_id)
        .order_by(desc("cost"))
        .limit(5)
        .all()
    )

    # Daily cost trend
    trend = (
        base.with_entities(
            func.date(LLMUsage.created_at).label("day"),
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .group_by(func.date(LLMUsage.created_at))
        .order_by("day")
        .all()
    )

    return UsageSummary(
        total_requests=total_requests,
        total_tokens=int(agg.tokens or 0),
        total_cost=float(agg.cost or 0),
        avg_latency_ms=float(agg.latency) if agg.latency else None,
        error_rate=round(error_rate, 4),
        top_models=[
            {"model_id": m.model_id, "total_cost": float(m.cost or 0), "request_count": m.cnt}
            for m in top
        ],
        cost_trend=[
            {"date": str(t.day), "cost": float(t.cost or 0)}
            for t in trend
        ],
    )


@router.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """AI-generated cost optimization suggestions based on usage patterns."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start("30d")
    recs: List[Recommendation] = []

    # Find agents using expensive models for simple tasks
    agent_usage = (
        db.query(
            LLMUsage.agent_id,
            LLMUsage.model_id,
            func.count(LLMUsage.id).label("cnt"),
            func.sum(LLMUsage.total_cost).label("cost"),
            func.avg(LLMUsage.output_tokens).label("avg_output"),
        )
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
            LLMUsage.agent_id.isnot(None),
        )
        .group_by(LLMUsage.agent_id, LLMUsage.model_id)
        .having(func.count(LLMUsage.id) >= 10)
        .all()
    )

    premium_models = {"gpt-4o", "gpt-4-turbo", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"}

    for row in agent_usage:
        if row.model_id in premium_models and (row.avg_output or 0) < 200:
            potential = float(row.cost or 0) * 0.85  # ~85% savings switching to mini
            recs.append(Recommendation(
                type="cost_optimization",
                title=f"Switch Agent {row.agent_id} to a cheaper model",
                description=(
                    f"Agent {row.agent_id} used {row.model_id} for {row.cnt} requests "
                    f"with avg {int(row.avg_output or 0)} output tokens. "
                    f"Consider gpt-4o-mini or claude-haiku-4-5 for simple outputs."
                ),
                potential_savings=round(potential, 2),
                affected_agent_ids=[row.agent_id] if row.agent_id else [],
            ))

    if not recs:
        recs.append(Recommendation(
            type="info",
            title="No optimization suggestions",
            description="Your model usage looks well-optimized. Keep it up!",
        ))

    return recs


# ── Model Comparison ─────────────────────────────────────────────────


class ModelComparisonItem(BaseModel):
    model_id: str
    display_name: str
    provider: str
    input_cost_per_1k: Optional[float] = None
    output_cost_per_1k: Optional[float] = None
    context_window: Optional[int] = None
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: Optional[float] = None
    error_rate: float = 0.0
    success_rate: float = 1.0


@router.get("/comparison", response_model=List[ModelComparisonItem])
async def get_model_comparison(
    model_ids: str = Query(..., description="Comma-separated model IDs (max 4)"),
    period: str = Query("30d", description="7d|30d|90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Compare selected models side-by-side with cost, usage, and capability data."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if len(ids) > 4:
        raise HTTPException(400, "Maximum 4 models for comparison")
    if not ids:
        raise HTTPException(400, "At least one model_id required")

    since = _period_start(period)
    results: List[ModelComparisonItem] = []

    for mid in ids:
        # Get registry data (fallback info if no usage exists)
        registry = db.query(LLMModel).filter(LLMModel.model_id == mid).first()

        # Get usage stats scoped to workspace + period
        usage = (
            db.query(
                func.count(LLMUsage.id).label("total_requests"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
            )
            .filter(
                LLMUsage.workspace_id == ctx.workspace_id,
                LLMUsage.model_id == mid,
                LLMUsage.created_at >= since,
            )
            .first()
        )

        total_requests = usage.total_requests or 0 if usage else 0

        # Error rate
        error_count = 0
        if total_requests > 0:
            error_count = (
                db.query(func.count(LLMUsage.id))
                .filter(
                    LLMUsage.workspace_id == ctx.workspace_id,
                    LLMUsage.model_id == mid,
                    LLMUsage.created_at >= since,
                    LLMUsage.status == "error",
                )
                .scalar() or 0
            )

        error_rate = error_count / total_requests if total_requests > 0 else 0.0
        success_rate = 1.0 - error_rate

        results.append(ModelComparisonItem(
            model_id=mid,
            display_name=registry.display_name if registry else mid,
            provider=registry.provider if registry else (usage and getattr(usage, "provider", None)) or "unknown",
            input_cost_per_1k=registry.input_cost_per_1k_tokens if registry else None,
            output_cost_per_1k=registry.output_cost_per_1k_tokens if registry else None,
            context_window=registry.context_window if registry else None,
            capabilities=registry.capabilities or {} if registry else {},
            total_requests=total_requests,
            total_tokens=int(usage.total_tokens or 0) if usage else 0,
            total_cost=float(usage.total_cost or 0) if usage else 0.0,
            avg_latency_ms=float(usage.avg_latency_ms) if usage and usage.avg_latency_ms else None,
            error_rate=round(error_rate, 4),
            success_rate=round(success_rate, 4),
        ))

    return results


# ── Cost Projections ─────────────────────────────────────────────────


class ProjectedItem(BaseModel):
    key: str
    projected_monthly_cost: float
    current_period_cost: float


class CostProjectionResponse(BaseModel):
    current_period_cost: float
    daily_average: float
    projected_monthly: float
    change_percent: Optional[float] = None
    projected_by_model: List[ProjectedItem] = Field(default_factory=list)
    projected_by_provider: List[ProjectedItem] = Field(default_factory=list)


@router.get("/projections", response_model=CostProjectionResponse)
async def get_cost_projections(
    period: str = Query("30d", description="7d|30d|90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Projected monthly costs based on current usage trajectory."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    delta = PERIOD_MAP.get(period, timedelta(days=30))
    since = datetime.utcnow() - delta

    # Current period total cost
    current_cost = (
        db.query(func.sum(LLMUsage.total_cost))
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .scalar() or 0.0
    )
    current_cost = float(current_cost)

    # Count distinct days with data for accurate daily average
    days_with_data = (
        db.query(func.count(func.distinct(func.date(LLMUsage.created_at))))
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .scalar() or 0
    )

    daily_avg = current_cost / days_with_data if days_with_data > 0 else 0.0
    projected_monthly = daily_avg * 30

    # Previous period for comparison
    prev_start = since - delta
    prev_cost = (
        db.query(func.sum(LLMUsage.total_cost))
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= prev_start,
            LLMUsage.created_at < since,
        )
        .scalar() or 0.0
    )
    prev_cost = float(prev_cost)

    change_percent = None
    if prev_cost > 0:
        change_percent = round(((current_cost - prev_cost) / prev_cost) * 100, 2)

    # Projected by model
    by_model_rows = (
        db.query(
            LLMUsage.model_id.label("key"),
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(LLMUsage.model_id)
        .order_by(desc("cost"))
        .all()
    )

    projected_by_model = []
    for r in by_model_rows:
        model_cost = float(r.cost or 0)
        model_daily = model_cost / days_with_data if days_with_data > 0 else 0.0
        projected_by_model.append(ProjectedItem(
            key=str(r.key or "unknown"),
            projected_monthly_cost=round(model_daily * 30, 6),
            current_period_cost=round(model_cost, 6),
        ))

    # Projected by provider
    by_provider_rows = (
        db.query(
            LLMUsage.provider.label("key"),
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= since,
        )
        .group_by(LLMUsage.provider)
        .order_by(desc("cost"))
        .all()
    )

    projected_by_provider = []
    for r in by_provider_rows:
        prov_cost = float(r.cost or 0)
        prov_daily = prov_cost / days_with_data if days_with_data > 0 else 0.0
        projected_by_provider.append(ProjectedItem(
            key=str(r.key or "unknown"),
            projected_monthly_cost=round(prov_daily * 30, 6),
            current_period_cost=round(prov_cost, 6),
        ))

    return CostProjectionResponse(
        current_period_cost=round(current_cost, 6),
        daily_average=round(daily_avg, 6),
        projected_monthly=round(projected_monthly, 6),
        change_percent=change_percent,
        projected_by_model=projected_by_model,
        projected_by_provider=projected_by_provider,
    )


# ── OpenRouter helpers ────────────────────────────────────────────────

def _resolve_openrouter_key(workspace_id, db: Session) -> str:
    """
    Find the OpenRouter API key for a workspace.
    Priority: BYOK key in user_api_keys → env OPENROUTER_API_KEY.
    Raises HTTPException 404 if not found.
    """
    row = (
        db.query(UserApiKey)
        .filter(
            UserApiKey.workspace_id == workspace_id,
            UserApiKey.provider == "openrouter",
            UserApiKey.is_active == True,
        )
        .order_by(UserApiKey.created_at.desc())
        .first()
    )
    if row:
        try:
            return get_encryption_service().decrypt(row.encrypted_key)
        except Exception:
            logger.warning("Failed to decrypt OpenRouter key id=%s", row.id)

    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key

    raise HTTPException(
        404,
        "No OpenRouter API key configured. Add one in Settings → API Keys or set OPENROUTER_API_KEY.",
    )


# ── OpenRouter endpoints ─────────────────────────────────────────────


class OpenRouterSyncResponse(BaseModel):
    synced: int
    skipped: int
    error: Optional[str] = None


class OpenRouterCreditsResponse(BaseModel):
    total_credits: float
    total_usage: float


class OpenRouterKeyInfoResponse(BaseModel):
    limit: Optional[float] = None
    limit_remaining: Optional[float] = None
    usage_daily: float = 0
    usage_weekly: float = 0
    usage_monthly: float = 0
    is_free_tier: bool = False
    rate_limit: Dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/openrouter/sync",
    response_model=OpenRouterSyncResponse,
    summary="Trigger OpenRouter activity sync",
)
async def sync_openrouter_activity(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Sync OpenRouter activity data into llm_usage for the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    api_key = _resolve_openrouter_key(ctx.workspace_id, db)

    from core.llm.openrouter_analytics import OpenRouterAnalyticsService

    svc = OpenRouterAnalyticsService()
    result = await svc.sync_activity(api_key, ctx.workspace_id)
    return OpenRouterSyncResponse(**result)


@router.get(
    "/openrouter/credits",
    response_model=OpenRouterCreditsResponse,
    summary="Get OpenRouter credits balance",
)
async def get_openrouter_credits(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the OpenRouter account credits balance."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    api_key = _resolve_openrouter_key(ctx.workspace_id, db)

    from core.llm.openrouter_analytics import OpenRouterAnalyticsService

    svc = OpenRouterAnalyticsService()
    data = await svc.get_credits(api_key)
    if data is None:
        raise HTTPException(502, "Failed to fetch credits from OpenRouter")

    return OpenRouterCreditsResponse(**data)


@router.get(
    "/openrouter/key-info",
    response_model=OpenRouterKeyInfoResponse,
    summary="Get OpenRouter key usage stats",
)
async def get_openrouter_key_info(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return OpenRouter key limits and daily/weekly/monthly usage stats."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    api_key = _resolve_openrouter_key(ctx.workspace_id, db)

    from core.llm.openrouter_analytics import OpenRouterAnalyticsService

    svc = OpenRouterAnalyticsService()
    data = await svc.get_key_info(api_key)
    if data is None:
        raise HTTPException(502, "Failed to fetch key info from OpenRouter")

    return OpenRouterKeyInfoResponse(**data)
