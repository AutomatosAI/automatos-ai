"""
LLM Analytics & Usage Tracking API (PRD-54)
=============================================

Usage tracking, cost analytics, and optimization recommendations.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import LLMUsage, LLMModel

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
