"""
LLM Analytics & Usage Tracking API (PRD-54)
=============================================

Usage tracking, cost analytics, optimization recommendations,
and OpenRouter integration (credits, key info, activity sync).
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, case

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.super_admin import require_super_admin
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.core import LLMUsage, LLMModel, UserApiKey, Agent, RecipeExecution
from core.models import WorkflowTemplate as WorkflowRecipe
from core.models.workspaces import Workspace
from core.credentials.encryption import get_encryption_service
from core.llm.providers import describe_usage_provider
from config import config

logger = logging.getLogger(__name__)

# Rows the OpenRouter activity sync copies from OpenRouter's own daily report.
# They are a RECONCILIATION source (what OpenRouter says it charged), not
# calls — summing them with the per-call rows counted every dollar twice.
ACTIVITY_SYNC_REQUEST_TYPE = "activity_sync"
ROUTE_KEY_SEPARATOR = "@"


def route_key(model_id: Optional[str], provider: Optional[str]) -> str:
    """One key per ROUTE — the same vendor model served by two providers (Kimi
    K3 on NVIDIA for free, on OpenRouter for $3/M) is two lines, never one."""
    return f"{model_id or 'unknown'}{ROUTE_KEY_SEPARATOR}{provider or 'unknown'}"


def route_facts(model_id: Optional[str], provider: Optional[str]) -> Dict[str, Any]:
    facts = describe_usage_provider(provider)
    return {
        "key": route_key(model_id, provider),
        "model_id": model_id or "unknown",
        "provider": facts["slug"],
        "provider_label": facts["label"],
        "billing": facts["billing"],
        "label": f"{model_id or 'unknown'} · {facts['label']}",
    }


def _calls(db: Session, workspace_id, since: datetime):
    """Every per-call row of the workspace in the period — never the sync copies."""
    return db.query(LLMUsage).filter(
        LLMUsage.workspace_id == workspace_id,
        LLMUsage.created_at >= since,
        LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE,
    )


def _platform_calls(db: Session, since: datetime):
    return db.query(LLMUsage).filter(
        LLMUsage.created_at >= since,
        LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE,
    )

# PRD-143 S7 locked BOTH routers to super-admin; 2026-07-30 (Gerard) relaxes
# the workspace-scoped router to workspace owners/admins: every endpoint here
# already filters LLMUsage by ctx.workspace_id (audited — no cross-workspace
# reads), so an owner/admin sees exactly their own workspace's analytics.
# require_workspace_admin passes super_admin unconditionally. The cross-
# workspace aggregate surface below (admin_router) stays super-admin-only.
router = APIRouter(
    prefix="/api/analytics/llm",
    tags=["LLM Analytics"],
    dependencies=[Depends(require_workspace_admin)],
)
admin_router = APIRouter(
    prefix="/api/admin/analytics",
    tags=["Admin Analytics"],
    dependencies=[Depends(require_super_admin)],
)
# Mutating obs routes never relax (authz boundary sweep enforces exactly one
# auth bucket per route) — the OpenRouter sync POST lives on its own
# super-admin-only router at the SAME path prefix, outside the workspace-admin
# relax above.
sync_router = APIRouter(
    prefix="/api/analytics/llm",
    tags=["LLM Analytics"],
    dependencies=[Depends(require_super_admin)],
)


# ── Pydantic schemas ─────────────────────────────────────────────────

class UsageGroup(BaseModel):
    key: str
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    error_count: int = 0
    avg_latency_ms: Optional[float] = None
    # Route facts (group_by=route / model / provider): who served it, what it costs
    model_id: Optional[str] = None
    provider: Optional[str] = None
    provider_label: Optional[str] = None
    billing: Optional[str] = None
    label: Optional[str] = None


class CostBreakdown(BaseModel):
    key: str
    input_cost: float
    output_cost: float
    total_cost: float
    request_count: int


class ProviderUsage(BaseModel):
    provider: str
    label: str
    billing: str
    kind: str
    request_count: int
    total_tokens: int
    total_cost: float
    error_count: int = 0


class UsageSummary(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    avg_latency_ms: Optional[float]
    error_rate: float
    top_models: List[Dict[str, Any]]
    cost_trend: List[Dict[str, Any]]
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    by_provider: List[ProviderUsage] = Field(default_factory=list)


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
    period: str = Query("7d", description="1h|24h|7d|30d|90d"),
    group_by: str = Query("model", description="model|route|provider|agent|tier|is_byok|request_type|execution"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Token usage grouped by dimension.

    ``route`` (model × serving provider) is the honest unit of cost: a free
    NVIDIA route and a paid OpenRouter route for the same vendor model are two
    rows. ``execution`` groups by the thing that spent (``mission:<id>``,
    ``board_task:<id>``, ``chat:<id>``) so a mission's cost can be joined back.
    """
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)

    group_col_map = {
        "model": (LLMUsage.model_id,),
        "route": (LLMUsage.model_id, LLMUsage.provider),
        "provider": (LLMUsage.provider,),
        "agent": (LLMUsage.agent_id,),
        "tier": (LLMUsage.tier,),
        "is_byok": (LLMUsage.is_byok,),
        "request_type": (LLMUsage.request_type,),
        "execution": (LLMUsage.execution_id,),
    }
    group_cols = group_col_map.get(group_by, (LLMUsage.model_id,))

    rows = (
        _calls(db, ctx.workspace_id, since)
        .with_entities(
            *group_cols,
            func.count(LLMUsage.id).label("request_count"),
            func.sum(LLMUsage.input_tokens).label("input_tokens"),
            func.sum(LLMUsage.output_tokens).label("output_tokens"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.cache_read_tokens).label("cache_read_tokens"),
            func.sum(LLMUsage.cache_write_tokens).label("cache_write_tokens"),
            func.sum(case((LLMUsage.status == "error", 1), else_=0)).label("error_count"),
            func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
        )
        .group_by(*group_cols)
        .order_by(desc("total_cost"), desc("total_tokens"))
        .all()
    )

    out: List[UsageGroup] = []
    for r in rows:
        values = tuple(r)[: len(group_cols)]
        if group_by == "route":
            facts = route_facts(values[0], values[1])
            key = facts["key"]
        elif group_by == "provider":
            facts = route_facts(None, values[0])
            facts["label"] = facts["provider_label"]
            key = facts["provider"]
        elif group_by == "model":
            facts = {"model_id": values[0] or "unknown", "label": values[0] or "unknown"}
            key = str(values[0] or "unknown")
        else:
            facts = {}
            key = str(values[0] if values[0] is not None else "unknown")
        out.append(UsageGroup(
            key=key,
            request_count=r.request_count,
            input_tokens=int(r.input_tokens or 0),
            output_tokens=int(r.output_tokens or 0),
            total_tokens=int(r.total_tokens or 0),
            total_cost=float(r.total_cost or 0),
            cache_read_tokens=int(r.cache_read_tokens or 0),
            cache_write_tokens=int(r.cache_write_tokens or 0),
            error_count=int(r.error_count or 0),
            avg_latency_ms=float(r.avg_latency_ms) if r.avg_latency_ms is not None else None,
            model_id=facts.get("model_id"),
            provider=facts.get("provider"),
            provider_label=facts.get("provider_label"),
            billing=facts.get("billing"),
            label=facts.get("label"),
        ))
    return out


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
            "is_byok": LLMUsage.is_byok,
        }
        group_col = col_map.get(breakdown, LLMUsage.model_id)

    rows = (
        _calls(db, ctx.workspace_id, since)
        .with_entities(
            group_col.label("key"),
            func.sum(LLMUsage.input_cost).label("input_cost"),
            func.sum(LLMUsage.output_cost).label("output_cost"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.count(LLMUsage.id).label("request_count"),
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
    """Dashboard summary: totals, cache share, top routes, per-provider split, cost trend."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)
    base = _calls(db, ctx.workspace_id, since)

    agg = base.with_entities(
        func.count(LLMUsage.id).label("cnt"),
        func.sum(LLMUsage.total_tokens).label("tokens"),
        func.sum(LLMUsage.total_cost).label("cost"),
        func.avg(LLMUsage.latency_ms).label("latency"),
        func.sum(LLMUsage.cache_read_tokens).label("cache_read"),
        func.sum(LLMUsage.cache_write_tokens).label("cache_write"),
        func.sum(case((LLMUsage.status == "error", 1), else_=0)).label("errors"),
    ).first()

    total_requests = agg.cnt or 0
    error_count = int(agg.errors or 0)
    error_rate = error_count / total_requests if total_requests > 0 else 0.0

    # Top ROUTES by cost, then by tokens (a free route that carried the work
    # still shows up — cost alone would hide every NVIDIA and subscription call)
    top = (
        base.with_entities(
            LLMUsage.model_id,
            LLMUsage.provider,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
            func.count(LLMUsage.id).label("cnt"),
        )
        .group_by(LLMUsage.model_id, LLMUsage.provider)
        .order_by(desc("cost"), desc("tokens"))
        .limit(5)
        .all()
    )

    provider_rows = (
        base.with_entities(
            LLMUsage.provider,
            func.count(LLMUsage.id).label("cnt"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(case((LLMUsage.status == "error", 1), else_=0)).label("errors"),
        )
        .group_by(LLMUsage.provider)
        .order_by(desc("cost"), desc("tokens"))
        .all()
    )
    by_provider = []
    for r in provider_rows:
        facts = describe_usage_provider(r.provider)
        by_provider.append(ProviderUsage(
            provider=facts["slug"], label=facts["label"], billing=facts["billing"], kind=facts["kind"],
            request_count=r.cnt or 0, total_tokens=int(r.tokens or 0), total_cost=float(r.cost or 0),
            error_count=int(r.errors or 0),
        ))

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
        cache_read_tokens=int(agg.cache_read or 0),
        cache_write_tokens=int(agg.cache_write or 0),
        by_provider=by_provider,
        top_models=[
            {
                **route_facts(m.model_id, m.provider),
                "total_cost": float(m.cost or 0),
                "total_tokens": int(m.tokens or 0),
                "request_count": m.cnt,
            }
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
            LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE,
            LLMUsage.agent_id.isnot(None),
        )
        .group_by(LLMUsage.agent_id, LLMUsage.model_id)
        .having(func.count(LLMUsage.id) >= 10)
        .all()
    )

    premium_models = {m.strip() for m in config.PREMIUM_MODELS.split(",") if m.strip()}
    budget_suggestions = config.BUDGET_MODELS
    savings_ratio = config.PREMIUM_TO_BUDGET_SAVINGS_RATIO

    for row in agent_usage:
        if row.model_id in premium_models and (row.avg_output or 0) < 200:
            potential = float(row.cost or 0) * savings_ratio
            recs.append(Recommendation(
                type="cost_optimization",
                title=f"Switch Agent {row.agent_id} to a cheaper model",
                description=(
                    f"Agent {row.agent_id} used {row.model_id} for {row.cnt} requests "
                    f"with avg {int(row.avg_output or 0)} output tokens. "
                    f"Consider {budget_suggestions} for simple outputs."
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


# ── Daily Cost by Model (for multi-line chart) ──────────────────────


@router.get("/costs/daily-by-model")
async def get_daily_costs_by_model(
    period: str = Query("30d", description="7d|30d|90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Daily cost per ROUTE (model × serving provider) for the multi-line chart.

    ``models`` holds the series keys (``<model>@<provider>``), ``routes`` the
    facts behind each key (label, provider, billing) — a free route and a paid
    route for the same vendor model are two lines.
    """
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    since = _period_start(period)

    rows = (
        _calls(db, ctx.workspace_id, since)
        .with_entities(
            func.date(LLMUsage.created_at).label("day"),
            LLMUsage.model_id,
            LLMUsage.provider,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
            func.count(LLMUsage.id).label("requests"),
        )
        .group_by(func.date(LLMUsage.created_at), LLMUsage.model_id, LLMUsage.provider)
        .order_by("day")
        .all()
    )

    date_map: Dict[str, Dict[str, float]] = {}
    routes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        facts = route_facts(r.model_id, r.provider)
        key = facts["key"]
        routes.setdefault(key, {**facts, "total_cost": 0.0, "total_tokens": 0, "request_count": 0})
        routes[key]["total_cost"] += float(r.cost or 0)
        routes[key]["total_tokens"] += int(r.tokens or 0)
        routes[key]["request_count"] += int(r.requests or 0)
        date_map.setdefault(str(r.day), {})[key] = round(float(r.cost or 0), 6)

    ordered = sorted(routes.values(), key=lambda f: (-f["total_cost"], -f["total_tokens"], f["key"]))
    keys = [f["key"] for f in ordered]
    series = []
    for day_str in sorted(date_map.keys()):
        entry: Dict[str, Any] = {"date": day_str}
        for k in keys:
            entry[k] = date_map[day_str].get(k, 0)
        series.append(entry)

    return {
        "models": keys,
        "routes": [{**f, "total_cost": round(f["total_cost"], 6)} for f in ordered],
        "series": series,
    }


# ── Model Comparison ─────────────────────────────────────────────────


class ModelComparisonItem(BaseModel):
    model_id: str
    display_name: str
    provider: str
    provider_label: Optional[str] = None
    billing: Optional[str] = None
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

    for raw in ids:
        # A route key (``model@provider``) pins the serving provider; a bare
        # model id compares the route that served most of its usage.
        mid, _, wanted_provider = raw.partition(ROUTE_KEY_SEPARATOR)
        served = (
            _calls(db, ctx.workspace_id, since)
            .filter(LLMUsage.model_id == mid)
            .with_entities(LLMUsage.provider, func.count(LLMUsage.id).label("cnt"))
            .group_by(LLMUsage.provider)
            .order_by(desc("cnt"))
            .first()
        )
        provider_slug = wanted_provider or (served.provider if served else None)

        registry = None
        if provider_slug:
            registry = (
                db.query(LLMModel)
                .filter(LLMModel.model_id == mid, LLMModel.serving_provider == provider_slug)
                .first()
            )
        if registry is None:
            registry = db.query(LLMModel).filter(LLMModel.model_id == mid).first()
        provider_slug = provider_slug or (registry.serving_provider if registry else None)
        facts = describe_usage_provider(provider_slug)

        usage_q = _calls(db, ctx.workspace_id, since).filter(LLMUsage.model_id == mid)
        if provider_slug:
            usage_q = usage_q.filter(LLMUsage.provider == provider_slug)
        usage = usage_q.with_entities(
            func.count(LLMUsage.id).label("total_requests"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
            func.sum(case((LLMUsage.status == "error", 1), else_=0)).label("errors"),
        ).first()

        total_requests = usage.total_requests or 0 if usage else 0
        error_count = int(usage.errors or 0) if usage else 0
        error_rate = error_count / total_requests if total_requests > 0 else 0.0
        success_rate = 1.0 - error_rate

        results.append(ModelComparisonItem(
            model_id=mid,
            display_name=registry.display_name if registry else mid,
            provider=facts["slug"],
            provider_label=facts["label"],
            billing=facts["billing"],
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
    label: Optional[str] = None
    model_id: Optional[str] = None
    provider: Optional[str] = None
    provider_label: Optional[str] = None
    billing: Optional[str] = None
    current_period_tokens: int = 0


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

    base = _calls(db, ctx.workspace_id, since)
    current_cost = float(base.with_entities(func.sum(LLMUsage.total_cost)).scalar() or 0.0)

    # Distinct days with data — the honest daily average for sparse usage
    days_with_data = (
        base.with_entities(func.count(func.distinct(func.date(LLMUsage.created_at)))).scalar() or 0
    )

    daily_avg = current_cost / days_with_data if days_with_data > 0 else 0.0
    projected_monthly = daily_avg * 30

    prev_start = since - delta
    prev_cost = float(
        db.query(func.sum(LLMUsage.total_cost))
        .filter(
            LLMUsage.workspace_id == ctx.workspace_id,
            LLMUsage.created_at >= prev_start,
            LLMUsage.created_at < since,
            LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE,
        )
        .scalar() or 0.0
    )

    change_percent = None
    if prev_cost > 0:
        change_percent = round(((current_cost - prev_cost) / prev_cost) * 100, 2)

    by_route_rows = (
        base.with_entities(
            LLMUsage.model_id,
            LLMUsage.provider,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
        )
        .group_by(LLMUsage.model_id, LLMUsage.provider)
        .order_by(desc("cost"), desc("tokens"))
        .all()
    )

    projected_by_model = []
    for r in by_route_rows:
        route_cost = float(r.cost or 0)
        route_daily = route_cost / days_with_data if days_with_data > 0 else 0.0
        facts = route_facts(r.model_id, r.provider)
        projected_by_model.append(ProjectedItem(
            key=facts["key"],
            projected_monthly_cost=round(route_daily * 30, 6),
            current_period_cost=round(route_cost, 6),
            current_period_tokens=int(r.tokens or 0),
            label=facts["label"],
            model_id=facts["model_id"],
            provider=facts["provider"],
            provider_label=facts["provider_label"],
            billing=facts["billing"],
        ))

    by_provider_rows = (
        base.with_entities(
            LLMUsage.provider.label("key"),
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
        )
        .group_by(LLMUsage.provider)
        .order_by(desc("cost"), desc("tokens"))
        .all()
    )

    projected_by_provider = []
    for r in by_provider_rows:
        prov_cost = float(r.cost or 0)
        prov_daily = prov_cost / days_with_data if days_with_data > 0 else 0.0
        facts = describe_usage_provider(r.key)
        projected_by_provider.append(ProjectedItem(
            key=facts["slug"],
            projected_monthly_cost=round(prov_daily * 30, 6),
            current_period_cost=round(prov_cost, 6),
            current_period_tokens=int(r.tokens or 0),
            label=facts["label"],
            provider=facts["slug"],
            provider_label=facts["label"],
            billing=facts["billing"],
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

def _resolve_openrouter_key(workspace_id, db: Session, byok_only: bool = False) -> str:
    """
    Find the OpenRouter API key for a workspace.
    Priority: BYOK key in user_api_keys → env OPENROUTER_API_KEY.
    If byok_only=True, only returns workspace BYOK keys (blocks env fallback).
    Raises HTTPException 404 if not found.
    """
    row = (
        db.query(UserApiKey)
        .filter(
            UserApiKey.workspace_id == workspace_id,
            UserApiKey.provider == "openrouter",
            UserApiKey.is_active.is_(True),
        )
        .order_by(UserApiKey.created_at.desc())
        .first()
    )
    if row:
        try:
            return get_encryption_service().decrypt(row.encrypted_key)
        except Exception:
            logger.warning("Failed to decrypt OpenRouter key id=%s", row.id)

    if not byok_only:
        env_key = config.OPENROUTER_API_KEY
        if env_key:
            return env_key

    raise HTTPException(
        404,
        "No OpenRouter API key configured. Add one in Settings → API Keys."
        if byok_only
        else "No OpenRouter API key configured. Add one in Settings → API Keys or set OPENROUTER_API_KEY.",
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


@sync_router.post(
    "/openrouter/sync",
    response_model=OpenRouterSyncResponse,
    summary="Trigger OpenRouter activity sync",
)
async def sync_openrouter_activity(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Sync OpenRouter activity data into llm_usage for the current workspace.
    Only works with workspace BYOK keys to prevent cross-workspace data duplication."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    api_key = _resolve_openrouter_key(ctx.workspace_id, db, byok_only=True)

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


# ── Admin Analytics schemas ──────────────────────────────────────────


class WorkspaceCostEntry(BaseModel):
    workspace_id: str
    workspace_name: str
    plan: str
    total_cost: float
    total_tokens: int
    total_requests: int
    top_model: Optional[str] = None


class ByokCostSplit(BaseModel):
    platform_cost: float
    platform_requests: int
    byok_cost: float
    byok_requests: int


class AdminCostAnalyticsResponse(BaseModel):
    total_platform_cost: float
    total_tokens: int
    total_requests: int
    byok_split: Optional[ByokCostSplit] = None
    cost_by_workspace: List[WorkspaceCostEntry] = Field(default_factory=list)
    cost_by_provider: List[CostBreakdown] = Field(default_factory=list)
    daily_cost_trend: List[Dict[str, Any]] = Field(default_factory=list)


# ── Admin Analytics endpoints ────────────────────────────────────────


@admin_router.get("/costs", response_model=AdminCostAnalyticsResponse)
async def get_admin_cost_analytics(
    period: str = Query("30d", description="7d|30d|90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Platform-wide cost analytics across all workspaces (super-admin via the router-wide lock)."""
    delta = PERIOD_MAP.get(period, timedelta(days=30))
    since = datetime.utcnow() - delta

    # Platform-wide aggregates (no workspace filter)
    agg = (
        db.query(
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.count(LLMUsage.id).label("total_requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .first()
    )

    total_platform_cost = float(agg.total_cost or 0) if agg else 0.0
    total_tokens = int(agg.total_tokens or 0) if agg else 0
    total_requests = agg.total_requests or 0 if agg else 0

    # Cost by workspace
    ws_rows = (
        db.query(
            LLMUsage.workspace_id,
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.count(LLMUsage.id).label("total_requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.workspace_id)
        .order_by(desc("total_cost"))
        .all()
    )

    # Lookup workspace names and plans
    ws_ids = [r.workspace_id for r in ws_rows if r.workspace_id]
    ws_map: Dict[Any, Any] = {}
    if ws_ids:
        workspaces = db.query(Workspace).filter(Workspace.id.in_(ws_ids)).all()
        ws_map = {str(w.id): w for w in workspaces}

    # Top model per workspace (subquery for each)
    cost_by_workspace = []
    for r in ws_rows:
        ws_id_str = str(r.workspace_id) if r.workspace_id else "unknown"
        ws = ws_map.get(ws_id_str)

        # Find top model for this workspace in the period
        top_model_row = (
            db.query(
                LLMUsage.model_id,
                func.sum(LLMUsage.total_cost).label("cost"),
            )
            .filter(
                LLMUsage.workspace_id == r.workspace_id,
                LLMUsage.created_at >= since,
                LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE,
            )
            .group_by(LLMUsage.model_id)
            .order_by(desc("cost"))
            .first()
        )

        cost_by_workspace.append(WorkspaceCostEntry(
            workspace_id=ws_id_str,
            workspace_name=ws.name if ws else ws_id_str,
            plan=ws.plan if ws else "unknown",
            total_cost=round(float(r.total_cost or 0), 6),
            total_tokens=int(r.total_tokens or 0),
            total_requests=r.total_requests or 0,
            top_model=top_model_row.model_id if top_model_row else None,
        ))

    # Cost by provider (platform-wide)
    provider_rows = (
        db.query(
            LLMUsage.provider.label("key"),
            func.sum(LLMUsage.input_cost).label("input_cost"),
            func.sum(LLMUsage.output_cost).label("output_cost"),
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.count(LLMUsage.id).label("request_count"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.provider)
        .order_by(desc("total_cost"))
        .all()
    )

    cost_by_provider = [
        CostBreakdown(
            key=describe_usage_provider(r.key)["slug"],
            input_cost=float(r.input_cost or 0),
            output_cost=float(r.output_cost or 0),
            total_cost=float(r.total_cost or 0),
            request_count=r.request_count,
        )
        for r in provider_rows
    ]

    # Daily cost trend (platform-wide)
    trend_rows = (
        db.query(
            func.date(LLMUsage.created_at).label("day"),
            func.sum(LLMUsage.total_cost).label("cost"),
            func.count(LLMUsage.id).label("requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(func.date(LLMUsage.created_at))
        .order_by("day")
        .all()
    )

    daily_cost_trend = [
        {
            "date": str(t.day),
            "cost": round(float(t.cost or 0), 6),
            "requests": t.requests or 0,
        }
        for t in trend_rows
    ]

    # BYOK vs platform cost split
    byok_rows = (
        db.query(
            LLMUsage.is_byok,
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.count(LLMUsage.id).label("request_count"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.is_byok)
        .all()
    )
    byok_split_data = {r.is_byok: r for r in byok_rows}
    platform_row = byok_split_data.get(False)
    byok_row = byok_split_data.get(True)
    byok_split = ByokCostSplit(
        platform_cost=round(float(platform_row.total_cost or 0), 6) if platform_row else 0.0,
        platform_requests=platform_row.request_count or 0 if platform_row else 0,
        byok_cost=round(float(byok_row.total_cost or 0), 6) if byok_row else 0.0,
        byok_requests=byok_row.request_count or 0 if byok_row else 0,
    )

    return AdminCostAnalyticsResponse(
        total_platform_cost=round(total_platform_cost, 6),
        total_tokens=total_tokens,
        total_requests=total_requests,
        byok_split=byok_split,
        cost_by_workspace=cost_by_workspace,
        cost_by_provider=cost_by_provider,
        daily_cost_trend=daily_cost_trend,
    )


# ── Comprehensive Admin Dashboard ────────────────────────────────────


@admin_router.get("/dashboard")
async def get_admin_dashboard(
    period: str = Query("30d", description="7d|30d|90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Comprehensive admin dashboard: workspaces with resource counts,
    platform-wide model usage, daily cost by provider, top models,
    and BYOK cost split.  All in a single call.
    """
    delta = PERIOD_MAP.get(period, timedelta(days=30))
    since = datetime.utcnow() - delta

    # ── Platform-wide aggregates ──
    agg = (
        db.query(
            func.sum(LLMUsage.total_cost).label("total_cost"),
            func.sum(LLMUsage.total_tokens).label("total_tokens"),
            func.count(LLMUsage.id).label("total_requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .first()
    )
    total_cost = float(agg.total_cost or 0) if agg else 0.0
    total_tokens = int(agg.total_tokens or 0) if agg else 0
    total_requests = agg.total_requests or 0 if agg else 0

    # ── All workspaces ──
    all_ws = db.query(Workspace).filter(Workspace.is_active.is_(True)).all()
    total_workspaces = len(all_ws)

    # Agent / recipe counts per workspace  (batch queries)
    agent_counts = dict(
        db.query(Agent.workspace_id, func.count(Agent.id))
        .group_by(Agent.workspace_id)
        .all()
    )
    recipe_counts = dict(
        db.query(WorkflowRecipe.workspace_id, func.count(WorkflowRecipe.id))
        .filter(WorkflowRecipe.owner_type == "workspace")
        .group_by(WorkflowRecipe.workspace_id)
        .all()
    )
    execution_counts = dict(
        db.query(RecipeExecution.workspace_id, func.count(RecipeExecution.id))
        .filter(RecipeExecution.started_at >= since)
        .group_by(RecipeExecution.workspace_id)
        .all()
    )

    # LLM usage per workspace
    ws_usage_rows = (
        db.query(
            LLMUsage.workspace_id,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
            func.count(LLMUsage.id).label("requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.workspace_id)
        .all()
    )
    ws_usage_map = {str(r.workspace_id): r for r in ws_usage_rows}

    workspaces = []
    for ws in all_ws:
        ws_id = str(ws.id)
        usage = ws_usage_map.get(ws_id)
        workspaces.append({
            "id": ws_id,
            "name": ws.name,
            "plan": ws.plan or "basic",  # PRD-222 W2·S1: entry tier (renamed from 'starter')
            "is_personal": ws.is_personal,
            "created_at": ws.created_at.isoformat() if ws.created_at else None,
            "agents": agent_counts.get(ws.id, 0),
            "recipes": recipe_counts.get(ws.id, 0),
            "executions": execution_counts.get(ws.id, 0),
            "cost": round(float(usage.cost or 0), 6) if usage else 0.0,
            "tokens": int(usage.tokens or 0) if usage else 0,
            "requests": usage.requests or 0 if usage else 0,
        })

    # Sort by cost descending
    workspaces.sort(key=lambda w: w["cost"], reverse=True)

    # ── Top models platform-wide ──
    top_models = (
        db.query(
            LLMUsage.model_id,
            LLMUsage.provider,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.sum(LLMUsage.total_tokens).label("tokens"),
            func.count(LLMUsage.id).label("requests"),
            func.count(func.distinct(LLMUsage.workspace_id)).label("workspace_count"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.model_id, LLMUsage.provider)
        .order_by(desc("cost"))
        .limit(15)
        .all()
    )

    models = [
        {
            **route_facts(r.model_id, r.provider),
            "cost": round(float(r.cost or 0), 6),
            "tokens": int(r.tokens or 0),
            "requests": r.requests or 0,
            "workspace_count": r.workspace_count or 0,
        }
        for r in top_models
    ]

    # ── Daily cost by provider (for stacked area chart) ──
    daily_provider_rows = (
        db.query(
            func.date(LLMUsage.created_at).label("day"),
            LLMUsage.provider,
            func.sum(LLMUsage.total_cost).label("cost"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(func.date(LLMUsage.created_at), LLMUsage.provider)
        .order_by("day")
        .all()
    )

    # Pivot into chart-ready series
    providers_set: set = set()
    day_map: Dict[str, Dict[str, float]] = {}
    for r in daily_provider_rows:
        d = str(r.day)
        p = r.provider or "unknown"
        providers_set.add(p)
        if d not in day_map:
            day_map[d] = {}
        day_map[d][p] = round(float(r.cost or 0), 6)

    providers = sorted(providers_set)
    provider_labels = {p: describe_usage_provider(p)["label"] for p in providers}
    daily_by_provider = []
    for d in sorted(day_map.keys()):
        entry: Dict[str, Any] = {"date": d}
        for p in providers:
            entry[p] = day_map[d].get(p, 0)
        daily_by_provider.append(entry)

    # ── BYOK vs Platform split ──
    byok_rows = (
        db.query(
            LLMUsage.is_byok,
            func.sum(LLMUsage.total_cost).label("cost"),
            func.count(LLMUsage.id).label("requests"),
        )
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .group_by(LLMUsage.is_byok)
        .all()
    )
    byok_map = {r.is_byok: r for r in byok_rows}
    p_row = byok_map.get(False)
    b_row = byok_map.get(True)

    # ── Revenue metrics ──
    # Compute daily average and projected monthly (for billing planning)
    days_with_data = (
        db.query(func.count(func.distinct(func.date(LLMUsage.created_at))))
        .filter(LLMUsage.created_at >= since, LLMUsage.request_type != ACTIVITY_SYNC_REQUEST_TYPE)
        .scalar() or 0
    )
    daily_avg = total_cost / days_with_data if days_with_data > 0 else 0.0

    return {
        "overview": {
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "total_workspaces": total_workspaces,
            "daily_average": round(daily_avg, 6),
            "projected_monthly": round(daily_avg * 30, 6),
        },
        "byok_split": {
            "platform_cost": round(float(p_row.cost or 0), 6) if p_row else 0.0,
            "platform_requests": p_row.requests or 0 if p_row else 0,
            "byok_cost": round(float(b_row.cost or 0), 6) if b_row else 0.0,
            "byok_requests": b_row.requests or 0 if b_row else 0,
        },
        "workspaces": workspaces,
        "models": models,
        "daily_by_provider": {
            "providers": providers,
            "labels": provider_labels,
            "series": daily_by_provider,
        },
    }
