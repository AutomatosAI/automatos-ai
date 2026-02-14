"""
PandasAI Analytics Chart Generation API (PRD-54 / US-004)
=========================================================

AI-powered chart generation from LLM usage data using PandasAI.
Supports natural-language queries and pre-built chart presets.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import LLMUsage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/charts", tags=["Analytics Charts"])


# ── Pydantic schemas ─────────────────────────────────────────────────

class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    pie = "pie"
    area = "area"
    scatter = "scatter"
    auto = "auto"


class ChartGenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language query describing the chart")
    chart_type: ChartType = Field(ChartType.auto, description="Chart type hint")


class ChartGenerateResponse(BaseModel):
    summary: str
    charts: List[str] = Field(default_factory=list, description="Base64-encoded PNG chart images")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Raw data rows")


class ChartPreset(BaseModel):
    id: str
    title: str
    description: str
    query: str
    chart_type: str


# ── Chart presets ─────────────────────────────────────────────────────

CHART_PRESETS: List[ChartPreset] = [
    ChartPreset(
        id="cost-by-model",
        title="Cost by Model",
        description="Total cost breakdown across all models used",
        query="Create a pie chart showing total cost by model",
        chart_type="pie",
    ),
    ChartPreset(
        id="tokens-over-time",
        title="Tokens Over Time",
        description="Daily token usage trend over the selected period",
        query="Create a line chart showing total tokens used per day over time",
        chart_type="line",
    ),
    ChartPreset(
        id="cost-trend-daily",
        title="Daily Cost Trend",
        description="Daily spending trend with input and output cost breakdown",
        query="Create a stacked area chart showing daily input_cost and output_cost over time",
        chart_type="area",
    ),
    ChartPreset(
        id="requests-by-provider",
        title="Requests by Provider",
        description="Number of API requests grouped by provider",
        query="Create a bar chart showing total request count by provider",
        chart_type="bar",
    ),
    ChartPreset(
        id="latency-by-model",
        title="Latency by Model",
        description="Average latency comparison across models",
        query="Create a horizontal bar chart showing average latency_ms by model, sorted highest to lowest",
        chart_type="bar",
    ),
    ChartPreset(
        id="cost-by-request-type",
        title="Cost by Request Type",
        description="Cost distribution across request types (chat, agent, recipe, etc.)",
        query="Create a pie chart showing total cost grouped by request_type",
        chart_type="pie",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────

def _query_llm_usage(db: Session, workspace_id: Any, query_lower: str) -> List[Dict[str, Any]]:
    """
    Query llm_usage data based on parsed intent from the natural language query.
    Returns list of row dicts suitable for PandasAI.
    """
    # Determine time range from query
    days = 30  # default
    if "7 day" in query_lower or "week" in query_lower or "7d" in query_lower:
        days = 7
    elif "90 day" in query_lower or "quarter" in query_lower or "90d" in query_lower:
        days = 90
    elif "24 hour" in query_lower or "today" in query_lower or "1d" in query_lower:
        days = 1

    since = datetime.utcnow() - timedelta(days=days)

    base_filter = [
        LLMUsage.workspace_id == workspace_id,
        LLMUsage.created_at >= since,
    ]

    # Determine grouping and metrics based on query intent
    if "over time" in query_lower or "trend" in query_lower or "daily" in query_lower or "per day" in query_lower:
        # Time-series query
        rows = (
            db.query(
                func.date(LLMUsage.created_at).label("date"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.sum(LLMUsage.input_cost).label("input_cost"),
                func.sum(LLMUsage.output_cost).label("output_cost"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.sum(LLMUsage.input_tokens).label("input_tokens"),
                func.sum(LLMUsage.output_tokens).label("output_tokens"),
                func.count(LLMUsage.id).label("request_count"),
            )
            .filter(*base_filter)
            .group_by(func.date(LLMUsage.created_at))
            .order_by(func.date(LLMUsage.created_at))
            .all()
        )
    elif "provider" in query_lower:
        rows = (
            db.query(
                LLMUsage.provider.label("provider"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.count(LLMUsage.id).label("request_count"),
                func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
            )
            .filter(*base_filter)
            .group_by(LLMUsage.provider)
            .order_by(desc("total_cost"))
            .all()
        )
    elif "request_type" in query_lower or "request type" in query_lower:
        rows = (
            db.query(
                LLMUsage.request_type.label("request_type"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.count(LLMUsage.id).label("request_count"),
            )
            .filter(*base_filter)
            .group_by(LLMUsage.request_type)
            .order_by(desc("total_cost"))
            .all()
        )
    elif "latency" in query_lower:
        rows = (
            db.query(
                LLMUsage.model_id.label("model"),
                func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
                func.min(LLMUsage.latency_ms).label("min_latency_ms"),
                func.max(LLMUsage.latency_ms).label("max_latency_ms"),
                func.count(LLMUsage.id).label("request_count"),
            )
            .filter(*base_filter)
            .group_by(LLMUsage.model_id)
            .order_by(desc("avg_latency_ms"))
            .all()
        )
    else:
        # Default: group by model
        rows = (
            db.query(
                LLMUsage.model_id.label("model"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.sum(LLMUsage.input_cost).label("input_cost"),
                func.sum(LLMUsage.output_cost).label("output_cost"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.count(LLMUsage.id).label("request_count"),
                func.avg(LLMUsage.latency_ms).label("avg_latency_ms"),
            )
            .filter(*base_filter)
            .group_by(LLMUsage.model_id)
            .order_by(desc("total_cost"))
            .all()
        )

    # Convert SQLAlchemy rows to dicts
    result = []
    for row in rows:
        d = {}
        for key in row._fields:
            val = getattr(row, key)
            if val is not None:
                # Ensure JSON-serializable types
                if isinstance(val, datetime):
                    d[key] = val.isoformat()
                else:
                    d[key] = float(val) if isinstance(val, (int, float)) else str(val)
            else:
                d[key] = None
        result.append(d)

    return result


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/generate", response_model=ChartGenerateResponse)
async def generate_chart(
    body: ChartGenerateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Generate an AI-powered chart from LLM usage data using PandasAI."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    from modules.tools.services.pandas_ai_service import get_pandasai_service

    svc = get_pandasai_service()
    if svc is None:
        raise HTTPException(
            404,
            "PandasAI service is not available. Ensure pandasai is installed and configured.",
        )

    query_lower = body.query.lower()

    # Enrich query with chart type hint if not auto
    enriched_query = body.query
    if body.chart_type != ChartType.auto:
        enriched_query = f"Create a {body.chart_type.value} chart: {body.query}"

    # Query usage data based on intent
    data_rows = _query_llm_usage(db, ctx.workspace_id, query_lower)
    if not data_rows:
        return ChartGenerateResponse(
            summary="No usage data found for the selected period.",
            charts=[],
            data=[],
        )

    # Generate insight via PandasAI
    columns = list(data_rows[0].keys()) if data_rows else None
    result = svc.generate_insight(enriched_query, data_rows, columns)

    if result is None:
        return ChartGenerateResponse(
            summary="Chart generation returned no results.",
            charts=[],
            data=data_rows,
        )

    # Extract base64 chart strings
    chart_images = [c["base64"] for c in result.get("charts", []) if c.get("base64")]

    return ChartGenerateResponse(
        summary=result.get("summary", ""),
        charts=chart_images,
        data=data_rows,
    )


@router.get("/presets", response_model=List[ChartPreset])
async def get_chart_presets(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return list of pre-built chart configurations."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    return CHART_PRESETS
