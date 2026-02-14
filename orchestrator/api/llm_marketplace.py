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


@router.get("/models/{model_id:path}", response_model=LLMModelOut)
async def get_model_detail(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full detail for a single model."""
    m = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
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

    m = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
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


@router.delete("/models/{model_id:path}/uninstall", response_model=InstallResult)
async def uninstall_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a model from the current workspace."""
    if not ctx.workspace_id:
        raise HTTPException(400, "Workspace context required")

    m = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
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
