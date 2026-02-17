"""
Admin System Prompts API (PRD-58)
=================================

REST endpoints for managing system prompts:
- List / get prompts
- Create new versions (draft)
- Activate / rollback versions
- Delete draft versions
- List evaluation runs (Phase 1B)

All endpoints require admin role.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.system_prompts import (
    SystemPrompt,
    SystemPromptVersion,
    SystemPromptEvalRun,
    SystemPromptResponse,
    SystemPromptVersionResponse,
    CreateVersionRequest,
    EvalRunResponse,
)
from core.services.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/prompts", tags=["Admin Prompts"])


# ===================================================================
# Helpers
# ===================================================================

def _assert_admin(ctx: RequestContext) -> None:
    if not ctx.user or getattr(ctx.user, "system_role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def _prompt_to_response(prompt: SystemPrompt, db: Session) -> SystemPromptResponse:
    """Convert ORM object to response, denormalizing the active version."""
    active_version_number = None
    active_content = None
    active_eval_scores = None

    if prompt.active_version_id:
        av = (
            db.query(SystemPromptVersion)
            .filter(SystemPromptVersion.id == prompt.active_version_id)
            .first()
        )
        if av:
            active_version_number = av.version
            active_content = av.content
            active_eval_scores = av.eval_scores

    return SystemPromptResponse(
        id=str(prompt.id),
        slug=prompt.slug,
        name=prompt.name,
        description=prompt.description,
        category=prompt.category,
        source_file=prompt.source_file,
        variables=prompt.variables or [],
        impact_description=prompt.impact_description,
        active_version_id=str(prompt.active_version_id) if prompt.active_version_id else None,
        active_version_number=active_version_number,
        active_content=active_content,
        active_eval_scores=active_eval_scores,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


def _version_to_response(v: SystemPromptVersion) -> SystemPromptVersionResponse:
    return SystemPromptVersionResponse(
        id=str(v.id),
        prompt_id=str(v.prompt_id),
        version=v.version,
        content=v.content,
        change_notes=v.change_notes,
        status=v.status,
        eval_scores=v.eval_scores,
        eval_model=v.eval_model,
        eval_run_at=v.eval_run_at,
        traffic_percentage=v.traffic_percentage,
        created_by=v.created_by,
        created_at=v.created_at,
    )


# ===================================================================
# Prompt CRUD
# ===================================================================

@router.get("/", response_model=List[SystemPromptResponse])
async def list_prompts(
    category: Optional[str] = Query(None, description="Filter by category"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all system prompts with active version info."""
    _assert_admin(ctx)

    query = db.query(SystemPrompt)
    if category:
        query = query.filter(SystemPrompt.category == category)

    prompts = query.order_by(SystemPrompt.category, SystemPrompt.name).all()
    return [_prompt_to_response(p, db) for p in prompts]


@router.get("/{slug}", response_model=SystemPromptResponse)
async def get_prompt(
    slug: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single prompt with its active version."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    return _prompt_to_response(prompt, db)


# ===================================================================
# Version Management
# ===================================================================

@router.get("/{slug}/versions", response_model=List[SystemPromptVersionResponse])
async def list_versions(
    slug: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all versions of a prompt, newest first."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    versions = (
        db.query(SystemPromptVersion)
        .filter(SystemPromptVersion.prompt_id == prompt.id)
        .order_by(desc(SystemPromptVersion.version))
        .all()
    )
    return [_version_to_response(v) for v in versions]


@router.post("/{slug}/versions", response_model=SystemPromptVersionResponse, status_code=201)
async def create_version(
    slug: str,
    body: CreateVersionRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new draft version of a prompt."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    # Determine next version number
    max_version = (
        db.query(func.max(SystemPromptVersion.version))
        .filter(SystemPromptVersion.prompt_id == prompt.id)
        .scalar()
    ) or 0

    version = SystemPromptVersion(
        prompt_id=prompt.id,
        version=max_version + 1,
        content=body.content,
        change_notes=body.change_notes,
        status="draft",
        created_by=getattr(ctx.user, "email", "admin") if ctx.user else "admin",
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    logger.info("Created version %d for prompt '%s'", version.version, slug)
    return _version_to_response(version)


@router.put("/{slug}/activate/{version_id}")
async def activate_version(
    slug: str,
    version_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Activate a specific version, archiving the previously active one."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    version = (
        db.query(SystemPromptVersion)
        .filter(
            SystemPromptVersion.id == version_id,
            SystemPromptVersion.prompt_id == prompt.id,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    # Archive currently active version
    if prompt.active_version_id:
        old = (
            db.query(SystemPromptVersion)
            .filter(SystemPromptVersion.id == prompt.active_version_id)
            .first()
        )
        if old and old.id != version.id:
            old.status = "archived"

    # Activate the new version
    version.status = "active"
    prompt.active_version_id = version.id
    db.commit()

    # Clear cache so next request picks up the new version
    PromptRegistry.clear_cache()

    logger.info("Activated version %d for prompt '%s'", version.version, slug)
    return {"status": "ok", "activated_version": version.version}


@router.put("/{slug}/rollback/{version_id}")
async def rollback_version(
    slug: str,
    version_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Rollback to a previous version (same as activate but semantically different)."""
    return await activate_version(slug, version_id, ctx, db)


@router.delete("/{slug}/versions/{version_id}")
async def delete_version(
    slug: str,
    version_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a draft version. Cannot delete active versions."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    version = (
        db.query(SystemPromptVersion)
        .filter(
            SystemPromptVersion.id == version_id,
            SystemPromptVersion.prompt_id == prompt.id,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    if version.status == "active":
        raise HTTPException(status_code=400, detail="Cannot delete the active version")

    db.delete(version)
    db.commit()

    logger.info("Deleted version %d for prompt '%s'", version.version, slug)
    return {"status": "ok", "deleted_version": version.version}


# ===================================================================
# Evaluation Runs (Phase 1B — stubs for now)
# ===================================================================

@router.get("/{slug}/eval-runs", response_model=List[EvalRunResponse])
async def list_eval_runs(
    slug: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List evaluation runs for a prompt's versions."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{slug}' not found")

    version_ids = [v.id for v in prompt.versions]
    if not version_ids:
        return []

    runs = (
        db.query(SystemPromptEvalRun)
        .filter(SystemPromptEvalRun.version_id.in_(version_ids))
        .order_by(desc(SystemPromptEvalRun.started_at))
        .all()
    )

    return [
        EvalRunResponse(
            id=str(r.id),
            version_id=str(r.version_id),
            eval_type=r.eval_type,
            metrics=r.metrics,
            model_used=r.model_used,
            dataset_size=r.dataset_size,
            algorithm=r.algorithm,
            overall_score=r.overall_score,
            passed=r.passed,
            optimized_content=r.optimized_content,
            started_at=r.started_at,
            completed_at=r.completed_at,
            status=r.status,
        )
        for r in runs
    ]


# ===================================================================
# Prompt Categories (for filter dropdown)
# ===================================================================

@router.get("/meta/categories", response_model=List[str])
async def list_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List distinct prompt categories."""
    _assert_admin(ctx)
    rows = db.query(SystemPrompt.category).distinct().all()
    return sorted([r[0] for r in rows])
