"""
PRD-58: Admin Prompts API
==========================

REST endpoints for managing system prompts (admin-only):
- List / filter / search prompts
- View version history
- Create new versions (draft or activate)
- Activate / rollback / delete drafts
- List assessment runs
- Trigger FutureAGI assessment / optimization / safety check
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.system_prompts import (
    EvalRunCreate,
    EvalRunResponse,
    PromptResponse,
    PromptVersionCreate,
    PromptVersionResponse,
    SystemPrompt,
    SystemPromptEvalRun,
    SystemPromptVersion,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/prompts", tags=["Admin Prompts"])


# ===================================================================
# Helpers
# ===================================================================

def _assert_admin(ctx: RequestContext) -> None:
    """Require admin role for Clerk users; allow API key auth (service-to-service)."""
    if ctx.auth_type == "api_key":
        return  # Service-to-service trust
    if ctx.user and getattr(ctx.user, "system_role", "user") == "admin":
        return
    raise HTTPException(status_code=403, detail="Admin access required")


def _prompt_to_response(prompt: SystemPrompt) -> PromptResponse:
    """Convert ORM model to response, denormalizing the active version."""
    active_version = next(
        (v for v in prompt.versions if v.status == "active"),
        None,
    )
    return PromptResponse(
        id=str(prompt.id),
        slug=prompt.slug,
        display_name=prompt.display_name,
        category=prompt.category,
        description=prompt.description,
        variables=prompt.variables,
        is_active=prompt.is_active,
        futureagi_eval_enabled=prompt.futureagi_eval_enabled,
        active_version_number=active_version.version_number if active_version else None,
        active_content=active_version.content if active_version else None,
        active_eval_scores=active_version.eval_scores if active_version else None,
        version_count=len(prompt.versions),
        created_at=prompt.created_at.isoformat(),
        updated_at=prompt.updated_at.isoformat(),
    )


def _version_to_response(v: SystemPromptVersion) -> PromptVersionResponse:
    return PromptVersionResponse(
        id=str(v.id),
        version_number=v.version_number,
        content=v.content,
        change_note=v.change_note,
        status=v.status,
        created_by=v.created_by,
        created_at=v.created_at.isoformat(),
        eval_scores=v.eval_scores,
    )


# ===================================================================
# Endpoints
# ===================================================================

@router.get("", response_model=List[PromptResponse])
def list_prompts(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List all system prompts, optionally filtered by category or search term."""
    _assert_admin(ctx)

    q = db.query(SystemPrompt).order_by(SystemPrompt.category, SystemPrompt.display_name)
    if category:
        q = q.filter(SystemPrompt.category == category)
    if search:
        pattern = f"%{search}%"
        q = q.filter(
            SystemPrompt.display_name.ilike(pattern)
            | SystemPrompt.slug.ilike(pattern)
            | SystemPrompt.description.ilike(pattern)
        )

    prompts = q.all()
    return [_prompt_to_response(p) for p in prompts]


@router.get("/categories")
def list_categories(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List distinct categories with counts."""
    _assert_admin(ctx)
    rows = (
        db.query(SystemPrompt.category, func.count(SystemPrompt.id))
        .group_by(SystemPrompt.category)
        .order_by(SystemPrompt.category)
        .all()
    )
    return [{"category": cat, "count": cnt} for cat, cnt in rows]


@router.get("/{prompt_id}", response_model=PromptResponse)
def get_prompt(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Get a single prompt with its active version."""
    _assert_admin(ctx)
    prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return _prompt_to_response(prompt)


@router.get("/{prompt_id}/versions", response_model=List[PromptVersionResponse])
def list_versions(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List all versions for a prompt (newest first)."""
    _assert_admin(ctx)
    versions = (
        db.query(SystemPromptVersion)
        .filter(SystemPromptVersion.prompt_id == prompt_id)
        .order_by(desc(SystemPromptVersion.version_number))
        .all()
    )
    return [_version_to_response(v) for v in versions]


@router.post("/{prompt_id}/versions", response_model=PromptVersionResponse, status_code=201)
def create_version(
    prompt_id: UUID,
    body: PromptVersionCreate,
    activate: bool = Query(False, description="Immediately activate this version"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Create a new version for a prompt (draft by default, or activate immediately)."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Determine next version number
    max_ver = (
        db.query(func.max(SystemPromptVersion.version_number))
        .filter(SystemPromptVersion.prompt_id == prompt_id)
        .scalar()
    ) or 0

    status = "active" if activate else "draft"

    # If activating, archive the current active version
    if activate:
        db.query(SystemPromptVersion).filter(
            SystemPromptVersion.prompt_id == prompt_id,
            SystemPromptVersion.status == "active",
        ).update({"status": "archived"})

    user_id = str(ctx.user.user_id) if ctx.user else "unknown"
    version = SystemPromptVersion(
        prompt_id=prompt_id,
        version_number=max_ver + 1,
        content=body.content,
        change_note=body.change_note,
        status=status,
        created_by=user_id,
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    # Clear registry cache
    try:
        from core.services.prompt_registry import prompt_registry
        prompt_registry.clear_cache(prompt.slug)
    except Exception:
        pass

    return _version_to_response(version)


@router.post("/{prompt_id}/versions/{version_id}/activate", response_model=PromptVersionResponse)
def activate_version(
    prompt_id: UUID,
    version_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Activate a specific version (archives the current active one)."""
    _assert_admin(ctx)

    version = (
        db.query(SystemPromptVersion)
        .filter(
            SystemPromptVersion.id == version_id,
            SystemPromptVersion.prompt_id == prompt_id,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    # Archive current active
    db.query(SystemPromptVersion).filter(
        SystemPromptVersion.prompt_id == prompt_id,
        SystemPromptVersion.status == "active",
    ).update({"status": "archived"})

    version.status = "active"
    db.commit()
    db.refresh(version)

    # Clear registry cache
    try:
        prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
        from core.services.prompt_registry import prompt_registry
        prompt_registry.clear_cache(prompt.slug if prompt else None)
    except Exception:
        pass

    return _version_to_response(version)


@router.post("/{prompt_id}/rollback", response_model=PromptVersionResponse)
def rollback_version(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Rollback to the previous version (re-activate the most recent archived version)."""
    _assert_admin(ctx)

    # Find the most recent archived version
    prev_version = (
        db.query(SystemPromptVersion)
        .filter(
            SystemPromptVersion.prompt_id == prompt_id,
            SystemPromptVersion.status == "archived",
        )
        .order_by(desc(SystemPromptVersion.version_number))
        .first()
    )
    if not prev_version:
        raise HTTPException(status_code=404, detail="No previous version to rollback to")

    # Archive current active
    db.query(SystemPromptVersion).filter(
        SystemPromptVersion.prompt_id == prompt_id,
        SystemPromptVersion.status == "active",
    ).update({"status": "archived"})

    prev_version.status = "active"
    db.commit()
    db.refresh(prev_version)

    # Clear cache
    try:
        prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
        from core.services.prompt_registry import prompt_registry
        prompt_registry.clear_cache(prompt.slug if prompt else None)
    except Exception:
        pass

    return _version_to_response(prev_version)


@router.delete("/{prompt_id}/versions/{version_id}", status_code=204)
def delete_draft(
    prompt_id: UUID,
    version_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Delete a draft version. Cannot delete active versions."""
    _assert_admin(ctx)

    version = (
        db.query(SystemPromptVersion)
        .filter(
            SystemPromptVersion.id == version_id,
            SystemPromptVersion.prompt_id == prompt_id,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    if version.status == "active":
        raise HTTPException(status_code=400, detail="Cannot delete the active version")

    db.delete(version)
    db.commit()


# ===================================================================
# FutureAGI toggle
# ===================================================================

@router.patch("/{prompt_id}/futureagi-toggle", response_model=PromptResponse)
def toggle_futureagi(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Toggle FutureAGI live traffic scoring on/off for a prompt."""
    _assert_admin(ctx)
    prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    prompt.futureagi_eval_enabled = not prompt.futureagi_eval_enabled
    db.commit()
    db.refresh(prompt)
    state = "enabled" if prompt.futureagi_eval_enabled else "disabled"
    logger.info(f"FutureAGI scoring {state} for {prompt.slug}")
    return _prompt_to_response(prompt)


# ===================================================================
# Assessment endpoints (FutureAGI integration)
# ===================================================================

@router.get("/{prompt_id}/assessment-runs", response_model=List[EvalRunResponse])
def list_assessment_runs(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List FutureAGI assessment runs for a prompt."""
    _assert_admin(ctx)
    runs = (
        db.query(SystemPromptEvalRun)
        .filter(SystemPromptEvalRun.prompt_id == prompt_id)
        .order_by(desc(SystemPromptEvalRun.created_at))
        .all()
    )
    return [
        EvalRunResponse(
            id=str(r.id),
            prompt_id=str(r.prompt_id),
            version_id=str(r.version_id) if r.version_id else None,
            run_type=r.run_type,
            status=r.status,
            scores=r.scores,
            error_message=r.error_message,
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.post("/{prompt_id}/assess", response_model=EvalRunResponse, status_code=201)
def trigger_assessment(
    prompt_id: UUID,
    body: EvalRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Trigger a FutureAGI assessment/optimization/safety run for a prompt version."""
    _assert_admin(ctx)

    prompt = db.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Resolve the version to assess
    version_id = None
    if body.version_id:
        version_id = UUID(body.version_id)
        version = db.query(SystemPromptVersion).filter(SystemPromptVersion.id == version_id).first()
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
    else:
        # Use active version
        version = next((v for v in prompt.versions if v.status == "active"), None)
        if not version:
            raise HTTPException(status_code=400, detail="No active version to assess")
        version_id = version.id

    # Create the run record
    run = SystemPromptEvalRun(
        prompt_id=prompt_id,
        version_id=version_id,
        run_type=body.run_type,
        status="pending",
        metadata_=body.config,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Dispatch assessment via FastAPI BackgroundTasks (works in sync endpoints)
    try:
        from core.services.futureagi_service import futureagi_service

        import asyncio

        def _run_assessment_sync(run_id: str):
            try:
                asyncio.run(futureagi_service.run_assessment(run_id))
            except Exception as e:
                logger.error(f"FutureAGI assessment run {run_id} failed: {e}")

        background_tasks.add_task(_run_assessment_sync, str(run.id))
    except ImportError:
        logger.warning("FutureAGI service not available, assessment run will stay pending")
    except Exception as e:
        logger.warning(f"Could not dispatch assessment run: {e}")

    return EvalRunResponse(
        id=str(run.id),
        prompt_id=str(run.prompt_id),
        version_id=str(run.version_id) if run.version_id else None,
        run_type=run.run_type,
        status=run.status,
        scores=run.scores,
        error_message=run.error_message,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        created_at=run.created_at.isoformat(),
    )
