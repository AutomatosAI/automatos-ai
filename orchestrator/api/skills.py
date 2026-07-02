
"""
Skills Management API - PRD-22 Implementation
=============================================

Complete API endpoints for skill source management, skill CRUD,
agent-skill assignment, and skill execution.

INTEGRATION NOTE:
- This extends your existing orchestrator/api/skills.py
- Add these routes to your FastAPI router
- Update imports to match your project structure
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import tempfile
import shutil
from pathlib import Path
import zipfile
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.security.git_sanitizer import validate_branch
from core.security.rate_limiter import check_rate_limit

# Adjust imports to match your project structure
from core.database.database import get_db
from core.models import (
    Skill, Agent, SkillSource, SkillFile, SkillVersion, SkillAuditLog,
    agent_skills
)
from sqlalchemy import text, or_
from pydantic import BaseModel, Field
from enum import Enum
from modules.agents.services.skill_loader import get_skill_loader
# PRD-22: Keep recommendations local to skills API to avoid coupling to agent factory

# Minimal PRD-22 request/response types (inlined to avoid external dependency)
class SkillLoadLevel(int, Enum):
    METADATA = 1
    CORE = 2
    RESOURCE = 3

class GitSkillSourceCreate(BaseModel):
    git_url: str = Field(..., description="Git repository URL")
    source_name: Optional[str] = Field(None, description="Friendly name for this source")
    branch: str = Field("main", description="Git branch to clone")
    auto_update: bool = Field(False, description="Enable automatic updates")
    update_frequency_hours: int = Field(24, description="Update frequency in hours")

class SkillSourceResponse(BaseModel):
    id: int
    source_name: str
    source_type: str
    git_url: Optional[str]
    git_branch: Optional[str]
    git_commit_sha: Optional[str]
    local_cache_path: str
    auto_update: bool
    skills_discovered: int
    status: str
    last_sync_at: Optional[datetime]
    last_sync_status: Optional[str]
    last_sync_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class SkillFileInfo(BaseModel):
    file_path: str
    file_type: str
    content_summary: Optional[str]
    file_size_bytes: Optional[int]
    estimated_tokens: Optional[int]
    load_level: int

class SkillContentResponse(BaseModel):
    skill_id: int
    skill_name: str
    load_level: int
    content: str
    estimated_tokens: int
    files_available: List[str]

class EnhancedSkillResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    skill_type: str
    category: str
    skill_version: Optional[str]
    skill_source: Optional[str]
    git_repo_url: Optional[str]
    git_commit_sha: Optional[str]
    filesystem_path: Optional[str]
    tags: List[str] = []
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_sync_at: Optional[datetime]
    files: List[SkillFileInfo] = []
    # PRD-71: Token estimate for budget warnings
    estimated_tokens: int = 0
    class Config:
        from_attributes = True

class AgentSkillsResponse(BaseModel):
    skills: List[EnhancedSkillResponse]
    total_estimated_tokens: int = 0
    warning: Optional[str] = None

class SkillRecommendationRequest(BaseModel):
    task_description: str
    task_type: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)

class SkillRecommendation(BaseModel):
    skill_id: int
    skill_name: str
    confidence: float
    reasoning: str

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/skills", tags=["skills-prd22"])


def _assert_admin(ctx: RequestContext) -> None:
    """Raise 403 if the current user is not an admin."""
    if ctx.auth_type == "clerk":
        role = getattr(ctx.user, "system_role", None) or getattr(ctx.user, "role", None)
        if role not in ("admin", "super_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")


# ---------------------------------------------------------------------------
# PRD-172 F002: tenant isolation for skills + agent-skill attach.
#
# A Skill with workspace_id IS NULL is a global/marketplace skill (readable by
# every workspace, deletable only by super-admin). A Skill with a UUID belongs
# to exactly one workspace. Agents are always workspace-scoped (non-nullable).
# These helpers replace the previous "ctx is accepted but never used" pattern.
# ---------------------------------------------------------------------------

def _is_super_admin(ctx: RequestContext) -> bool:
    """True for cross-workspace admins (mirrors core.auth.super_admin semantics).

    An admin operating with the ``__all__`` sentinel (``admin_all_workspaces``)
    or literally ``system_role == 'super_admin'`` may reach any workspace's
    skills. Fail-closed: anything else is workspace-scoped.
    """
    if getattr(ctx, "admin_all_workspaces", False):
        return True
    return getattr(getattr(ctx, "user", None), "system_role", None) == "super_admin"


def _skill_visible_to(skill: "Skill", ctx: RequestContext) -> bool:
    """True if the caller may read/attach *skill*.

    Global skills (``workspace_id IS NULL``) are visible to all workspaces;
    workspace-owned skills only to their owning workspace (or a super-admin).
    """
    if _is_super_admin(ctx):
        return True
    if skill.workspace_id is None:
        return True
    return skill.workspace_id == ctx.workspace_id


def _assert_agent_in_workspace(agent: "Agent", ctx: RequestContext) -> None:
    """Raise 404 if *agent* is not owned by the caller's workspace.

    404 (not 403) so the endpoint does not confirm the existence of another
    workspace's agent — it looks identical to "no such agent".
    """
    if _is_super_admin(ctx):
        return
    if agent.workspace_id != ctx.workspace_id:
        raise HTTPException(status_code=404, detail="Agent not found")


# ----------------------------------------------------------------------------
# PRD-22: Local skill recommendation helper
# ----------------------------------------------------------------------------
def recommend_skills_for_task(
    task_description: str,
    task_type: Optional[str] = None,
    limit: int = 5,
    db: Session = None
) -> List[Dict[str, Any]]:
    """
    Recommend relevant skills for a task using a simple lexical scoring approach.
    Keeps PRD-22 self-contained without depending on agent_factory.
    """
    if not db:
        return []

    text = (task_description or "").lower()

    # Fetch active skills, optionally filter by category/task_type
    query = db.query(Skill).filter(Skill.is_active == True)
    if task_type:
        query = query.filter(Skill.category == task_type)
    skills = query.all()

    results: List[Dict[str, Any]] = []
    for s in skills:
        name = (s.name or "").lower()
        desc = (s.description or "").lower()
        tags = [t.lower() for t in (s.tags or []) if isinstance(t, str)]

        score = 0.0
        # Simple keyword presence boosts
        if name and any(tok in text for tok in name.split()):
            score += 0.5
        if desc and any(tok in text for tok in desc.split()):
            score += 0.35
        if tags and any(tag in text for tag in tags):
            score += 0.25

        # Small bonus if task_type matches category
        if task_type and s.category and s.category.lower() == task_type.lower():
            score += 0.2

        if score > 0:
            results.append({
                "skill_id": s.id,
                "skill_name": s.name,
                "confidence": min(0.99, round(score, 2)),
                "reasoning": "Matched by name/description/tags{}".format(
                    " and category" if task_type else ""
                )
            })

    # Sort and trim
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results[: max(1, limit)]

# ============================================================================
# Skill Source Management Endpoints
# ============================================================================

@router.post("/sources/git", response_model=Dict[str, Any])
async def import_git_repository(
    source_data: GitSkillSourceCreate,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Import skills from a Git repository.
    
    This clones the repository, indexes all SKILL.md files, and creates
    skill records in the database.
    
    Example:
        POST /api/v1/skills/sources/git
        {
            "git_url": "https://github.com/anthropics/skills.git",
            "source_name": "anthropic-official",
            "branch": "main",
            "auto_update": true
        }
    """
    # PRD-70 FIX-01: Skills import is admin-only until a safe user-facing flow
    # with full security scan + approval workflow is built.
    _assert_admin(ctx)

    # PRD-70 FIX-07: Rate limit git clone operations per workspace
    ws_id = str(getattr(ctx, "workspace_id", "global"))
    await check_rate_limit(ws_id, "skill_import")

    # Validate branch to prevent argument injection (--upload-pack, etc.)
    if source_data.branch:
        ok, err = validate_branch(source_data.branch)
        if not ok:
            raise HTTPException(status_code=400, detail=f"Invalid branch: {err}")

    try:
        skill_loader = get_skill_loader(db)

        # Import repository (this may take a while for large repos)
        result = skill_loader.add_git_repository(
            git_url=source_data.git_url,
            source_name=source_data.source_name,
            branch=source_data.branch,
            auto_update=source_data.auto_update,
            db=db
        )
        
        if result["status"] == "error":
            error_detail = result.get("error", "Unknown error")
            logger.warning("Git import failed: %s (git_url=%s)", error_detail, source_data.git_url)
            raise HTTPException(status_code=400, detail=error_detail)
        
        return {
            "message": f"Successfully imported {result['skills_discovered']} skills from Git repository",
            "source_id": result["source_id"],
            "source_name": result["source_name"],
            "skills_discovered": result["skills_discovered"],
            "skills": result["skills"],
            "commit_sha": result.get("commit_sha")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing Git repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sources", response_model=List[SkillSourceResponse])
async def list_skill_sources(
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all skill sources.
    
    Example:
        GET /api/v1/skills/sources
        GET /api/v1/skills/sources?source_type=git&status=active
    """
    try:
        query = db.query(SkillSource)
        
        if source_type:
            query = query.filter(SkillSource.source_type == source_type)
        
        if status:
            query = query.filter(SkillSource.status == status)
        
        sources = query.order_by(SkillSource.created_at.desc()).all()
        
        return sources
        
    except Exception as e:
        logger.error(f"Error listing skill sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sources/{source_id}", response_model=SkillSourceResponse)
async def get_skill_source(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific skill source.
    
    Example:
        GET /api/v1/skills/sources/1
    """
    try:
        source = db.query(SkillSource).filter(SkillSource.id == source_id).first()
        
        if not source:
            raise HTTPException(status_code=404, detail="Skill source not found")
        
        return source
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skill source: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sources/{source_id}/update", response_model=Dict[str, Any])
async def update_git_repository(
    source_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Update a Git repository (git pull) and re-index skills.
    
    Example:
        POST /api/v1/skills/sources/1/update
    """
    try:
        # Get source
        source = db.query(SkillSource).filter(SkillSource.id == source_id).first()
        
        if not source:
            raise HTTPException(status_code=404, detail="Skill source not found")
        
        if source.source_type != "git":
            raise HTTPException(status_code=400, detail="Source is not a Git repository")
        
        # Update repository
        skill_loader = get_skill_loader(db)
        result = skill_loader.update_git_repository(source.source_name, db=db)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "message": "Repository updated successfully",
            "changes": result["changes"],
            "old_commit": result["old_commit"],
            "new_commit": result["new_commit"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating Git repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sources/{source_id}/rollback", response_model=Dict[str, Any])
async def rollback_git_repository(
    source_id: int,
    commit_sha: str = Query(..., description="Commit SHA to rollback to"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Rollback a Git repository to a specific commit.
    
    Example:
        POST /api/v1/skills/sources/1/rollback?commit_sha=abc123
    """
    try:
        # Get source
        source = db.query(SkillSource).filter(SkillSource.id == source_id).first()
        
        if not source:
            raise HTTPException(status_code=404, detail="Skill source not found")
        
        if source.source_type != "git":
            raise HTTPException(status_code=400, detail="Source is not a Git repository")
        
        # Rollback
        skill_loader = get_skill_loader(db)
        result = skill_loader.rollback_git_repository(source.source_name, commit_sha, db=db)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "message": f"Repository rolled back to {commit_sha}",
            "commit_sha": commit_sha
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back Git repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/sources/{source_id}")
async def deactivate_skill_source(
    source_id: int,
    deactivate_skills: bool = Query(False, description="Also deactivate associated skills"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Deactivate a skill source (soft delete).
    
    Example:
        DELETE /api/v1/skills/sources/1?deactivate_skills=true
    """
    try:
        source = db.query(SkillSource).filter(SkillSource.id == source_id).first()
        
        if not source:
            raise HTTPException(status_code=404, detail="Skill source not found")
        
        # Deactivate source
        source.status = "inactive"
        
        # Optionally deactivate skills
        if deactivate_skills:
            db.query(Skill).filter(
                Skill.skill_source == str(source_id)
            ).update({"is_active": False})
        
        db.commit()
        
        return {"message": "Skill source deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating skill source: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Skill Management Endpoints
# ============================================================================

@router.get("", response_model=List[EnhancedSkillResponse])
async def list_skills(
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List skills with filtering and pagination.
    
    Example:
        GET /api/v1/skills?category=development&limit=20
        GET /api/v1/skills?search=security&tags=authentication,authorization
    """
    try:
        # Marketplace endpoint: only surface system-owned skills (workspace_id IS NULL).
        # Workspace-owned skills (forks, custom) are listed via /workspaces/{id}/skills,
        # which already shadows marketplace originals when a fork exists. Admins see all.
        # PRD-174 F043: shared admin check — super_admin ⊇ admin when the plane is on.
        from core.auth.roles import caller_is_admin
        is_admin = caller_is_admin(ctx.user)
        query = db.query(Skill).filter(Skill.is_active == True)
        if not is_admin:
            query = query.filter(Skill.workspace_id.is_(None))

        # Apply filters
        if category:
            query = query.filter(Skill.category == category)
        
        if source:
            query = query.filter(Skill.skill_source == source)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (Skill.name.ilike(search_pattern)) |
                (Skill.description.ilike(search_pattern))
            )
        
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            # PostgreSQL JSON contains check
            for tag in tag_list:
                query = query.filter(Skill.tags.contains([tag]))
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        skills = query.order_by(Skill.created_at.desc()).offset(offset).limit(limit).all()
        
        # Build enhanced responses with file info
        result = []
        for skill in skills:
            skill_dict = EnhancedSkillResponse(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                skill_type=skill.skill_type,
                category=getattr(skill, 'category', None) or skill.skill_type,  # Fallback to skill_type if category missing
                skill_version=skill.skill_version,
                skill_source=skill.skill_source,
                git_repo_url=skill.git_repo_url,
                git_commit_sha=skill.git_commit_sha,
                filesystem_path=skill.filesystem_path,
                tags=skill.tags or [],
                is_active=skill.is_active,
                created_at=skill.created_at,
                updated_at=skill.updated_at,
                last_sync_at=skill.last_sync_at,
                files=[
                    SkillFileInfo(
                        file_path=f.file_path,
                        file_type=f.file_type,
                        content_summary=f.content_summary,
                        file_size_bytes=f.file_size_bytes,
                        estimated_tokens=f.estimated_tokens,
                        load_level=f.load_level
                    )
                    for f in skill.files
                ] if hasattr(skill, 'files') else []
            )
            result.append(skill_dict)
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing skills: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{skill_id}", response_model=EnhancedSkillResponse)
async def get_skill_details(
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific skill.
    
    Example:
        GET /api/v1/skills/123
    """
    try:
        skill = db.query(Skill).filter(Skill.id == skill_id).first()

        # PRD-172 F002: workspace-owned skills are invisible to other workspaces.
        # 404 (not 403) so the endpoint doesn't confirm another tenant's skill exists.
        if not skill or not _skill_visible_to(skill, ctx):
            raise HTTPException(status_code=404, detail="Skill not found")

        return EnhancedSkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=getattr(skill, 'category', None) or skill.skill_type,  # Fallback to skill_type if category missing
            skill_version=skill.skill_version,
            skill_source=skill.skill_source,
            git_repo_url=skill.git_repo_url,
            git_commit_sha=skill.git_commit_sha,
            filesystem_path=skill.filesystem_path,
            tags=skill.tags or [],
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            last_sync_at=skill.last_sync_at,
            files=[
                SkillFileInfo(
                    file_path=f.file_path,
                    file_type=f.file_type,
                    content_summary=f.content_summary,
                    file_size_bytes=f.file_size_bytes,
                    estimated_tokens=f.estimated_tokens,
                    load_level=f.load_level
                )
                for f in skill.files
            ] if hasattr(skill, 'files') else []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skill details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{skill_id}/content", response_model=SkillContentResponse)
async def get_skill_content(
    skill_id: int,
    level: int = Query(2, ge=1, le=3, description="Load level: 1=metadata, 2=core, 3=resource"),
    resource_path: Optional[str] = Query(None, description="Resource path for level 3"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get skill content at different load levels (progressive disclosure).
    
    Examples:
        GET /api/v1/skills/123/content?level=1  # Metadata only
        GET /api/v1/skills/123/content?level=2  # Core content
        GET /api/v1/skills/123/content?level=3&resource_path=advanced.md  # Specific resource
    """
    try:
        skill = db.query(Skill).filter(Skill.id == skill_id).first()

        # PRD-172 F002: don't serve another workspace's private SKILL.md content.
        if not skill or not _skill_visible_to(skill, ctx):
            raise HTTPException(status_code=404, detail="Skill not found")

        skill_loader = get_skill_loader(db)
        content = ""
        estimated_tokens = 0
        
        if level == 1:
            # Level 1: Metadata
            metadata = skill_loader.load_skill_metadata(skill.name, db=db)
            if metadata:
                content = str(metadata)
                estimated_tokens = 50  # Rough estimate for metadata
        
        elif level == 2:
            # Level 2: Core content
            core_content = skill_loader.load_skill_core(skill.name, db=db)
            if core_content:
                content = core_content
                from modules.agents.services.skill_loader import estimate_tokens
                estimated_tokens = estimate_tokens(content)
        
        elif level == 3:
            # Level 3: Resource
            if not resource_path:
                raise HTTPException(status_code=400, detail="resource_path required for level 3")
            
            resource_content = skill_loader.load_skill_resource(skill.name, resource_path, db=db)
            if resource_content:
                content = resource_content
                from modules.agents.services.skill_loader import estimate_tokens
                estimated_tokens = estimate_tokens(content)
            else:
                raise HTTPException(status_code=404, detail=f"Resource not found: {resource_path}")
        
        # Get available files
        files_available = []
        if hasattr(skill, 'files'):
            files_available = [f.file_path for f in skill.files]
        
        return SkillContentResponse(
            skill_id=skill.id,
            skill_name=skill.name,
            load_level=level,
            content=content,
            estimated_tokens=estimated_tokens,
            files_available=files_available
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skill content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/admin/cleanup-old-mappings")  # Admin endpoint to avoid route conflicts
async def cleanup_old_skill_mappings(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Remove all skill-agent mappings for skills with source='Unknown' (legacy skills).
    This helps clean up old mappings before migrating to PRD-22 skills.
    """
    try:
        # Find all skills with source='Unknown'
        old_skills = db.query(Skill).filter(
            or_(
                Skill.skill_source == 'Unknown',
                Skill.skill_source.is_(None)
            )
        ).all()
        
        if not old_skills:
            return {
                "message": "No old skill mappings found",
                "removed_count": 0
            }
        
        old_skill_ids = [s.id for s in old_skills]
        
        # Delete agent_skills mappings for these old skills (use IN clause for compatibility)
        if old_skill_ids:
            placeholders = ','.join([f':id{i}' for i in range(len(old_skill_ids))])
            params = {f'id{i}': skill_id for i, skill_id in enumerate(old_skill_ids)}
            # SAFETY: placeholders contains only generated `:idN` bind-parameter names, not user input
            delete_sql = "DELETE FROM agent_skills WHERE skill_id IN (" + placeholders + ")"
            deleted_count = db.execute(
                text(delete_sql),
                params
            ).rowcount
        else:
            deleted_count = 0
        
        db.commit()
        
        return {
            "message": f"Removed {deleted_count} old skill-agent mappings",
            "removed_count": deleted_count,
            "old_skill_ids": old_skill_ids
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error cleaning up old skill mappings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{skill_id}")
async def deactivate_skill(
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Deactivate a skill (soft delete).
    
    Example:
        DELETE /api/v1/skills/123
    """
    try:
        skill = db.query(Skill).filter(Skill.id == skill_id).first()

        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")

        # PRD-172 F002: a global builtin-core skill (workspace_id IS NULL) is
        # shared by every tenant — deactivating it lobotomises every workspace's
        # Auto. Only a super-admin may delete a global skill. A workspace-scoped
        # caller may delete ONLY a skill its own workspace owns; another
        # workspace's private skill is 404 (existence not confirmed).
        if not _is_super_admin(ctx):
            if skill.workspace_id is None:
                raise HTTPException(
                    status_code=403,
                    detail="Global skills can only be deleted by a super-admin",
                )
            if skill.workspace_id != ctx.workspace_id:
                raise HTTPException(status_code=404, detail="Skill not found")

        skill.is_active = False

        # Remove from all agent assignments
        db.query(agent_skills).filter(agent_skills.c.skill_id == skill_id).delete()

        db.commit()

        return {"message": "Skill deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating skill: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Agent-Skill Assignment Endpoints
# ============================================================================

@router.get("/agents/{agent_id}/skills", response_model=AgentSkillsResponse)
async def get_agent_skills(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get skills assigned to a specific agent.
    
    Example:
        GET /api/v1/skills/agents/456/skills
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # PRD-172 F002: don't expose another workspace's agent-skill roster.
        _assert_agent_in_workspace(agent, ctx)

        TOKEN_WARNING_THRESHOLD = 15_000

        skills_out = []
        total_tokens = 0
        for skill in agent.skills:
            est = len(skill.prompt_template or "") // 4 if skill.prompt_template else 0
            total_tokens += est
            skills_out.append(
                EnhancedSkillResponse(
                    id=skill.id,
                    name=skill.name,
                    description=skill.description,
                    skill_type=skill.skill_type,
                    category=getattr(skill, 'category', None) or skill.skill_type,
                    skill_version=skill.skill_version,
                    skill_source=skill.skill_source,
                    git_repo_url=skill.git_repo_url,
                    git_commit_sha=skill.git_commit_sha,
                    filesystem_path=skill.filesystem_path,
                    tags=skill.tags or [],
                    is_active=skill.is_active,
                    created_at=skill.created_at,
                    updated_at=skill.updated_at,
                    last_sync_at=skill.last_sync_at,
                    files=[],
                    estimated_tokens=est,
                )
            )

        warning = None
        if total_tokens > TOKEN_WARNING_THRESHOLD:
            warning = (
                f"Combined skill tokens ({total_tokens:,}) exceed recommended "
                f"threshold ({TOKEN_WARNING_THRESHOLD:,}). Consider removing "
                f"lower-priority skills to leave room for conversation context."
            )

        return {
            "skills": skills_out,
            "total_estimated_tokens": total_tokens,
            "warning": warning,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent skills: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/agents/{agent_id}/skills")
async def assign_skills_to_agent(
    agent_id: int,
    skill_ids: List[int],
    replace: bool = Query(False, description="Replace existing skills instead of adding"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Assign skills to an agent.
    
    Example:
        POST /api/v1/skills/agents/456/skills
        {
            "skill_ids": [1, 2, 3]
        }
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # PRD-172 F002: the caller may only attach skills to an agent its own
        # workspace owns — otherwise a workspace could graft its skills onto
        # another tenant's agent.
        _assert_agent_in_workspace(agent, ctx)

        # Verify skills exist. PRD-172 F002: only global skills (workspace_id
        # IS NULL) or skills owned by the caller's workspace are attachable —
        # never another workspace's private SKILL.md (prompt-injection /
        # exfiltration vector). Super-admins may attach any skill.
        skill_query = db.query(Skill).filter(
            Skill.id.in_(skill_ids),
            Skill.is_active == True
        )
        if not _is_super_admin(ctx):
            skill_query = skill_query.filter(
                or_(
                    Skill.workspace_id.is_(None),
                    Skill.workspace_id == ctx.workspace_id,
                )
            )
        skills = skill_query.all()

        if len(skills) != len(skill_ids):
            raise HTTPException(status_code=400, detail="One or more skills not found or inactive")

        # Assign skills
        if replace:
            agent.skills = skills
        else:
            # Add only new skills (avoid duplicates)
            existing_skill_ids = {s.id for s in agent.skills}
            new_skills = [s for s in skills if s.id not in existing_skill_ids]
            agent.skills.extend(new_skills)

        db.commit()

        # PRD-71: Calculate token budget warning
        total_tokens = sum(
            len(s.prompt_template or "") // 4
            for s in agent.skills
            if s.prompt_template
        )
        warning = None
        if total_tokens > 15_000:
            warning = (
                f"Combined skill tokens ({total_tokens:,}) exceed 15,000. "
                f"Consider reducing the number of skills for optimal performance."
            )

        return {
            "message": f"Successfully assigned {len(skills)} skills to agent",
            "agent_id": agent_id,
            "skill_ids": skill_ids,
            "total_estimated_tokens": total_tokens,
            "warning": warning,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning skills to agent: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/agents/{agent_id}/skills")
async def remove_skills_from_agent(
    agent_id: int,
    skill_ids: List[int] = Query(..., description="Skill IDs to remove"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Remove skills from an agent.
    
    Example:
        DELETE /api/v1/skills/agents/456/skills?skill_ids=1&skill_ids=2
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # PRD-172 F002: only mutate skills on an agent the caller's workspace owns.
        _assert_agent_in_workspace(agent, ctx)

        # Remove skills
        agent.skills = [s for s in agent.skills if s.id not in skill_ids]
        
        db.commit()
        
        return {
            "message": f"Successfully removed {len(skill_ids)} skills from agent",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing skills from agent: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Skill Recommendation Endpoint
# ============================================================================

@router.post("/recommend", response_model=List[SkillRecommendation])
async def recommend_skills(
    request: SkillRecommendationRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get AI-recommended skills for a task.
    
    Example:
        POST /api/v1/skills/recommend
        {
            "task_description": "I need to analyze and refactor legacy Python code",
            "task_type": "development",
            "limit": 5
        }
    """
    try:
        recommendations = recommend_skills_for_task(
            task_description=request.task_description,
            task_type=request.task_type,
            limit=request.limit,
            db=db
        )
        
        return [
            SkillRecommendation(
                skill_id=r["skill_id"],
                skill_name=r["skill_name"],
                confidence=r["confidence"],
                reasoning=r["reasoning"]
            )
            for r in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Error recommending skills: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Integration Notes
# ============================================================================

"""
INTEGRATION STEPS:

1. Add this router to your main FastAPI app:
   
   from api.skills_prd22 import router as skills_prd22_router
   app.include_router(skills_prd22_router)

2. Make sure all model imports match your project structure

3. Test endpoints:
   - Import Git repo: POST /api/v1/skills/sources/git
   - List skills: GET /api/v1/skills
   - Assign to agent: POST /api/v1/skills/agents/{id}/skills
   
4. Add authentication/authorization middleware as needed

5. Monitor performance and add caching if needed
"""
