"""
Blog Management API (Authenticated)
====================================

CRUD endpoints for managing blog posts from the dashboard.
All endpoints require workspace auth via get_request_context_hybrid.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.workspaces import Workspace
from core.services.blog_service import BlogService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/blog", tags=["Blog"])


def _resolve_author_name(db: Session, workspace_id) -> str:
    """Public byline for a post: the workspace/brand name.

    ``author_name`` is exposed on public widget endpoints, so it must never be
    user PII. We use the workspace name (e.g. "InBuildUK") and fall back to a
    neutral label when it is missing or blank.
    """
    name = db.query(Workspace.name).filter(Workspace.id == workspace_id).scalar()
    name = (name or "").strip()
    return name or "Workspace Author"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CreatePostRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    excerpt: Optional[str] = Field(None, max_length=500)
    cover_image_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    category: Optional[str] = None
    status: str = Field(default="draft")
    seo_title: Optional[str] = Field(None, max_length=200)
    seo_description: Optional[str] = Field(None, max_length=300)


class UpdatePostRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    excerpt: Optional[str] = Field(None, max_length=500)
    cover_image_url: Optional[str] = None
    tags: Optional[list[str]] = None
    category: Optional[str] = None
    status: Optional[str] = None
    seo_title: Optional[str] = Field(None, max_length=200)
    seo_description: Optional[str] = Field(None, max_length=300)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/posts", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def create_post(
    body: CreatePostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Create a new blog post (draft by default)."""
    svc = BlogService(db, ctx.workspace_id)
    author_name = _resolve_author_name(db, ctx.workspace_id)
    post = await svc.create_post(
        title=body.title,
        content=body.content,
        author_name=author_name,
        excerpt=body.excerpt,
        cover_image_url=body.cover_image_url,
        tags=body.tags,
        category=body.category,
        status=body.status,
        seo_title=body.seo_title,
        seo_description=body.seo_description,
    )
    return post.to_dict(include_content=True)


@router.get("/posts")
async def list_posts(
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List all blog posts in workspace (all statuses)."""
    svc = BlogService(db, ctx.workspace_id)
    result = svc.list_posts(
        status=status,
        category=category,
        tag=tag,
        page=page,
        per_page=per_page,
    )
    return {
        "posts": [p.to_dict(include_content=False) for p in result["posts"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "total_pages": result["total_pages"],
    }


@router.get("/posts/{post_id}")
async def get_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Get full post detail by ID."""
    svc = BlogService(db, ctx.workspace_id)
    post = svc.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    data = post.to_dict(include_content=False)
    data["content"] = await svc.get_content(post)
    return data


@router.put("/posts/{post_id}", dependencies=[Depends(require_workspace_permission("documents:update"))])
async def update_post(
    post_id: UUID,
    body: UpdatePostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Update post fields."""
    svc = BlogService(db, ctx.workspace_id)
    updates = body.dict(exclude_unset=True)
    post = await svc.update_post(post_id, **updates)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post.to_dict(include_content=True)


@router.delete("/posts/{post_id}", dependencies=[Depends(require_workspace_permission("documents:delete"))])
async def delete_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Archive post (soft delete)."""
    svc = BlogService(db, ctx.workspace_id)
    if not svc.delete_post(post_id):
        raise HTTPException(status_code=404, detail="Post not found")
    return {"success": True}


@router.post("/posts/{post_id}/publish", dependencies=[Depends(require_workspace_permission("documents:update"))])
async def publish_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Publish a post."""
    svc = BlogService(db, ctx.workspace_id)
    post = svc.publish_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post.to_dict(include_content=False)


@router.post("/posts/{post_id}/unpublish", dependencies=[Depends(require_workspace_permission("documents:update"))])
async def unpublish_post(
    post_id: UUID,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Unpublish a post (back to draft)."""
    svc = BlogService(db, ctx.workspace_id)
    post = svc.unpublish_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post.to_dict(include_content=False)


# ---------------------------------------------------------------------------
# Create Blog Mission — single entry point used by the "Create Blog" UI button.
# Same code path as the platform_create_blog_post agent tool: builds the
# standardized goal and dispatches to CoordinatorService.
# ---------------------------------------------------------------------------

class CreateBlogMissionRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    category: Optional[str] = Field(None, max_length=100)


_ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
}
_MAX_COVER_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB


@router.post("/cover-image/upload", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def upload_cover_image(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Upload a user-supplied cover image for a blog post. Saves to the same
    image store used by platform_generate_cover_image so the URL pattern is
    identical (/api/generated-images/{image_id}).
    """
    import base64

    from core.services.image_store import get_image_store

    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type {content_type!r}. Allowed: {sorted(_ALLOWED_IMAGE_MIMES)}",
        )

    body_bytes = await file.read()
    if len(body_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(body_bytes) > _MAX_COVER_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(body_bytes)} bytes, max {_MAX_COVER_IMAGE_BYTES})",
        )

    b64 = base64.b64encode(body_bytes).decode("ascii")
    store = get_image_store()
    image_id = await store.save_image(b64, mime_type=content_type, workspace_id=str(ctx.workspace_id))
    return {
        "image_id": image_id,
        "cover_image_url": f"{config.BACKEND_URL.rstrip('/')}/api/generated-images/{image_id}",
        "size_bytes": len(body_bytes),
        "content_type": content_type,
    }


@router.post("/missions", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def create_blog_mission(
    body: CreateBlogMissionRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Fire a research-and-write blog mission for a topic. Same end-to-end
    pipeline whether triggered from the UI button, an agent, or a scheduled
    playbook — they all converge on this code path.
    """
    from modules.tools.discovery.handlers_blog import create_blog_post_from_topic

    user_id = ctx.user.clerk_user_id if ctx.user else None
    params = {
        "topic": body.topic.strip(),
        "category": (body.category or "AI & Automation").strip(),
        "_user_id": user_id,
    }
    result = await create_blog_post_from_topic(db, ctx.workspace_id, params)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Mission failed to start"))
    return result
