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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.services.blog_service import BlogService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/blog", tags=["Blog"])


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

@router.post("/posts")
async def create_post(
    body: CreatePostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Create a new blog post (draft by default)."""
    svc = BlogService(db, ctx.workspace_id)
    author_name = ctx.user.display_name if ctx.user and ctx.user.display_name else "Workspace Author"
    post = svc.create_post(
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
    return post.to_dict(include_content=True)


@router.put("/posts/{post_id}")
async def update_post(
    post_id: UUID,
    body: UpdatePostRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Update post fields."""
    svc = BlogService(db, ctx.workspace_id)
    updates = body.dict(exclude_unset=True)
    post = svc.update_post(post_id, **updates)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post.to_dict(include_content=True)


@router.delete("/posts/{post_id}")
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


@router.post("/posts/{post_id}/publish")
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


@router.post("/posts/{post_id}/unpublish")
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
