"""
Widget Blog API (Public)
========================

Public read-only endpoints for the embeddable blog widget.
No authentication required — workspace_id is passed as a query parameter.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.services.blog_service import BlogService
from core.utils.markdown_renderer import render_markdown_to_html

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Widget Blog"])


@router.get("/posts")
async def list_published_posts(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    category: str | None = Query(None),
    tag: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """List published blog posts for a workspace (no auth required)."""
    svc = BlogService(db, workspace_id)
    result = svc.list_posts(
        status="published",
        category=category,
        tag=tag,
        page=page,
        per_page=per_page,
    )
    posts = [p.to_dict(include_content=False) for p in result["posts"]]
    response = JSONResponse(content={
        "posts": posts,
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "total_pages": result["total_pages"],
    })
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


@router.get("/posts/{slug}")
async def get_published_post(
    slug: str,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    db: Session = Depends(get_db),
):
    """Get a single published post by slug with HTML content."""
    svc = BlogService(db, workspace_id)
    post = svc.get_post_by_slug(slug)
    if not post or post.status != "published":
        raise HTTPException(status_code=404, detail="Post not found")

    svc.increment_views(post.id)

    data = post.to_dict(include_content=False)
    content = await svc.get_content(post) or ""
    data["content"] = render_markdown_to_html(content)

    response = JSONResponse(content=data)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@router.get("/categories")
async def list_categories(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    db: Session = Depends(get_db),
):
    """List categories with post counts for published posts."""
    svc = BlogService(db, workspace_id)
    categories = svc.get_categories()
    response = JSONResponse(content=categories)
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


@router.get("/tags")
async def list_tags(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    db: Session = Depends(get_db),
):
    """List tags with post counts for published posts."""
    svc = BlogService(db, workspace_id)
    tags = svc.get_tags()
    response = JSONResponse(content=tags)
    response.headers["Cache-Control"] = "public, max-age=300"
    return response
