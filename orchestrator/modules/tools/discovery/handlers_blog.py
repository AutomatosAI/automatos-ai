"""Blog handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def publish_blog_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create and optionally publish a blog post."""
    from core.services.blog_service import BlogService

    title = params.get("title")
    content = params.get("content")
    if not title or not content:
        return {"success": False, "error": "title and content are required"}

    publish_immediately = params.get("publish_immediately", True)
    status = "published" if publish_immediately else "draft"

    agent_name = params.get("_agent_name", "AI Agent")
    agent_id = params.get("_agent_id")

    svc = BlogService(db, workspace_id)
    post = await svc.create_post(
        title=title,
        content=content,
        author_name=agent_name,
        author_agent_id=agent_id,
        excerpt=params.get("excerpt"),
        cover_image_url=params.get("cover_image_url"),
        tags=params.get("tags", []),
        category=params.get("category"),
        status=status,
    )

    return {
        "success": True,
        "post_id": str(post.id),
        "title": post.title,
        "slug": post.slug,
        "status": post.status,
        "file_path": post.file_path,
        "url": f"/api/widgets/blog/posts/{post.slug}?workspace_id={workspace_id}",
    }


async def list_blog_posts(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List blog posts in the workspace."""
    from core.services.blog_service import BlogService

    status = params.get("status", "published")
    limit = min(params.get("limit", 10), 50)
    category = params.get("category")

    svc = BlogService(db, workspace_id)
    result = svc.list_posts(
        status=status,
        category=category,
        page=1,
        per_page=limit,
    )

    posts = [
        {
            "post_id": str(p.id),
            "title": p.title,
            "slug": p.slug,
            "status": p.status,
            "published_at": p.published_at.isoformat() if p.published_at else None,
            "category": p.category,
            "tags": p.tags or [],
            "reading_time_minutes": p.reading_time_minutes,
        }
        for p in result["posts"]
    ]

    return {
        "success": True,
        "posts": posts,
        "total": result["total"],
    }


async def get_blog_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full blog post content by ID or slug."""
    from core.services.blog_service import BlogService

    post_id = params.get("post_id")
    slug = params.get("slug")
    if not post_id and not slug:
        return {"success": False, "error": "post_id or slug is required"}

    svc = BlogService(db, workspace_id)
    if post_id:
        post = svc.get_post(UUID(post_id))
    else:
        post = svc.get_post_by_slug(slug)

    if not post:
        return {"success": False, "error": "Post not found"}

    content = await svc.get_content(post)

    return {
        "success": True,
        "post_id": str(post.id),
        "title": post.title,
        "slug": post.slug,
        "status": post.status,
        "content": content,
        "excerpt": post.excerpt,
        "cover_image_url": post.cover_image_url,
        "tags": post.tags or [],
        "category": post.category,
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "reading_time_minutes": post.reading_time_minutes,
    }


async def update_blog_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update fields on an existing blog post."""
    from core.services.blog_service import BlogService

    post_id = params.get("post_id")
    if not post_id:
        return {"success": False, "error": "post_id is required"}

    # Extract only the updatable fields that were provided
    updatable = ("title", "content", "excerpt", "tags", "category", "cover_image_url")
    updates = {k: params[k] for k in updatable if k in params and params[k] is not None}

    if not updates:
        return {"success": False, "error": "No fields to update"}

    svc = BlogService(db, workspace_id)
    post = await svc.update_post(UUID(post_id), **updates)
    if not post:
        return {"success": False, "error": "Post not found"}

    return {
        "success": True,
        "post_id": str(post.id),
        "title": post.title,
        "slug": post.slug,
        "status": post.status,
    }
