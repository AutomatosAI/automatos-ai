"""Blog handlers for PlatformActionExecutor."""

import json
import logging
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _normalize_tags(raw) -> List[str]:
    """Normalize tags from LLM input — handles strings, JSON strings, and lists."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if t]
            except (json.JSONDecodeError, ValueError):
                pass
        # Comma-separated string fallback
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list):
        # Ensure all elements are strings (not nested)
        result = []
        for item in raw:
            if isinstance(item, str) and len(item) > 1:
                result.append(item.strip())
            elif isinstance(item, str) and len(item) <= 1:
                # Single char — likely from a broken string iteration, skip
                continue
            else:
                result.append(str(item).strip())
        return result if result else []
    return []


async def publish_blog_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create and optionally publish a blog post."""
    from core.services.blog_service import BlogService

    title = params.get("title")
    content = params.get("content")
    if not title or not content:
        return {"success": False, "error": "title and content are required"}

    publish_immediately = params.get("publish_immediately", False)
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
        tags=_normalize_tags(params.get("tags")),
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
    updatable = ("title", "content", "excerpt", "tags", "category", "cover_image_url", "seo_title", "seo_description")
    updates = {k: params[k] for k in updatable if k in params and params[k] is not None}
    if "tags" in updates:
        updates["tags"] = _normalize_tags(updates["tags"])

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
