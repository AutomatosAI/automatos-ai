"""Blog handlers for PlatformActionExecutor."""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS = 500
PLACEHOLDER_PATTERNS = (
    re.compile(r"^\s*\[[^\[\]]+\]\s*$", re.IGNORECASE),
    re.compile(r"\[(blog post|article|post)\s+content[^\]]*\]", re.IGNORECASE),
    re.compile(r"\[insert[^\]]+\]", re.IGNORECASE),
    re.compile(r"\[your\s+\w+\s+here\]", re.IGNORECASE),
    re.compile(r"\bTODO\b|\bTBD\b|\blorem ipsum\b", re.IGNORECASE),
)


def _validate_content(content: str) -> Optional[str]:
    """Reject placeholder or stub blog content. Returns error message if invalid, None if OK."""
    stripped = (content or "").strip()
    if len(stripped) < MIN_CONTENT_CHARS:
        return (
            f"content too short ({len(stripped)} chars, min {MIN_CONTENT_CHARS}). "
            "Pass the full article body, not a summary or placeholder."
        )
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(stripped):
            return (
                "content appears to contain placeholder text "
                f"(matched pattern: {pattern.pattern!r}). "
                "Pass the FULL article body — actual paragraphs of writing, "
                "not bracketed descriptions of what should go there."
            )
    return None


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

    content_error = _validate_content(content)
    if content_error:
        logger.warning("publish_blog_post rejected: %s | title=%r", content_error, title)
        return {"success": False, "error": content_error}

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

    if "content" in updates:
        content_error = _validate_content(updates["content"])
        if content_error:
            logger.warning("update_blog_post rejected: %s | post_id=%s", content_error, post_id)
            return {"success": False, "error": content_error}

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


# ---------------------------------------------------------------------------
# Standardized blog mission goal — used by platform_create_blog_post so every
# call site (UI button, scheduled playbook, agent suggestion) produces the same
# pipeline. Encodes the content quality bar and the cover-image substep.
# ---------------------------------------------------------------------------

_BLOG_MISSION_GOAL_TEMPLATE = (
    "Research, write, and publish a 1000-2000 word blog post about: {topic} "
    "(category: {category}). Target audience: technical professionals.\n\n"
    "Pipeline: research the topic → write the full article (real prose, not "
    "an outline) → call platform_publish_blog_post(content=full article, "
    "category={category}, publish_immediately=false) → call "
    "platform_generate_cover_image(post_id, prompt) → call platform_create_task "
    "with approval_action.type=publish_blog for human review."
)


async def create_blog_post_from_topic(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Single entry point for blog creation. Builds a standardized mission goal from
    the topic + category, then dispatches it to the mission coordinator. Used by
    the 'Create Blog' UI button, scheduled playbooks, and agents that suggest
    topics. All paths converge on the same pipeline.
    """
    topic = (params.get("topic") or "").strip()
    if not topic:
        return {"success": False, "error": "topic is required"}

    category = (params.get("category") or "AI & Automation").strip()
    goal = _BLOG_MISSION_GOAL_TEMPLATE.format(topic=topic, category=category)
    created_by = str(params.get("_agent_id") or params.get("_user_id") or "system")

    try:
        from services.coordinator_service import CoordinatorService

        coordinator = CoordinatorService()
        run = await coordinator.create_mission(
            db=db,
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=params.get("config") or {},
        )

        plan = run.plan or {}
        tasks = plan.get("tasks", [])
        return {
            "success": True,
            "mission_id": run.id,
            "state": run.state,
            "topic": topic,
            "category": category,
            "task_count": len(tasks),
            "message": (
                f"Blog mission {run.id} created for topic '{topic}'. "
                f"{len(tasks)} tasks queued — research, write, publish, cover image, review."
            ),
        }

    except Exception as e:
        logger.error("create_blog_post_from_topic failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to create blog mission: {str(e)[:300]}"}


# ---------------------------------------------------------------------------
# platform_generate_cover_image — wraps Gemini Nano Banana Pro server-side.
# Generates → uploads via image_store → updates blog_posts.cover_image_url.
# CANVAS (or any agent) calls this with a single tool call.
# ---------------------------------------------------------------------------

_DATA_URL_RE = re.compile(
    r"data:(image/(?:png|jpeg|jpg|webp|gif));base64,([A-Za-z0-9+/=\n\r]+)"
)


def _extract_image_from_response(content: str) -> Optional[tuple]:
    """Find the first base64 image in an LLM response. Returns (mime, b64) or None."""
    if not content:
        return None
    match = _DATA_URL_RE.search(content)
    if not match:
        return None
    mime = match.group(1)
    # Strip whitespace/newlines from the base64 chunk
    b64 = re.sub(r"\s+", "", match.group(2))
    return mime, b64


async def generate_cover_image(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a blog cover image via Gemini Nano Banana Pro and attach it to the
    post in one tool call. Saves the image to the standard image_store (S3 or
    local) and updates blog_posts.cover_image_url.
    """
    from core.services.blog_service import BlogService
    from core.services.image_store import get_image_store
    from core.llm import create_llm_manager

    post_id = params.get("post_id")
    prompt = (params.get("prompt") or "").strip()
    if not post_id or not prompt:
        return {"success": False, "error": "post_id and prompt are required"}

    svc = BlogService(db, workspace_id)
    try:
        post = svc.get_post(UUID(str(post_id)))
    except (ValueError, AttributeError):
        return {"success": False, "error": "Invalid post_id"}
    if not post:
        return {"success": False, "error": f"Post {post_id} not found"}

    full_prompt = (
        f"Generate a 16:9 cover image for a blog post titled: '{post.title}'. "
        f"Image direction: {prompt}. "
        "Style: abstract/conceptual, modern and clean, no embedded text "
        "(title overlay handled by CSS). Output the image only."
    )

    try:
        from config import config

        cover_model = config.BLOG_COVER_MODEL
        llm = create_llm_manager(
            service_name="blog_cover_gen",
            provider="openrouter",
            model=cover_model,
            workspace_id=str(workspace_id),
            request_type="cover_image",
        )
        response = await llm.generate_response(
            messages=[{"role": "user", "content": full_prompt}],
        )
    except Exception as e:
        logger.error("Cover image LLM call failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Image generation failed: {str(e)[:200]}"}

    extracted = _extract_image_from_response(getattr(response, "content", "") or "")
    if not extracted:
        logger.warning(
            "Cover image generation returned no image data | post_id=%s | content_len=%d",
            post_id, len(getattr(response, "content", "") or ""),
        )
        return {
            "success": False,
            "error": "Image model did not return base64 image data — try a clearer prompt",
        }
    mime, b64 = extracted

    try:
        store = get_image_store()
        image_id = await store.save_image(b64, mime_type=mime, workspace_id=str(workspace_id))
    except Exception as e:
        logger.error("Image store save failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Image upload failed: {str(e)[:200]}"}

    cover_url = f"/api/generated-images/{image_id}"

    try:
        updated = await svc.update_post(post.id, cover_image_url=cover_url)
    except Exception as e:
        logger.error("Update post cover failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Cover saved but post update failed: {str(e)[:200]}"}

    if not updated:
        return {"success": False, "error": "Post not found during cover attach"}

    return {
        "success": True,
        "post_id": str(updated.id),
        "title": updated.title,
        "slug": updated.slug,
        "cover_image_url": cover_url,
        "image_id": image_id,
        "message": f"Cover image generated and attached to post '{updated.title}'.",
    }
