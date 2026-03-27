"""
Blog Service
=============

CRUD operations for blog posts. Content stored as .md files in
workspace storage (same pattern as ReportService). Metadata stays
in the blog_posts table; the file_path column points to the .md file.

Falls back to the legacy content column for pre-migration posts.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from core.models.core import BlogPost
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to URL-safe kebab-case slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:480]


def _reading_time(content: str) -> int:
    """Estimate reading time in minutes (200 wpm, minimum 1)."""
    word_count = len(content.split())
    return max(1, math.ceil(word_count / 200))


def _unique_slug(db: Session, workspace_id: UUID, base_slug: str, exclude_id: UUID | None = None) -> str:
    """Return a slug unique within the workspace, appending -2, -3, etc. if needed."""
    slug = base_slug
    suffix = 1
    while True:
        query = db.query(BlogPost.id).filter(
            BlogPost.workspace_id == workspace_id,
            BlogPost.slug == slug,
        )
        if exclude_id:
            query = query.filter(BlogPost.id != exclude_id)
        if query.first() is None:
            return slug
        suffix += 1
        slug = f"{base_slug}-{suffix}"


def _content_file_path(slug: str) -> str:
    """Workspace-relative path for blog content files."""
    return f"content/blog/{slug}.md"


class BlogService:
    """Workspace-scoped blog post CRUD with workspace file storage."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id

    def _ws_client(self) -> WorkspaceClient:
        return WorkspaceClient(str(self.workspace_id))

    # ------------------------------------------------------------------
    # Content I/O (workspace files)
    # ------------------------------------------------------------------
    async def _write_content(self, file_path: str, content: str) -> bool:
        """Write markdown content to workspace file. Returns True on success."""
        result = await self._ws_client().write_file(file_path, content)
        if not result.get("success", False):
            logger.error(
                "[BlogService] Failed to write %s: %s",
                file_path, result.get("error", "unknown"),
            )
            return False
        return True

    async def get_content(self, post: BlogPost) -> str | None:
        """Read full markdown content for a post.

        Reads from workspace file if file_path is set, falls back to
        the legacy content column for pre-migration posts.
        """
        if post.file_path:
            result = await self._ws_client().read_file(post.file_path)
            if result.get("success"):
                return result.get("content", "")
            logger.warning(
                "[BlogService] Failed to read %s, falling back to DB: %s",
                post.file_path, result.get("error", "unknown"),
            )
        # Fallback: legacy content column
        return post.content

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    async def create_post(
        self,
        title: str,
        content: str,
        author_name: str,
        author_agent_id: int | None = None,
        excerpt: str | None = None,
        cover_image_url: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
        status: str = "draft",
        seo_title: str | None = None,
        seo_description: str | None = None,
    ) -> BlogPost:
        slug = _unique_slug(self.db, self.workspace_id, _slugify(title))
        if not excerpt:
            excerpt = content[:300].strip()

        file_path = _content_file_path(slug)
        now = datetime.now(timezone.utc) if status == "published" else None

        # Write .md to workspace
        wrote = await self._write_content(file_path, content)

        post = BlogPost(
            workspace_id=self.workspace_id,
            author_agent_id=author_agent_id,
            author_name=author_name,
            title=title,
            slug=slug,
            excerpt=excerpt,
            content=content[:500] if wrote else content,  # truncated fallback if wrote to file
            file_path=file_path if wrote else None,
            cover_image_url=cover_image_url,
            tags=tags or [],
            category=category,
            status=status,
            published_at=now,
            reading_time_minutes=_reading_time(content),
            seo_title=seo_title,
            seo_description=seo_description,
        )
        self.db.add(post)
        self.db.commit()
        self.db.refresh(post)
        return post

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_post(self, post_id: UUID) -> BlogPost | None:
        return self.db.query(BlogPost).filter(
            BlogPost.id == post_id,
            BlogPost.workspace_id == self.workspace_id,
        ).first()

    def get_post_by_slug(self, slug: str) -> BlogPost | None:
        return self.db.query(BlogPost).filter(
            BlogPost.slug == slug,
            BlogPost.workspace_id == self.workspace_id,
        ).first()

    def list_posts(
        self,
        status: str | None = None,
        category: str | None = None,
        tag: str | None = None,
        page: int = 1,
        per_page: int = 10,
    ) -> dict[str, Any]:
        query = self.db.query(BlogPost).filter(
            BlogPost.workspace_id == self.workspace_id,
        )
        if status:
            query = query.filter(BlogPost.status == status)
        if category:
            query = query.filter(BlogPost.category == category)
        if tag:
            query = query.filter(BlogPost.tags.any(tag))

        total = query.count()
        posts = (
            query
            .order_by(BlogPost.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        return {
            "posts": posts,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, math.ceil(total / per_page)),
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    async def update_post(self, post_id: UUID, **kwargs: Any) -> BlogPost | None:
        post = self.get_post(post_id)
        if not post:
            return None

        # If content is being updated, write to workspace file
        if "content" in kwargs:
            new_content = kwargs["content"]
            file_path = post.file_path or _content_file_path(post.slug)
            wrote = await self._write_content(file_path, new_content)
            if wrote:
                post.file_path = file_path
                kwargs["content"] = new_content[:500]  # truncated fallback in DB
            post.reading_time_minutes = _reading_time(new_content)

        for key, value in kwargs.items():
            if hasattr(post, key):
                setattr(post, key, value)

        if "title" in kwargs and post.status == "draft":
            new_slug = _unique_slug(
                self.db, self.workspace_id, _slugify(post.title), exclude_id=post.id
            )
            # If slug changed, write content to new file path
            if new_slug != post.slug:
                new_file_path = _content_file_path(new_slug)
                content = await self.get_content(post)
                if content:
                    wrote = await self._write_content(new_file_path, content)
                    if wrote:
                        post.file_path = new_file_path
                post.slug = new_slug

        self.db.commit()
        self.db.refresh(post)
        return post

    # ------------------------------------------------------------------
    # Publish / Unpublish
    # ------------------------------------------------------------------
    def publish_post(self, post_id: UUID) -> BlogPost | None:
        post = self.get_post(post_id)
        if not post:
            return None
        post.status = "published"
        post.published_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(post)
        return post

    def unpublish_post(self, post_id: UUID) -> BlogPost | None:
        post = self.get_post(post_id)
        if not post:
            return None
        post.status = "draft"
        post.published_at = None
        self.db.commit()
        self.db.refresh(post)
        return post

    # ------------------------------------------------------------------
    # Delete (soft)
    # ------------------------------------------------------------------
    def delete_post(self, post_id: UUID) -> bool:
        post = self.get_post(post_id)
        if not post:
            return False
        post.status = "archived"
        self.db.commit()
        return True

    # ------------------------------------------------------------------
    # View counting
    # ------------------------------------------------------------------
    def increment_views(self, post_id: UUID) -> None:
        self.db.query(BlogPost).filter(
            BlogPost.id == post_id,
            BlogPost.workspace_id == self.workspace_id,
        ).update({BlogPost.view_count: BlogPost.view_count + 1})
        self.db.commit()

    # ------------------------------------------------------------------
    # Aggregations (for public API)
    # ------------------------------------------------------------------
    def get_categories(self) -> list[dict[str, Any]]:
        rows = (
            self.db.query(BlogPost.category, sa_func.count(BlogPost.id))
            .filter(
                BlogPost.workspace_id == self.workspace_id,
                BlogPost.status == "published",
                BlogPost.category.isnot(None),
            )
            .group_by(BlogPost.category)
            .all()
        )
        return [{"category": cat, "count": cnt} for cat, cnt in rows]

    def get_tags(self) -> list[dict[str, Any]]:
        rows = (
            self.db.query(
                sa_func.unnest(BlogPost.tags).label("tag"),
                sa_func.count(BlogPost.id),
            )
            .filter(
                BlogPost.workspace_id == self.workspace_id,
                BlogPost.status == "published",
            )
            .group_by("tag")
            .all()
        )
        return [{"tag": tag, "count": cnt} for tag, cnt in rows]
