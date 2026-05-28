"""
Widget Blog API (Public Read)
==============================

Read-only endpoints for the embeddable blog widget.

Workspace resolution accepts two paths:

1. **Authorization Bearer ak_pub_*** (preferred — used by widget.global.js
   and any new client integration). Workspace is derived from the SDK API
   key, origin is checked against the key's ``allowed_domains``.
2. **``workspace_id`` query param** (back-compat for the existing
   automatos.app /blog page and Shopify themes shipped before unified
   auth). Slated for deprecation once all consumers migrate.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.widgets.auth import _extract_origin
from core.database.database import get_db
from core.services.api_key_service import ApiKeyService
from core.services.blog_service import BlogService
from core.utils.markdown_renderer import render_markdown_to_html

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Widget Blog"])


# ---------------------------------------------------------------------------
# Workspace resolver — accepts auth key OR workspace_id query (back-compat)
# ---------------------------------------------------------------------------

async def resolve_blog_workspace(
    request: Request,
    workspace_id: Optional[UUID] = Query(
        None,
        description="Workspace ID (deprecated — use Authorization Bearer ak_pub_* key instead)",
    ),
    db: Session = Depends(get_db),
) -> UUID:
    """Resolve the target workspace for a blog widget read.

    Order:
      1. ``Authorization: Bearer <ak_pub_*>`` → derive workspace from the key
         and enforce origin allow-list. This is the path used by all new
         widget integrations (script-tag embed, React component, Shopify
         theme blocks shipped after unified auth).
      2. ``?workspace_id=<uuid>`` query — legacy fallback for the
         automatos.app /blog page and any embedded site that hasn't migrated.

    Raises 422 if neither is provided.
    """

    auth_header = request.headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        api_key_record = ApiKeyService.validate_api_key(db, token)
        if api_key_record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )

        origin = _extract_origin(request)
        if origin and not ApiKeyService.check_domain(api_key_record, origin):
            logger.warning(
                "blog widget: origin %s not in allowed_domains for key %s",
                origin,
                api_key_record.key_prefix,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Origin not allowed for this API key",
            )

        if workspace_id and workspace_id != api_key_record.workspace_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="workspace_id query does not match API key workspace",
            )

        return api_key_record.workspace_id

    if workspace_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Provide either an Authorization Bearer ak_pub_* header "
                "(preferred) or a workspace_id query parameter (legacy)."
            ),
        )

    return workspace_id


@router.get("/posts")
async def list_published_posts(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    category: str | None = Query(None),
    tag: str | None = Query(None),
    workspace_id: UUID = Depends(resolve_blog_workspace),
    db: Session = Depends(get_db),
):
    """List published blog posts for a workspace."""
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
    workspace_id: UUID = Depends(resolve_blog_workspace),
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
    workspace_id: UUID = Depends(resolve_blog_workspace),
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
    workspace_id: UUID = Depends(resolve_blog_workspace),
    db: Session = Depends(get_db),
):
    """List tags with post counts for published posts."""
    svc = BlogService(db, workspace_id)
    tags = svc.get_tags()
    response = JSONResponse(content=tags)
    response.headers["Cache-Control"] = "public, max-age=300"
    return response
