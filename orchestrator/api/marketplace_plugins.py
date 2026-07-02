"""
Public Marketplace Plugin API Endpoints (PRD-42: US-010)
========================================================

Provides REST APIs for browsing approved marketplace plugins:
- List approved + active plugins with filters (category, search, tags, sort)
- Get full plugin details including manifest from S3
- Get full plugin content (skills, commands) from S3
- List plugin categories
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import asc, desc, or_
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)


def _is_admin(ctx: RequestContext) -> bool:
    """Check if current user has admin role."""
    # PRD-174 F043: shared admin check — super_admin ⊇ admin when the plane is on.
    from core.auth.roles import caller_is_admin
    return caller_is_admin(ctx.user)

router = APIRouter(prefix="/api/marketplace/plugins", tags=["Marketplace Plugins"])


# ===================================================================
# Pydantic Response Models
# ===================================================================

class CategoryOut(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    sort_order: int = 0
    parent_id: Optional[str] = None

    class Config:
        from_attributes = True


class PluginSummaryOut(BaseModel):
    id: str
    slug: str
    name: str
    version: str
    description: Optional[str] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    tags: Optional[List[str]] = None
    skills_count: int = 0
    commands_count: int = 0
    agents_count: int = 0
    hooks_count: int = 0
    token_estimate: int = 0
    enable_count: int = 0
    is_featured: bool = False
    security_status: Optional[str] = None
    original_author: Optional[str] = None
    license: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class PluginContentItem(BaseModel):
    """A skill, command, or agent inside a plugin."""
    name: str
    description: Optional[str] = None


class PluginDetailOut(BaseModel):
    id: str
    slug: str
    name: str
    version: str
    description: Optional[str] = None
    long_description: Optional[str] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    tags: Optional[List[str]] = None
    skills_count: int = 0
    commands_count: int = 0
    agents_count: int = 0
    hooks_count: int = 0
    token_estimate: int = 0
    recommended_models: Optional[List[str]] = None
    enable_count: int = 0
    is_featured: bool = False
    is_active: bool = True
    approval_status: Optional[str] = None
    security_status: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    original_author: Optional[str] = None
    license: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None
    # Enriched content lists extracted from manifest
    skills: List[PluginContentItem] = Field(default_factory=list)
    commands: List[PluginContentItem] = Field(default_factory=list)
    agents: List[PluginContentItem] = Field(default_factory=list)

    class Config:
        from_attributes = True


class PluginContentOut(BaseModel):
    plugin_id: str
    slug: str
    version: str
    files: Dict[str, str] = Field(default_factory=dict, description="Map of file path to content")


# ===================================================================
# Helpers
# ===================================================================

def _extract_content_items(manifest: Optional[Dict], key: str) -> List[PluginContentItem]:
    """Extract named content items (skills, commands, agents) from a manifest dict.

    Items in the manifest can be plain strings (filenames) or dicts with name/description.
    We normalise them all into PluginContentItem objects with human-friendly names.
    """
    if not manifest:
        return []
    contents = manifest.get("contents", {})
    items = contents.get(key, [])
    if not isinstance(items, list):
        return []

    result: List[PluginContentItem] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name", item.get("slug", ""))
            desc = item.get("description")
        elif isinstance(item, str):
            # Strip file extension and convert kebab-case to title
            name = item.rsplit(".", 1)[0] if "." in item else item
            desc = None
        else:
            continue
        if name:
            result.append(PluginContentItem(
                name=name.replace("-", " ").replace("_", " ").title(),
                description=desc,
            ))
    return result


# ===================================================================
# Endpoints
# ===================================================================

@router.get("", response_model=None)
async def list_plugins(
    category: Optional[str] = Query(None, description="Filter by category slug"),
    search: Optional[str] = Query(None, description="Search plugin name/description"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort: str = Query("popular", description="Sort: popular, newest, name"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List approved and active marketplace plugins with optional filters."""
    try:
        from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory

        query = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.approval_status == "approved",
            MarketplacePlugin.is_active == True,
        )

        # Category filter (by slug) — return empty set if category not found
        if category:
            cat = db.query(PluginCategory).filter(
                PluginCategory.slug == category,
            ).first()
            if cat:
                query = query.filter(MarketplacePlugin.category_id == cat.id)
            else:
                # Unknown category → return nothing rather than ignoring the filter
                query = query.filter(MarketplacePlugin.id == None)

        # Search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    MarketplacePlugin.name.ilike(search_term),
                    MarketplacePlugin.description.ilike(search_term),
                    MarketplacePlugin.slug.ilike(search_term),
                )
            )

        # Tags filter
        if tags:
            for tag in tags:
                query = query.filter(MarketplacePlugin.tags.any(tag))

        # Sorting
        if sort == "popular":
            query = query.order_by(desc(MarketplacePlugin.enable_count))
        elif sort == "newest":
            query = query.order_by(desc(MarketplacePlugin.created_at))
        elif sort == "name":
            query = query.order_by(asc(MarketplacePlugin.name))
        else:
            query = query.order_by(desc(MarketplacePlugin.enable_count))

        total = query.count()
        offset = (page - 1) * limit
        plugins = query.offset(offset).limit(limit).all()

        # Build response with category names
        items = []
        for p in plugins:
            cat_name = None
            if p.category_id:
                cat = db.query(PluginCategory).filter(
                    PluginCategory.id == p.category_id,
                ).first()
                if cat:
                    cat_name = cat.name

            items.append(PluginSummaryOut(
                id=str(p.id),
                slug=p.slug,
                name=p.name,
                version=p.version,
                description=p.description,
                category_id=str(p.category_id) if p.category_id else None,
                category_name=cat_name,
                tags=p.tags or [],
                skills_count=p.skills_count or 0,
                commands_count=p.commands_count or 0,
                agents_count=p.agents_count or 0,
                hooks_count=p.hooks_count or 0,
                token_estimate=p.token_estimate or 0,
                enable_count=p.enable_count or 0,
                is_featured=p.is_featured or False,
                security_status=p.security_status,
                original_author=p.original_author,
                license=p.license,
                created_at=p.created_at.isoformat() if p.created_at else None,
            ))

        return {
            "items": [item.model_dump() for item in items],
            "total": total,
            "page": page,
            "limit": limit,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing marketplace plugins: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/categories", response_model=List[CategoryOut])
async def list_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all plugin categories."""
    try:
        from core.models.marketplace_plugins import PluginCategory

        categories = db.query(PluginCategory).order_by(
            asc(PluginCategory.sort_order),
            asc(PluginCategory.name),
        ).all()

        return [
            CategoryOut(
                id=str(c.id),
                slug=c.slug,
                name=c.name,
                description=c.description,
                icon=c.icon,
                sort_order=c.sort_order or 0,
                parent_id=str(c.parent_id) if c.parent_id else None,
            )
            for c in categories
        ]

    except Exception as e:
        logger.error("Error listing categories: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{plugin_id}", response_model=PluginDetailOut)
async def get_plugin_detail(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full plugin details including contents manifest from S3."""
    try:
        from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory

        # Admins can view any plugin (pending, blocked, etc.) for review.
        # Regular users only see approved plugins.
        query = db.query(MarketplacePlugin).filter(MarketplacePlugin.id == plugin_id)
        if not _is_admin(ctx):
            query = query.filter(MarketplacePlugin.approval_status == "approved")
        plugin = query.first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Get category name
        cat_name = None
        if plugin.category_id:
            cat = db.query(PluginCategory).filter(
                PluginCategory.id == plugin.category_id,
            ).first()
            if cat:
                cat_name = cat.name

        # Try to load manifest from S3
        manifest = None
        try:
            from core.services.marketplace_s3 import MarketplaceS3Service
            s3 = MarketplaceS3Service()
            manifest = await s3.get_manifest(plugin.slug, plugin.version)
        except Exception as manifest_err:
            logger.warning("Could not load manifest for %s@%s: %s", plugin.slug, plugin.version, manifest_err)

        return PluginDetailOut(
            id=str(plugin.id),
            slug=plugin.slug,
            name=plugin.name,
            version=plugin.version,
            description=plugin.description,
            long_description=plugin.long_description,
            category_id=str(plugin.category_id) if plugin.category_id else None,
            category_name=cat_name,
            tags=plugin.tags or [],
            skills_count=plugin.skills_count or 0,
            commands_count=plugin.commands_count or 0,
            agents_count=plugin.agents_count or 0,
            hooks_count=plugin.hooks_count or 0,
            token_estimate=plugin.token_estimate or 0,
            recommended_models=plugin.recommended_models or [],
            enable_count=plugin.enable_count or 0,
            is_featured=plugin.is_featured or False,
            is_active=plugin.is_active if plugin.is_active is not None else True,
            approval_status=plugin.approval_status,
            security_status=plugin.security_status,
            source_type=plugin.source_type,
            source_url=plugin.source_url,
            original_author=plugin.original_author,
            license=plugin.license,
            created_at=plugin.created_at.isoformat() if plugin.created_at else None,
            updated_at=plugin.updated_at.isoformat() if plugin.updated_at else None,
            manifest=manifest,
            skills=_extract_content_items(manifest, "skills"),
            commands=_extract_content_items(manifest, "commands"),
            agents=_extract_content_items(manifest, "agents"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching plugin %s: %s", plugin_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{plugin_id}/content", response_model=PluginContentOut)
async def get_plugin_content(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full plugin content (skills text, commands text) loaded from S3.

    Used by AgentFactory to build the full agent context at runtime.
    """
    try:
        from core.models.marketplace_plugins import MarketplacePlugin

        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
            MarketplacePlugin.approval_status == "approved",
            MarketplacePlugin.is_active == True,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Load all files from S3
        from core.services.marketplace_s3 import MarketplaceS3Service
        s3 = MarketplaceS3Service()

        file_keys = await s3.list_plugin_files(plugin.slug, plugin.version)

        files: Dict[str, str] = {}
        prefix = f"plugins/{plugin.slug}/{plugin.version}/"
        for key in file_keys:
            try:
                content = await s3.get_file(key)
                # Store with relative path (strip the prefix)
                relative_path = key[len(prefix):] if key.startswith(prefix) else key
                files[relative_path] = content
            except Exception as file_err:
                logger.warning("Could not read file %s: %s", key, file_err)

        return PluginContentOut(
            plugin_id=str(plugin.id),
            slug=plugin.slug,
            version=plugin.version,
            files=files,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching plugin content %s: %s", plugin_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")
